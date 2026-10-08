import pandas as pd
import numpy as np
import networkx as nx
from collections import deque
from utils import RandomWalkUtils
from .temporal_graph import TemporalGraph

class ATDGEB_Graph(TemporalGraph):
    """
    << now >>
    효율성을 위한 다음 기법들은 추후에 개발한다.
        - Transform to time-expanded static graph
        - Alias sampling

    Local Structure Biased Sampling (LSBS)
    -> '현재 노드에서 어떤 이웃을 방문할 것인가?' 를 결정하는 이웃 샘플링 전략
    LSBS 실행 순서
        1. generate_init_local_struct_vec() -> self.init_stru
        2. aggregate_local_struct_vec() -> self.stru
        3. compute_visit_prob() -> self.visit_prob
        4. local_structure_biased_sampling() 
    
    Adaptive Walk Strategy
    -> '언제 walk를 시작하고, 얼마나 오래 walk를 진행할 것인가?' 를 결정하는 walk 생성 전략
    
    """
    def __init__(self,
            graph_df:pd.DataFrame,
            train_df:pd.DataFrame,
            bipartite:bool=False
        ):
        super().__init__(
            graph_df=graph_df,
            bipartite=bipartite
        )
        # set graph_df for training
        self.train_df=train_df
        train_adj=[[] for _ in range(self.n_node+1)]
        train_adj_edge=[[] for _ in range(self.n_node+1)]
        train_adj_t=[[] for _ in range(self.n_node+1)]
        self.train_edge_events=[]
        for event in train_df.itertuples(index=False): # col: [u,i,ts,idx=edge_id]
            src=int(event.u)
            dst=int(event.i)
            t=float(event.t)
            edge_id=int(event.idx)
            # edge 양방향 저장
            train_adj[dst].append(src)
            train_adj_edge[dst].append(edge_id)
            train_adj_t[dst].append(t)
            train_adj[src].append(dst)
            train_adj_edge[src].append(edge_id)
            train_adj_t[src].append(t)
            # edge event 저장
            self.train_edge_events.append((src,dst,t,edge_id))

        # TemporalGraph의 adjacency 표현과 동일하게 노드별 NumPy
        # 배열로 저장한다. train_df는 시간순 정렬되어 있다고 가정한다.
        self.train_adj=[
            np.asarray(values,dtype=np.int64)
            for values in train_adj
        ]
        self.train_adj_edge=[
            np.asarray(values,dtype=np.int64)
            for values in train_adj_edge
        ]
        self.train_adj_t=[
            np.asarray(values,dtype=np.float64)
            for values in train_adj_t
        ]

        # set train_n_node, train_n_event, train_max_t
        self.train_n_node=max(train_df["u"].max(),train_df["i"].max())
        self.train_n_event=train_df["idx"].max()
        self.train_max_t=train_df["t"].max()

        # setting for random walk
        self.topological_graph=nx.from_pandas_edgelist(
            self.train_df,
            source="u",
            target="i",
            create_using=nx.Graph() # Undirected Graph
        )
        self.init_stru=None # 초기 local structure vector
        self.stru=None # aggregated local structure vector
        self.visit_prob=None # 각 노드의 인접 노드 사이의 기본 방문 점수(visit probability)

    def detect_k_core_community(self,
            w_list:list
        )->dict[int,set[int]]:
        """
        Input:
            w_list: list of w
        Output:
            dict
                key: level
                value: node set
        """
        G=self.topological_graph.copy()
        core_numbers=nx.core_number(G) # 각 노드의 core_number dict
        return {
            level: {
                node
                for node,core_number in core_numbers.items()
                if core_number>=level
            }
            for level in w_list
        }

    def detect_k_truss_community(self,
            y_list:list
        )->dict[int, set[int]]:
        """
        Input:
            y: list of y
        Output:
            dict
                key: level
                value: node set
        """
        if any(level<2 for level in y_list):
            raise ValueError(
                "k-truss level은 2 이상이어야 합니다."
            )
        G=self.topological_graph.copy()
        # 간선이 없는 고립 노드 제거
        G.remove_nodes_from(
            list(nx.isolates(G))
        )
        return {
            level:set(
                nx.k_truss(
                    G,
                    k=level
                ).nodes()
            )
            for level in y_list
        }

    def detect_k_clique_community(self,
            z_list:list
        )->dict[int,set[int]]:
        """
        Input:
            z: list of z
        Output:
            dict
                key: level
                value: node set
        """
        if any(level<2 for level in z_list):
            raise ValueError(
                "k-clique level은 2 이상이어야 합니다."
            )
        G=self.topological_graph.copy()

        # maximal clique 탐지
        maximal_cliques=list(
            nx.find_cliques(G)
        )

        return {
            level: {
                node
                for clique in maximal_cliques
                if len(clique)>=level
                for node in clique
            }
            for level in z_list
        }

    def generate_init_local_struct_vec(self,
            k_list:list
        ):
        """
        Input:
            k_list: list of k
        Output:
            local_struct_vec:
                길이: self.n_node + 1

                struct_vec[node_id]:
                    해당 노드의 초기 local structure vector

                struct_vec[0]:
                    padding node의 zero vector

                각 노드 벡터의 길이:
                    3 * len(k_list)

                벡터 구성:
                    [k-core | k-truss | k-clique]
        """
        if len(k_list)!=len(set(k_list)):
            raise ValueError(
                "k_list에는 중복된 값이 없어야 합니다."
            )

        # 세 종류의 community 탐지
        k_core_com=self.detect_k_core_community(
            w_list=k_list
        )
        k_truss_com=self.detect_k_truss_community(
            y_list=k_list
        )
        k_clique_com=self.detect_k_clique_community(
            z_list=k_list
        )
        n_level=len(k_list)
        struct_dim=3*n_level

        # index가 node ID와 같도록 self.n_node + 1개 생성
        # node 0은 padding node이므로 zero vector로 유지
        local_struct_vec=[
            np.zeros(
                struct_dim,
                dtype=np.float32
            )
            for _ in range(self.train_n_node+1)
        ]

        level_to_idx={
            k: idx
            for idx,k in enumerate(k_list)
        }
        community_results=[
            k_core_com,
            k_truss_com,
            k_clique_com,
        ]

        for community_idx,community_result in enumerate(
                community_results
            ):
            offset=community_idx*n_level
            for k,node_set in community_result.items():
                level_idx=level_to_idx[k]
                for node in node_set:
                    local_struct_vec[node][offset+level_idx]=k
        self.init_stru=local_struct_vec
        return local_struct_vec

    def aggregate_local_struct_vec(self,
            n_aggr:int
        )->list[np.ndarray]:
        """
        Input:
            init_stru: init local structure vector list
            n_aggr: aggregate 반복 횟수
        Output:
            stru: aggregated local structure vector list
        """
        if self.init_stru is None:
            raise Exception("init_stru 생성 필요.")

        # 원본 init_stru를 변경하지 않도록 복사
        stru=[
            np.asarray(
                vec,
                dtype=np.float32
            ).copy()
            for vec in self.init_stru
        ]

        # node 0은 padding node
        stru[0].fill(0.0)

        for _ in range(n_aggr):
            # 현재 layer 계산에는 이전 layer의 벡터만 사용
            prev_stru=stru
            next_stru=[
                vec.copy()
                for vec in prev_stru
            ]
            for node in self.topological_graph.nodes:
                neighbors=list(
                    self.topological_graph.neighbors(node)
                )
                # 고립 노드는 이전 벡터 유지
                if not neighbors:
                    continue

                similarities=np.asarray([
                    RandomWalkUtils.compute_similarity(
                        prev_stru[node],
                        prev_stru[neighbor]
                    )
                    for neighbor in neighbors
                ],dtype=np.float32)

                similarity_sum=float(
                    similarities.sum()
                )

                # 모든 이웃과의 유사도가 0인 경우
                # 논문의 weight 식을 계산할 수 없으므로 집계하지 않음
                if similarity_sum==0.0:
                    continue

                weights=(
                    similarities / similarity_sum
                )
                aggregated_vec=np.zeros_like(
                    prev_stru[node]
                )
                for neighbor,weight in zip(
                    neighbors,
                    weights
                ):
                    aggregated_vec+=(
                        weight*prev_stru[neighbor]
                    )
                next_stru[node]=(
                    prev_stru[node]+aggregated_vec
                )
            next_stru[0].fill(0.0) # padding node는 모든 layer에서 zero vector 유지
            stru=next_stru
        self.stru=stru
        return stru

    def compute_visit_prob(self)->dict[int,dict[int,float]]:
        """
        Aggregated local structure vector를 이용하여
        각 노드의 인접 노드 사이의 기본 방문 점수(visit probability)를 계산
        
        Input:
            stru: list[np.ndarray]
        Output:
            visit_p: dict[int,dict[int,float]]
        """
        if self.stru is None:
            raise Exception(f"aggregated final strue 생성 필요.")

        visit_prob:dict[int,dict[int,float]]={}
        for node in self.topological_graph.nodes:
            node=int(node)
            visit_prob[node]={}
            for neighbor in self.topological_graph.neighbors(node):
                neighbor=int(neighbor)
                visit_prob[node][neighbor]=RandomWalkUtils.compute_similarity(
                    vec_a=self.stru[node],
                    vec_b=self.stru[neighbor]
                )
        self.visit_prob=visit_prob
        return visit_prob

    def DBSCAN_clustering(self,
            node:int,
            min_points:int=2
        )->list[tuple[float,float]]:
        """
        논문의 adaptive sampling 전략에 따라 node의 active time을
        1차원 DBSCAN으로 군집화한다.

        Input:
            node: active time을 군집화할 노드
            min_points:
                core point를 판별할 최소 이웃 수.
                기준 시각 자기 자신도 이웃 수에 포함한다.
        Output:
            time_intervals:
                각 cluster의 (최소 active time, 최대 active time) 목록.
                noise point는 (active time, active time) 단일 구간으로
                보존하며, 전체 구간은 시작 시간순으로 정렬한다.
        """
        if min_points<1:
            raise ValueError(
                "min_points는 1 이상이어야 합니다."
            )

        active_times=sorted(
            set(
                float(timestamp)
                for timestamp in self.train_adj_t[node]
            )
        )
        n_active_time=len(active_times)

        if n_active_time==0: # 발생 시간 없으면 빈 리스트 반환
            return []
        if n_active_time==1: # 발생 시간이 하나뿐이면 해당 시간을 시작과 끝이 같은 구간으로 반환
            active_time=active_times[0]
            return [
                (
                    active_time,
                    active_time
                )
            ]

        # DBSCAN 반경 계산
        # 논문의 식: r = TS_u / (|T_u| - 1)
        radius=(
            active_times[-1]-active_times[0]
        )/(n_active_time-1)

        # DBSCAN 상태 초기화
        labels=[-1]*n_active_time # labels[i]: 각 발생 시각이 속한 군집 번호
        visited=[False]*n_active_time # visited[i]: 해당 발생 시각을 검사했는지 표시
        cluster_id=0 # 새 군집에 부여할 번호

        def region_query(point_idx:int)->list[int]:
            """
            선택된 발생 시각과의 시간 차이가 radius 이하인 모든 발생 시각을 찾습니다.
            """
            point=active_times[point_idx]
            return [
                neighbor_idx
                for neighbor_idx,neighbor in enumerate(active_times)
                if abs(neighbor-point)<=radius
            ]

        for point_idx in range(n_active_time):
            if visited[point_idx]:
                continue

            visited[point_idx]=True
            neighbors=region_query(point_idx)
            if len(neighbors)<min_points:
                continue

            labels[point_idx]=cluster_id
            seeds=list(neighbors)
            seed_set=set(seeds)
            seed_idx=0

            while seed_idx<len(seeds):
                neighbor_idx=seeds[seed_idx]
                seed_idx+=1

                if not visited[neighbor_idx]:
                    visited[neighbor_idx]=True
                    neighbor_neighbors=region_query(neighbor_idx)
                    if len(neighbor_neighbors)>=min_points:
                        for candidate_idx in neighbor_neighbors:
                            if candidate_idx not in seed_set:
                                seeds.append(candidate_idx)
                                seed_set.add(candidate_idx)

                if labels[neighbor_idx]==-1:
                    labels[neighbor_idx]=cluster_id
            cluster_id+=1

        time_intervals=[]
        for current_cluster_id in range(cluster_id):
            cluster=[
                timestamp
                for timestamp,label in zip(active_times,labels)
                if label==current_cluster_id
            ]
            time_intervals.append(
                (
                    cluster[0],
                    cluster[-1]
                )
            )

        # 논문은 DBSCAN noise 처리 방법을 명시하지 않는다.
        # noise를 제거하면 해당 시각의 temporal interaction이 walk
        # 생성에서 완전히 제외되므로 각각 단일 시간 구간으로 보존한다.
        time_intervals.extend(
            (
                timestamp,
                timestamp
            )
            for timestamp,label in zip(active_times,labels)
            if label==-1
        )
        return sorted(
            time_intervals,
            key=lambda interval:(
                interval[0],
                interval[1]
            )
        )

    def local_structure_biased_sampling(self,
            node:int,
            walk_path:list[int],
            time_interval:tuple[float,float],
            start_inclusive:bool=True,
            excluded_neighbors:set[int]|None=None
        )->int|None:
        """
        Local Structure Biased Sampling (LSBS)

        Input:
            node: 현재 노드
            walk_path: 현재까지 생성된 walk path
            time_interval: walk가 허용되는 시간 구간
            start_inclusive: time_interval의 시작 시각 포함 여부
            excluded_neighbors: 현재 tree node에서 이미 샘플된 이웃
        Output:
            sampled_neighbor: 선택된 다음 이웃 노드, 시간 구간 안에 방문 가능한 이웃이 없으면 None
        """
        if self.visit_prob is None:
            raise Exception("visit probability 계산 필요.")
        if len(time_interval)!=2:
            raise ValueError(
                "time_interval은 (start_time, end_time) 형식이어야 합니다."
            )

        start_time,end_time=time_interval
        if start_time>end_time:
            raise ValueError(
                "time_interval의 시작 시간은 종료 시간보다 클 수 없습니다."
            )
        if walk_path and walk_path[-1]!=node:
            raise ValueError(
                "walk_path의 마지막 노드는 현재 node와 같아야 합니다."
            )
        excluded_neighbors=(
            set()
            if excluded_neighbors is None
            else excluded_neighbors
        )

        # 동일한 이웃과 여러 번 접촉했더라도 sampling 후보에는 한 번만 포함한다.
        # 주어진 시간 구간에 접촉한 이웃만 선택한다.
        candidate_neighbors=[] # 실제 샘플링에 사용할 이웃 목록
        candidate_set=set() # 같은 이웃이 중복으로 추가되는 것을 방지하는 집합
        for neighbor,timestamp in zip(
                self.train_adj[node],
                self.train_adj_t[node]
            ):
            neighbor=int(neighbor)
            timestamp=float(timestamp)
            if (
                (
                    start_time<=timestamp
                    if start_inclusive
                    else start_time<timestamp
                )
                and timestamp<=end_time
                and neighbor not in excluded_neighbors
                and neighbor not in candidate_set
            ):
                candidate_neighbors.append(neighbor)
                candidate_set.add(neighbor)

        if not candidate_neighbors:
            return None # valid 이웃 없음

        weights=[]
        is_start_node=len(walk_path)<=1
        previous_node=(
            None
            if is_start_node
            else walk_path[-2]
        )

        for neighbor in candidate_neighbors:
            # 식 (4)로 미리 계산한 P(v, x)
            weight=self.visit_prob.get(
                node,{}
            ).get(
                neighbor,
                0.0
            )

            if (
                previous_node is not None
                and neighbor!=previous_node
                and self.topological_graph.has_edge(
                    previous_node,
                    neighbor
                )
            ):
                # 식 (5)의 d(u, x)=1인 경우:
                # P(u, x)가 더 크면 해당 확률로 갱신한다.
                previous_weight=self.visit_prob.get(
                    previous_node,{}
                ).get(
                    neighbor,
                    0.0
                )
                if previous_weight>weight:
                    weight=previous_weight

            weights.append(weight)

        # 모든 구조 유사도가 0이면 가중치 선택이 불가능하므로,
        # 유효한 temporal neighbor 사이에서 균등하게 선택한다.
        sampling_weights=(
            weights
            if any(weight>0.0 for weight in weights)
            else None
        )
        return RandomWalkUtils.random_sampling(
            rng=self.rng,
            population=candidate_neighbors,
            weights=sampling_weights
        )

    def get_walks_using_path_tree(self,
            node:int,
            time_interval:tuple[float,float],
            walk_len:int=20,
            n_lsbs:int=1
        )->list[list[int]]:
        """
        논문의 Algorithm 3 PathTree에 따라 주어진 시간 구간에서
        temporal reachable walk path들을 생성한다.

        Input:
            node: path tree의 root node
            time_interval: walk가 허용되는 시간 구간
            walk_len: 하나의 walk에 포함할 최대 노드 수
            n_path: 각 tree node에서 수행할 최대 LSBS 횟수
        Output:
            walks: list[list[int]], list of walk path
        """
        if len(time_interval)!=2:
            raise ValueError(
                "time_interval은 (start_time, end_time) 형식이어야 합니다."
            )

        start_time,end_time=time_interval
        if start_time>end_time:
            raise ValueError(
                "time_interval의 시작 시간은 종료 시간보다 클 수 없습니다."
            )
        if walk_len<1:
            raise ValueError(
                "walk_len은 1 이상이어야 합니다."
            )
        if n_lsbs<1:
            raise ValueError(
                "n_lsbs 1 이상이어야 합니다."
            )

        # 각 queue item은 논문의 tree node instance에 해당한다.
        # 동일한 원본 node도 서로 다른 parent나 arrival time을 가지면
        # 별개의 tree node instance로 저장한다.
        queue=deque([
            {
                "node":node,
                "arrival_time":None,
                "walk_path":[node],
                "visited_states":set(),
            }
        ])
        walks=[]

        while queue:
            tree_node=queue.popleft()

            current_node=tree_node["node"]
            arrival_time=tree_node["arrival_time"]
            walk_path=tree_node["walk_path"]
            visited_states=tree_node["visited_states"]

            if len(walk_path)>=walk_len:
                walks.append(walk_path)
                continue

            # root에서는 전체 interval을 사용하고, 이후에는 현재
            # tree node의 arrival time 이후에 발생한 edge만 허용한다.
            candidate_start=(
                start_time
                if arrival_time is None
                else arrival_time
            )

            # 논문 line 9의 u.Arr_u < max(C_i)에 해당한다.
            if (
                arrival_time is not None
                and arrival_time>=end_time
            ):
                walks.append(walk_path)
                continue

            # 시간 도달성을 만족하는 서로 다른 이웃과 각 이웃으로
            # 이동할 수 있는 가장 이른 arrival time을 구한다.
            eligible_arrival_times={}
            interval_neighbors=set()
            for neighbor,timestamp in zip(
                    self.train_adj[current_node],
                    self.train_adj_t[current_node]
                ):
                neighbor=int(neighbor)
                timestamp=float(timestamp)
                is_after_start=(
                    candidate_start<=timestamp
                    if arrival_time is None
                    else candidate_start<timestamp
                )
                if is_after_start and timestamp<=end_time:
                    interval_neighbors.add(neighbor)
                    if (neighbor,timestamp) in visited_states:
                        continue
                    previous_timestamp=eligible_arrival_times.get(
                        neighbor
                    )
                    if (
                        previous_timestamp is None
                        or timestamp<previous_timestamp
                    ):
                        eligible_arrival_times[neighbor]=timestamp

            if not eligible_arrival_times:
                walks.append(walk_path)
                continue

            # 모든 유효 이웃을 매 단계 확장하면 path tree의 크기가
            # 지수적으로 증가하므로 지정한 횟수까지만 LSBS를 수행한다.
            n_current_lsbs=min(
                len(eligible_arrival_times),
                n_lsbs
            )
            sampled_children=set()
            child_created=False

            for _ in range(n_current_lsbs):
                sampled_neighbor=self.local_structure_biased_sampling(
                    node=current_node,
                    walk_path=walk_path,
                    time_interval=(
                        candidate_start,
                        end_time
                    ),
                    # root의 첫 edge는 interval start를 포함하고,
                    # 이후 edge는 직전 arrival time보다 엄격히 커야 한다.
                    start_inclusive=arrival_time is None,
                    # Algorithm 3의 한 tree node에서 같은 child를
                    # 중복 생성하지 않도록 비복원 샘플링한다.
                    # 이미 방문한 time-expanded state만 가진 이웃도
                    # 샘플링 후보에서 제외한다.
                    excluded_neighbors=(
                        sampled_children
                        | (
                            interval_neighbors
                            - eligible_arrival_times.keys()
                        )
                    )
                )
                if sampled_neighbor is None:
                    break

                sampled_children.add(sampled_neighbor)
                next_arrival_time=eligible_arrival_times[
                    sampled_neighbor
                ]
                next_state=(
                    sampled_neighbor,
                    next_arrival_time
                )

                # 같은 time-expanded node를 한 경로에서 다시 방문하면
                # 동일 시각의 edge를 왕복하며 tree가 무한히 확장될 수 있다.
                if next_state in visited_states:
                    continue

                next_visited_states=set(visited_states)
                next_visited_states.add(
                    (
                        current_node,
                        next_arrival_time
                        if arrival_time is None
                        else arrival_time
                    )
                )
                next_visited_states.add(next_state)

                queue.append(
                    {
                        "node":sampled_neighbor,
                        "arrival_time":next_arrival_time,
                        "walk_path":walk_path+[sampled_neighbor],
                        "visited_states":next_visited_states,
                    }
                )
                child_created=True

            # 샘플 결과가 모두 중복 상태라 자식이 만들어지지 않은
            # 경우에도 현재 tree node는 leaf이다.
            if not child_created:
                walks.append(walk_path)
        return walks

    def generate_walks(self,
            k_list:list,
            n_aggr:int,
            min_points:int=2,
            walk_len:int=20,
            n_lsbs:int=1
        )->list[list[str]]:
        """
        Local structure vector와 visit probability를 계산한 뒤,
        논문의 Algorithm 2에 따라 모든 노드의 active time을
        clustering하고 각 time interval에서 path tree walk를 생성한다.

        Input:
            k_list: community detection에 사용할 k 목록
            n_aggr: local structure vector aggregation 반복 횟수
            min_points: Adaptive Walk Strategy를 위한 DBSCAN core point를 판별할 최소 이웃 수
            walk_len: 하나의 walk에 포함할 최대 노드 수
            n_lsbs: 각 tree node에서 수행할 최대 LSBS 횟수
        Output:
            walks: list[list[str]], list of walk path
        """
        if n_aggr<0:
            raise ValueError(
                "n_aggr은 0 이상이어야 합니다."
            )
        self.generate_init_local_struct_vec(k_list=k_list)
        self.aggregate_local_struct_vec(n_aggr=n_aggr)
        self.compute_visit_prob()

        walks=[]
        for node in self.topological_graph.nodes:
            node=int(node)
            time_intervals=self.DBSCAN_clustering(
                node=node,
                min_points=min_points
            )

            for time_interval in time_intervals:
                node_walks=self.get_walks_using_path_tree(
                    node=node,
                    time_interval=time_interval,
                    walk_len=walk_len,
                    n_lsbs=n_lsbs
                )
                walks.extend(
                    [
                        str(walk_node)
                        for walk_node in walk
                    ]
                    for walk in node_walks
                )
        return walks
