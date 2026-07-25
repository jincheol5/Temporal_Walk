import random
import math
import numpy as np
import networkx as nx
from typing import Literal
from collections import defaultdict
from utils import SamplingUtils
from .temporal_graph import TemporalGraph

class ATDGEB_Graph(TemporalGraph):
    """
    << now >>
    효율성을 위한 다음 기법들은 추후에 개발한다.
        - Alias sampling
        - Path tree


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
            graph_df,
            bipartite:bool=False
        ):
        super().__init__(
            graph_df=graph_df,
            bipartite=bipartite
        )
        self.topological_graph=nx.from_pandas_edgelist(
            self.graph_df,
            source="u",
            target="i",
            create_using=nx.Graph() # Undirected Graph
        )
        self.static_graph=None
        self.init_stru=None # 초기 local structure vector
        self.stru=None # aggregated local structure vector
        self.visit_prob=None # 각 노드의 인접 노드 사이의 기본 방문 점수(visit probability)

    def transform_temporal_graph_to_static_graph(self):
        """
        temporal graph를 time-expanded static graph로 변환합니다.
        node id는 (id,timestamp) tuple 로 구성된다. 
        edge는 temporal edge, contact edge 두 종류가 생성된다.
            temporal edge: (a,1) -> (a,2) -> (a,3)
            contact edge: (a,1) -> (b,1), (b,5) -> (c,5)
        """
        df=self.graph_df.loc[:,["u", "i", "ts"]].copy()

        # 결과 그래프는 시간 방향을 표현해야 하므로 directed graph 사용
        self.static_graph=nx.DiGraph()

        # 각 원래 노드가 활동한 timestamp 저장
        node_active_times:dict[int,set[float]]=defaultdict(set)
        for event in df.itertuples(index=False):
            src=int(event.u)
            dst=int(event.i)
            timestamp=float(event.ts)
            node_active_times[src].add(timestamp)
            node_active_times[dst].add(timestamp)

        # 시간별 복제 노드 생성
        for original_node,active_times in node_active_times.items():
            for timestamp in sorted(active_times):
                expanded_node=(
                    original_node,
                    timestamp
                )
                self.static_graph.add_node(
                    expanded_node,
                    original_node=original_node,
                    timestamp=timestamp
                )

        # 동일한 원래 노드의 복제 노드를 시간순으로 연결
        for original_node,active_times in node_active_times.items():
            sorted_times=sorted(active_times)
            for previous_t,next_t in zip(
                    sorted_times[:-1],
                    sorted_times[1:]
                ):
                previous_node=(
                    original_node,
                    previous_t
                )
                next_node=(
                    original_node,
                    next_t
                )
                self.static_graph.add_edge(
                    previous_node,
                    next_node,
                    edge_type="temporal",
                    timestamp=next_t
                )

        # 원래 temporal interaction을 같은 시간의 노드끼리 연결
        for event in df.itertuples(index=False):
            src=int(event.u)
            dst=int(event.i)
            timestamp=float(event.ts)
            expanded_src=(
                src,
                timestamp
            )
            expanded_dst=(
                dst,
                timestamp
            )
            self.static_graph.add_edge(
                expanded_src,
                expanded_dst,
                edge_type="contact",
                timestamp=timestamp
            )
            self.static_graph.add_edge(
                expanded_dst,
                expanded_src,
                edge_type="contact",
                timestamp=timestamp
            )

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
            for _ in range(self.n_node+1)
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
            L:int
        )->list[np.ndarray]:
        """
        Input:
            init_stru: init local structure vector list
            L: aggregate 반복 횟수
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

        for _ in range(L):
            # 현재 layer 계산에는 이전 layer의 벡터만 사용
            prev_stru=stru
            next_stru=[
                vec.copy()
                for vec in prev_stru
            ]
            for node in range(1,self.n_node+1):
                neighbors=list(
                    self.topological_graph.neighbors(node)
                )
                # 고립 노드는 이전 벡터 유지
                if not neighbors:
                    continue

                similarities=np.asarray([
                    SamplingUtils.compute_similarity(
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
                visit_prob[node][neighbor]=SamplingUtils.compute_similarity(
                    vec_a=self.stru[node],
                    vec_b=self.stru[neighbor]
                )
        self.visit_prob=visit_prob
        return visit_prob

    def local_structure_biased_sampling(self,
            node:int,
            walk_path:list[int],
            time_interval:tuple[float,float]
        )->int|None:
        """
        time-expanded static graph 기반으로 LSBS 수행.

        Input:
            node: 현재 노드
            walk_path: 현재까지 생성된 walk path
            time_interval: walk가 허용되는 시간 구간
        Output:
            sampled_neighbor: 선택된 다음 이웃 노드, 시간 구간 안에 방문 가능한 이웃이 없으면 None
        """ 
