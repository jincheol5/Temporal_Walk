import random
import math
import pandas as pd
import numpy as np
from typing import Literal
from .temporal_graph import TemporalGraph
from utils import RandomWalkUtils

class CTDNE_Graph(TemporalGraph):
    """
    exponential 방법은 CTDNE 논문에서 제안되었지만 실험에서 사용 안함 
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

    def select_start_temporal_edge(self,
            sampling_method:Literal[
                "uniform",
                "exponential",
                "linear"
            ],
            temperature:float=0.1
        ):
        """
        Walk의 시작 edge event를 선정

        Input:
            sampling_method:
                uniform: 모든 edge를 동일한 확률로 선택
                exponential: 최근 edge일수록 지수적으로 높은 확률로 선택, P(e) ∝ exp((t_e - t_max) / temperature)
                linear: 시간순 rank에 비례해 temporal edge 선택
            temperature: 
        Return:
            sampled edge_event tuple
        """
        match sampling_method:
            case "uniform":
                return RandomWalkUtils.random_sampling(
                    rng=self.rng,
                    population=self.train_edge_events
                )
            case "exponential":
                if temperature<=0:
                    raise ValueError("temperature는 0보다 커야 합니다.")
                weights=[
                    math.exp((t-self.train_max_t)/temperature)
                    for _,_,t,_ in self.train_edge_events
                ]
                return RandomWalkUtils.random_sampling(
                    rng=self.rng,
                    population=self.train_edge_events,
                    weights=weights
                )
            case "linear":
                n_edge=len(self.train_edge_events)
                return RandomWalkUtils.random_sampling(
                    rng=self.rng,
                    population=self.train_edge_events,
                    weights=range(1,n_edge+1) # 가장 최근 edge가 가중치 높도록 설정
                )

    def select_temporal_neighbor(self,
            cur_node:int,
            cur_t:float,
            sampling_method:Literal[
                "uniform",
                "exponential",
                "linear"
            ]
        ):
        """
        Random Walk 중 다음 이웃 노드 선택

        Input:
            cur_node: 현재 노드 ID
            cur_t: 현재 기준 시간
            sampling method:
                uniform: 방문 가능한 모든 이웃에게 동일한 확률을 부여
                exponential: 
                    시간적 근접성을 강조하기 위해 지수 함수를 사용
                    최근의 상호작용이 향후 발생할 이벤트와 더 밀접한 관련이 있다는 가정(시간적 감쇠, Time Decay)에 기반
                linear: 
                    시간적으로 연속된 두 edge 사이의 시간 차이가 클 때, edge를 이산 시간 단계로 매핑
                    시간 순서의 중요성은 반영하되 완만한 가중치를 원할 때
        Return:
            selected neighbor id, selected neighbor event time
        """
        timestamps=self.train_adj_t[cur_node]
        if timestamps.size==0:
            return None

        # timestamp > current_t를 처음 만족하는 위치
        start_idx=int(np.searchsorted(
            timestamps,
            cur_t,
            side="right"
        ))
        if start_idx==len(timestamps):
            return None
        candidate_indices=list(
            range(start_idx,len(timestamps))
        )

        match sampling_method:
            case "uniform":
                # [start_idx,len(timestamps)-1]에서 균등하게 index 선택
                selected_idx=RandomWalkUtils.random_sampling(
                    rng=self.rng,
                    population=candidate_indices
                )
            case "exponential":
                time_diffs=[
                    timestamps[idx]-cur_t
                    for idx in candidate_indices
                ]
        
                # 가장 가까운 temporal edge의 시간 차이
                min_time_diff=min(time_diffs)
        
                # exp(-Δt)를 그대로 계산하면 timestamp가 큰 경우 underflow가
                # 발생할 수 있으므로 최소 시간 차이를 뺀다.
                # exp(-(Δt - min_Δt))
                # 공통 상수를 곱하거나 나누는 것은 정규화된 선택 확률을 바꾸지 않는다.
                weights=[
                    math.exp(-(time_diff-min_time_diff))
                    for time_diff in time_diffs
                ]
                selected_idx=RandomWalkUtils.random_sampling(
                    rng=self.rng,
                    population=candidate_indices,
                    weights=weights
                )
            case "linear":
                # 가까운 timestamp부터 큰 가중치 부여
                # candidate가 시간순으로 [가까운 edge, ..., 먼 edge]이므로
                # weights는 [n, n-1, ..., 1]
                n_candidate=len(candidate_indices)
                weights=list(
                    range(n_candidate,0,-1)
                )
                selected_idx=RandomWalkUtils.random_sampling(
                    rng=self.rng,
                    population=candidate_indices,
                    weights=weights
                )
        return (
            int(self.train_adj[cur_node][selected_idx]),
            float(self.train_adj_t[cur_node][selected_idx])
        )

    def temporal_random_walk(self,
            source:int,
            start_t:float,
            walk_len:int,
            neighbor_sampling:Literal[
                "uniform",
                "linear",
                "exponential"
            ]="uniform"
        ):
        """
        source node = 선택된 edge_event의 dst node
        start_t = 선택된 edge_event의 event time
        source 노드와 현재 시각에서 시작하여 temporal random walk를 생성한다.
        각 단계에서는 현재 timestamp보다 큰 timestamp를 가진 temporal edge만 다음 edge의 후보로 사용한다. 
        유효한 temporal neighbor가 없으면 walk_len에 도달하기 전이라도 walk를 종료한다.

        Input:
            source: source node id
            start_t: start time
            walk_len: walk에 포함할 최대 노드 수
            neighbor_sampling: temporal neighbor sampling 방법

        return:
            walk: list of str node_id
        """
        walk=[str(source)]
        cur_node=source
        cur_t=start_t
        while len(walk)<walk_len:
            selected_neighbor=self.select_temporal_neighbor(
                cur_node=cur_node,
                cur_t=cur_t,
                sampling_method=neighbor_sampling
            )
            if selected_neighbor is None:
                break
            neighbor,neighbor_t=selected_neighbor
            walk.append(str(neighbor))
            cur_node=neighbor
            cur_t=neighbor_t
        return walk

    def generate_walks(self,
            walk_len:int,
            min_walk_len:int,
            n_walk:int,
            n_window:int,
            edge_sampling:Literal[
                "uniform",
                "linear",
                "exponential"
            ]="uniform",
            neighbor_sampling:Literal[
                "uniform",
                "linear",
                "exponential"
            ]="uniform"
        ):
        """
        Input:
            walk_len:
            min_walk_len: 
            n_walk: 각 source node마다 random walk n_walk번 수행
            n_window: 생성할 temporal context window의 목표 개수(β)
            max_attempt: 최대 walk 생성 시도 횟수
            edge_sampling:
            neighbor_sampling: 
            seed: 
        Return:
            walks: list of walk
        """
        walks=[]
        window_count=0
        max_attempt=n_window*10
        for _ in range(max_attempt):
            if n_window<=window_count:
                break
            ### 1. 시작 edge event 선택
            src,dst,t,_=self.select_start_temporal_edge(
                sampling_method=edge_sampling
            )
            ### 2. n_walk번 walk 생성, skip-gram 학습을 위해 node_id=str
            for _ in range(n_walk):
                walk=self.temporal_random_walk(
                    source=dst,
                    start_t=t,
                    walk_len=walk_len-1,
                    neighbor_sampling=neighbor_sampling,
                )
                walk=[str(src)]+walk

                # walk가 최소 길이를 만족하지 못하면 폐기, 만족하면 walks에 추가
                if len(walk)<min_walk_len:
                    continue
                walks.append(walk)

                ### 해당 walk_seq가 생성하는 temporal context window 개수 계산 후 종료 조건 확인
                window_count+=(len(walk)-min_walk_len+1)
        return walks
