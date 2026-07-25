import random
import math
from typing import Literal
from bisect import bisect_right
from .temporal_graph import TemporalGraph
from utils import SamplingUtils

class CTDNE_Graph(TemporalGraph):
    def __init__(self,
            graph_df,
            bipartite:bool=False
        ):
        super().__init__(
            graph_df=graph_df,
            bipartite=bipartite
        )

    def select_temporal_edge_uniform(self):
        """
        모든 temporal edge를 동일한 확률로 선택
        """
        return SamplingUtils.random_sampling(
                rng=self.rng,
                population=self.edge_events
            )

    def select_temporal_edge_linear(self):
        """
        시간순 rank에 비례해 temporal edge 선택
        """
        n_edge=len(self.edge_events)
        return SamplingUtils.random_sampling(
                rng=self.rng,
                population=self.edge_events,
                weights=range(1,n_edge+1) # 가장 최근 edge가 가중치 높도록 설정
            )

    def select_temporal_edge_exponential(self,
            temperature:float=1.0
        ):
        """
        최근 edge일수록 지수적으로 높은 확률로 선택
        P(e) ∝ exp((t_e - t_max) / temperature)
        """
        if temperature<=0:
            raise ValueError("temperature는 0보다 커야 합니다.")
        weights=[
            math.exp((t-self.max_t)/temperature)
            for _,_,t in self.edge_events
        ]
        return SamplingUtils.random_sampling(
                rng=self.rng,
                population=self.edge_events,
                weights=weights
            )

    def select_temporal_neighbor_uniform(self,
            node:int,
            cur_t:float
        ):
        """
        현재 source 노드에서 timestamp >= cur_t temporal edge 중
        하나를 균등한 확률로 선택한다.

        self.adj_t[source]는 timestamp 오름차순으로 정렬되어 있다고 가정한다.
        Input:
        
        Output:
            neighbor_id
            neighbor_t
        """
        timestamps=self.adj_t[node]
        if not timestamps:
            return None

        # timestamp > current_t를 처음 만족하는 위치
        start_idx=bisect_right(timestamps,cur_t)
        if start_idx==len(timestamps):
            return None

        # [start_idx,len(timestamps)-1]에서 균등하게 index 선택
        selected_idx=self.rng.randrange(
            start_idx,
            len(timestamps),
        )
        return self.adj[node][selected_idx],self.adj_t[node][selected_idx]

    def select_temporal_neighbor_linear(self,
            node:int,
            cur_t:float
        ):
        """
        현재 node 노드에서 timestamp > cur_t temporal edge 중
        현재 시간과 가까운 edge에 더 큰 선형 가중치를 부여하여 선택한다.

        self.adj_t[node]는 timestamp 오름차순으로 정렬되어 있다고 가정한다.
        """
        timestamps=self.adj_t[node]
        if not timestamps:
            return None

        # timestamp > cur_t 처음 만족하는 위치
        start_idx=bisect_right(timestamps,cur_t)
        if start_idx==len(timestamps):
            return None

        candidate_indices=list(
            range(start_idx,len(timestamps))
        )
        n_candidate=len(candidate_indices)

        # 가까운 timestamp부터 큰 가중치 부여
        # candidate가 시간순으로 [가까운 edge, ..., 먼 edge]이므로
        # weights는 [n, n-1, ..., 1]
        weights=list(
            range(n_candidate,0,-1)
        )
        selected_idx=SamplingUtils.random_sampling(
            rng=self.rng,
            population=candidate_indices,
            weights=weights
        )
        return self.adj[node][selected_idx],self.adj_t[node][selected_idx]
    
    def select_temporal_neighbor_exponential(self,
            node:int,
            cur_t:float
        ):
        """
        현재 node 노드에서 timestamp > cur_t temporal edge 중
        현재 시간과의 간격이 짧은 edge에 더 큰 지수 가중치를 부여하여 선택한다.

        각 temporal edge의 가중치는 다음에 비례한다.

            weight = exp(-(neighbor_t - current_t))

        수치 안정성을 위해 모든 시간 차이에서 최소 시간 차이를 뺀 뒤
        지수 함수를 적용한다. 이는 최종 선택 확률을 바꾸지 않는다.

        self.adj_t[node]는 timestamp 오름차순으로 정렬되어 있다고 가정한다.
        """
        timestamps=self.adj_t[node]
        if not timestamps:
            return None

        # timestamp > current_t를 처음 만족하는 위치
        start_idx=bisect_right(timestamps,cur_t)
        if start_idx==len(timestamps):
            return None

        candidate_indices=list(
            range(start_idx,len(timestamps))
        )

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
        selected_idx=SamplingUtils.random_sampling(
            rng=self.rng,
            population=candidate_indices,
            weights=weights
        )
        return self.adj[node][selected_idx],self.adj_t[node][selected_idx]

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
        주어진 source 노드와 현재 시각에서 시작하여 temporal random walk를 생성한다.
        각 단계에서는 현재 timestamp보다 큰 timestamp를 가진 temporal edge만 다음 edge의 후보로 사용한다. 
        유효한 temporal neighbor가 없으면 walk_len에 도달하기 전이라도 walk를 종료한다.

        Input:
            source: source node id
            start_t: start time
            walk_len: walk에 포함할 최대 노드 수
            neighbor_sampling: temporal neighbor sampling 방법

        Output:
            walk_seq: list of str node id
        """
        walk_seq=[str(source)]
        cur_node=source
        cur_t=start_t

        while len(walk_seq)<walk_len:
            match neighbor_sampling:
                case "uniform":
                    selected_neighbor=(
                        self.select_temporal_neighbor_uniform(
                            node=cur_node,
                            cur_t=cur_t,
                        )
                    )
                case "linear":
                    selected_neighbor=(
                        self.select_temporal_neighbor_linear(
                            node=cur_node,
                            cur_t=cur_t,
                        )
                    )
                case "exponential":
                    selected_neighbor=(
                        self.select_temporal_neighbor_exponential(
                            node=cur_node,
                            cur_t=cur_t,
                        )
                    )
            # 현재 시간 이후의 temporal edge가 없는 경우
            if selected_neighbor is None:
                break
            next_node,next_t=selected_neighbor
            walk_seq.append(str(next_node))

            # 다음 walk step을 위한 상태 갱신
            cur_node=next_node
            cur_t=next_t
        return walk_seq

    def generate_walks(self,
            walk_len:int,
            min_walk_len:int,
            n_context_window:int,
            max_attempt:int|None=None,
            edge_sampling:Literal[
                "uniform",
                "linear",
                "exponential"
            ]="uniform",
            neighbor_sampling:Literal[
                "uniform",
                "linear",
                "exponential"
            ]="uniform",
            seed:int=1
        ):
        """
        Input:
            walk_len: walk에 포함할 최대 노드 수
            min_walk_len: 허용할 최소 walk 길이
            n_context_window: 생성할 temporal context window의 목표 개수(β)
            max_attempt: 최대 walk 생성 시도 횟수
        Output:
            walks: list of walk_seq
        """
        if walk_len<2:
            raise ValueError("walk_len은 2 이상이어야 합니다.")
        if min_walk_len<1:
            raise ValueError("min_walk_len는 1 이상이어야 합니다.")
        if min_walk_len>walk_len:
            raise ValueError("min_walk_len는 walk_len보다 클 수 없습니다.")
        if max_attempt is None:
            max_attempt=n_context_window*100

        self.set_random_seed(seed=seed)

        edge_sampling_fn={
            "uniform":self.select_temporal_edge_uniform,
            "linear":self.select_temporal_edge_linear,
            "exponential":self.select_temporal_edge_exponential,
        }[edge_sampling]

        context_window_count=0
        attempt_count=0
        walks=[]
        while (
                context_window_count< n_context_window
                and attempt_count<max_attempt
            ):
            attempt_count+=1
            # initial temporal edge
            src,dst,t=edge_sampling_fn()
            walk_seq=self.temporal_random_walk(
                source=dst,
                start_t=t,
                walk_len=walk_len-1,
                neighbor_sampling=neighbor_sampling,
            )
            walk_seq=[str(src)]+walk_seq

            # 최소 길이를 만족하지 못하면 폐기
            if len(walk_seq)<min_walk_len:
                continue
            walks.append(walk_seq)

            # 해당 walk_seq가 생성하는 temporal context window 개수 -> 길이가 min_walk_len인 연속 구간을 walk_seq 안에서 몇 번 만들 수 있는지 계산하는 식
            context_window_count+=(
                len(walk_seq)-min_walk_len+1
            )
        return walks