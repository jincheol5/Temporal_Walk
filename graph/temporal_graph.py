import random
import pandas as pd
import torch
from typing import Literal
from collections import defaultdict

class TemporalGraph:
    """
    node_id=0: padding node
    """
    def __init__(self,
            graph_df:pd.DataFrame,
            bipartite:bool=False
        ):
        """
        Input:
            graph_df: pd.DataFrame
            bipartite: bool
        """
        # set graph_df, adj, adj_t, edge_events
        self.graph_df=graph_df
        self.adj=defaultdict(list)
        self.adj_t=defaultdict(list)
        self.edge_events=[]
        for event in graph_df.itertuples(index=False): # col: [u,i,ts]
            src=int(event.u)
            dst=int(event.i)
            t=float(event.ts)
            # edge 양방향 저장
            self.adj[dst].append(src)
            self.adj_t[dst].append(t)
            self.adj[src].append(dst)
            self.adj_t[src].append(t)
            self.edge_events.append((src,dst,t))
            self.edge_events.append((dst,src,t))

        # set n_node, bipartite, max_u
        self.n_node=max(graph_df["u"].max(),graph_df["i"].max())
        self.bipartite=bipartite
        self.max_u=graph_df["u"].max()
        self.max_t=graph_df["ts"].max()

    def set_random_seed(self,
            seed:int
        ):
        self.rng=random.Random(seed)

    def random_negative_sampling(self,
            src:torch.Tensor,
            event_t:torch.Tensor,
        ):
        """
        """
        if self.bipartite:
            candidates=range(int(self.max_u)+1,int(self.n_node))
        else:
            # node_id=0 is reserved for padding.
            candidates=range(1,int(self.n_node)+1)

        src_list=src.detach().cpu().tolist()
        t_list=event_t.detach().cpu().tolist()

        neg_dst=[]
        for source,cur_t in zip(src_list,t_list):
            source=int(source)
            cur_t=float(cur_t)

            neighbors=self.adj.get(source,[])
            neighbor_times=self.adj_t.get(source,[])

            # 정확히 cur_t에 연결된 이웃만 positive로 처리
            positive_dst_at_cur_t={
                neighbor
                for neighbor,edge_t in zip(neighbors,neighbor_times)
                if edge_t==cur_t
            }
            valid_candidates=[
                node
                for node in candidates
                if node not in positive_dst_at_cur_t
                and node!=source
            ]
            neg_dst.append(self.rng.choice(valid_candidates))
        return torch.tensor(
            neg_dst,
            dtype=src.dtype,
            device=src.device
        )