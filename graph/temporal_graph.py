import random
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

class TemporalGraph:
    def __init__(self,
            graph_df:pd.DataFrame,
            bipartite:bool=False
        ):
        self.graph_df=graph_df

        ### graph information
        self.n_node=int(max(graph_df["u"].max(),graph_df["i"].max()))
        self.n_event=int(graph_df["idx"].max())
        self.bipartite=bipartite
        self.max_u=int(graph_df["u"].max())
        self.max_t=float(graph_df["t"].max())

        ### save adj info
        self.edge_events=[]
        adj=[[] for _ in range(self.n_node+1)]
        adj_edge=[[] for _ in range(self.n_node+1)]
        adj_t=[[] for _ in range(self.n_node+1)]
        for event in graph_df.itertuples(index=False): # col: [u,i,t,idx=edge_id]
            src=int(event.u)
            dst=int(event.i)
            t=float(event.t)
            edge_id=int(event.idx)
            # edge event 저장
            self.edge_events.append((src,dst,t,edge_id))
            # edge 양방향 저장
            adj[src].append(dst)
            adj_edge[src].append(edge_id)
            adj_t[src].append(t)
            adj[dst].append(src)
            adj_edge[dst].append(edge_id)
            adj_t[dst].append(t)
        # list -> numpy array
        self.adj=[
            np.asarray(values,dtype=np.int64)
            for values in adj
        ]
        self.adj_edge=[
            np.asarray(values,dtype=np.int64)
            for values in adj_edge
        ]
        self.adj_t=[
            np.asarray(values,dtype=np.float64)
            for values in adj_t
        ]

    def set_random_seed(self,seed:int):
        self.rng=random.Random(seed)

    def get_num_node(self):
        return self.n_node

    def get_num_event(self):
        return self.n_event