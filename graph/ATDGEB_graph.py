import random
import math
import numpy as np
import networkx as nx
from typing import Literal
from .temporal_graph import TemporalGraph
from .ATDGEB_LSBS import LSBS

class ATDGEB_Graph(TemporalGraph):
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
        self.LSBS=LSBS(
            n_node=self.n_node,
            topological_graph=self.topological_graph
        )



