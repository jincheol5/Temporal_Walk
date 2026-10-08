import os
import pandas as pd
import numpy as np
import torch
from typing import Literal

class DataUtils:
    base_path=os.path.join('..','data','temporal_graph')

    @staticmethod
    def _preprocess_snap_dataset(
            dataset_name:Literal[
                "CollegeMsg"
                "bitcoin-otc",
                "bitcoin-alpha"
            ]
        ):
        """
        timestamp: UTC unix timestamp day 단위로 변환
        """
        match dataset_name:
            case "CollegeMsg":
                dataset_path=os.path.join(DataUtils.base_path,dataset_name,f"{dataset_name}.txt")
                graph_df=pd.read_csv(
                    dataset_path,
                    header=None,
                    sep=r"\s+",
                    usecols=[0,1,2],
                    names=["u","i","t"],
                )
            case "bitcoin-otc"|"bitcoin-alpha":
                dataset_path=os.path.join(DataUtils.base_path,dataset_name,f"{dataset_name}.csv")
                graph_df=pd.read_csv(
                    dataset_path,
                    header=None,
                    usecols=[0,1,3],
                    names=["u","i","t"],
                )

        # 숫자로 변환할 수 없는 값은 결측값으로 처리한 뒤 제거
        graph_df=graph_df[["u","i","t"]].apply(
            pd.to_numeric,
            errors="coerce"
        )
        graph_df=(
            graph_df.dropna(subset=["u","i","t"])
            .astype(np.int64)
            .reset_index(drop=True)
        )

        # self-loop 제거
        self_loop_mask=graph_df["u"]==graph_df["i"]
        graph_df=graph_df.loc[~self_loop_mask].copy()

        # Unix timestamp(초)를 day 단위로 변환하고 최초 시각을 0으로 조정
        graph_df["t"]=(graph_df["t"]//(24*60*60)).astype(np.int64)
        min_event_time=graph_df["t"].min()
        graph_df["t"]=(graph_df["t"]-min_event_time).astype(np.int64)

        # 같은 방향의 동일 edge event는 하나만 유지
        graph_df=graph_df.drop_duplicates(
            subset=["u","i","t"],
            keep="first"
        )

        # 동일 시각의 a→b와 b→a는 undirected edge 하나로 취급
        graph_df["_node_min"]=graph_df[["u","i"]].min(axis=1)
        graph_df["_node_max"]=graph_df[["u","i"]].max(axis=1)
        graph_df=(
            graph_df.drop_duplicates(
                subset=["_node_min","_node_max","t"],
                keep="first"
            )
            .drop(columns=["_node_min","_node_max"])
        )

        # event time 순으로 정렬
        graph_df=(
            graph_df.sort_values("t",kind="stable")
            .reset_index(drop=True)
        )

        # 모든 node ID 수집
        node_ids=sorted(
            set(graph_df["u"]).union(graph_df["i"])
        )

        # 기존 node ID -> 1 ~ N 매핑
        node_mapping={
            original_id:mapped_id
            for mapped_id,original_id in enumerate(
                node_ids,
                start=1,
            )
        }

        # node ID 재매핑
        graph_df["u"]=graph_df["u"].map(node_mapping).astype(np.int64)
        graph_df["i"]=graph_df["i"].map(node_mapping).astype(np.int64)

        # 중복 제거된 시간순 event에 맞춰 edge ID를 1 ~ E로 재매핑
        graph_df["idx"]=np.arange(1,len(graph_df)+1,dtype=np.int64)

        return {
            "graph_df":graph_df,
            "n_node":len(node_ids),
            "max_u":graph_df["u"].max(),
            "node_dim":None,
            "edge_dim":None,
            "node_ft":None,
            "edge_ft":None
        }

    @staticmethod
    def _preprocess_zenodo_dataset(
            dataset_name:Literal[
                    "enron",
                    "wikipedia",
                    "reddit"
                ]
        ):
        """
        timestamp: UTC unix timestamp day 단위로 변환
        """
        graph_path=os.path.join(DataUtils.base_path,dataset_name,f"ml_{dataset_name}.csv")
        node_ft_path=os.path.join(DataUtils.base_path,dataset_name,f"ml_{dataset_name}_node.npy")
        edge_ft_path=os.path.join(DataUtils.base_path,dataset_name,f"ml_{dataset_name}.npy")

        graph_df=pd.read_csv(
            graph_path,
            index_col=0,
        )[["u","i","ts","idx"]].rename(
            columns={"ts": "t"}
        )

        node_ft=np.load(node_ft_path)
        edge_ft=np.load(edge_ft_path)

        # timestamp(초)를 Unix timestamp의 일(day) 단위 정수로 변환
        graph_df["t"]=(
            pd.to_numeric(graph_df["t"],errors="raise")
            .floordiv(24*60*60)
            .astype(np.int64)
        )

        # 최초 event 시각을 0으로 맞춘 상대 Unix timestamp(day)로 변환
        min_event_time=graph_df["t"].min()
        graph_df["t"]=(graph_df["t"]-min_event_time).astype(np.int64)

        ### remove self-loop, 동일한 시각의 동일한 방향 edge(u -> i)는 하나만 유지
        graph_df=(
            graph_df[graph_df["u"]!=graph_df["i"]]
            .drop_duplicates(subset=["u","i","t"],keep="first")
            .reset_index(drop=True)
        )

        # 동일 시각의 a→b와 b→a는 undirected edge 하나로 취급
        graph_df["_node_min"]=graph_df[["u","i"]].min(axis=1)
        graph_df["_node_max"]=graph_df[["u","i"]].max(axis=1)
        graph_df=(
            graph_df.drop_duplicates(
                subset=["_node_min","_node_max","t"],
                keep="first"
            )
            .drop(columns=["_node_min","_node_max"])
            .reset_index(drop=True)
        )

        ### remap edge index
        edge_ft=np.vstack([
            np.zeros(
                (1,edge_ft.shape[1]),
                dtype=edge_ft.dtype
            ),
            edge_ft[graph_df["idx"].to_numpy()]
        ])
        graph_df["idx"]=np.arange(1,len(graph_df)+1)

        # remap node id
        used_nodes=np.sort(np.unique(graph_df[["u","i"]].to_numpy()))
        node_map={
            old_id:new_id
            for new_id,old_id
            in enumerate(used_nodes,start=1)
        }
        graph_df[["u","i"]]=(
            graph_df[["u","i"]]
            .replace(node_map)
            .astype(int)
        )
        node_ft=np.vstack([
            np.zeros(
                (1,node_ft.shape[1]),
                dtype=node_ft.dtype
            ),
            node_ft[used_nodes]
        ])

        return {
            "graph_df": graph_df,
            "n_node": len(used_nodes),
            "max_u": graph_df["u"].max(),
            "node_dim": node_ft.shape[1],
            "edge_dim": edge_ft.shape[1],
            "node_ft": node_ft,
            "edge_ft": edge_ft
        }

    @staticmethod
    def preprocess_graph_dataset(
            dataset_name:Literal[
                "CollegeMsg",
                "bitcoin-otc",
                "bitcoin-alpha",
                "enron"
            ]
        ):
        """
        """
        match dataset_name:
            case "CollegeMsg"|"bitcoin-otc"|"bitcoin-alpha":
                return DataUtils._preprocess_snap_dataset(dataset_name=dataset_name)
            case "enron":
                return DataUtils._preprocess_zenodo_dataset(dataset_name=dataset_name)