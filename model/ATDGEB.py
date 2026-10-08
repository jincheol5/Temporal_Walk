import torch
import torch.nn as nn
from graph import ATDGEB_Graph
from module import SkipGram

class ATDGEB(nn.Module):
    def __init__(self,
            embed_dim:int,
            latent_dim:int,
            window_size:int,
            graph:ATDGEB_Graph
        ):
        super().__init__()
        self.embed_dim=embed_dim
        self.latent_dim=latent_dim
        self.graph=graph

        self.skip_gram=SkipGram(
            vector_size=embed_dim,
            window_size=window_size
        )
        # row 0: padding, row 1 ~ N: 실제 node ID 1 ~ N
        self.node_ft=nn.Embedding(
            num_embeddings=self.graph.n_node+1,
            embedding_dim=self.embed_dim,
            padding_idx=0,
        )
        # downstream 학습 시 노드 임베딩을 고정
        self.node_ft.weight.requires_grad_(False)

        # decoder
        self.decoder=nn.Linear(
            in_features=embed_dim+embed_dim,
            out_features=1
        )

    def convert_SkipGram_to_torch_embedding(self):
        """
        Gensim Skip-gram의 center embedding을 PyTorch 임베딩으로 변환
        CPU에서 수행
        """

        embedding_weight=torch.zeros(
            size=(self.graph.n_node+1,self.embed_dim),
            dtype=torch.float32,
        )

        # row i에 실제 node ID i의 embedding 저장
        for node in range(1,self.graph.n_node+1):
            node_vector=torch.from_numpy(
                self.skip_gram.wv[str(node)].copy()
            ).to(dtype=torch.float32)
            embedding_weight[node].copy_(node_vector)
        
        # 기존 nn.Embedding 객체는 유지하고 weight 값만 복사
        with torch.no_grad():
            self.node_ft.weight.copy_(embedding_weight)
            self.node_ft.weight[0].zero_() # row 0은 항상 padding zero vector
        self.skipgram_trained=False

    def train_skipgram(self,
            k_list:list[int],
            n_aggr:int=2,
            min_points:int=3,
            walk_len:int=5,
            n_lsbs:int=3,
            epoch:int=5
        ):
        """
        """
        ### Walk 생성
        walks=self.graph.generate_walks(
            k_list=k_list,
            n_aggr=n_aggr,
            min_points=min_points,
            walk_len=walk_len,
            n_lsbs=n_lsbs
        )

        ### vocabulary 생성
        all_nodes=[[str(node)] for node in range(0,self.graph.n_node+1)]
        vocab_corpus=walks+all_nodes
        self.skip_gram.build_vocab(vocab_corpus)

        ### Skip-Gram 학습
        self.skip_gram.train(
            corpus_iterable=walks,
            total_examples=self.skip_gram.corpus_count,
            epochs=epoch
        )

        ### Gensim 임베딩을 PyTorch nn.Embedding으로 변환 후 저장
        self.convert_SkipGram_to_torch_embedding()
        self.skipgram_trained=True

    def forward(self,
            src:torch.Tensor,
            dst:torch.Tensor
        ):
        """
        Input:
            src: [B,] 
            dst: [B,] 
        Return:
            pred_logit: [B,1]
        """
        ### 1. current batch에 대한 embedding
        src_ft=self.node_ft(src)
        dst_ft=self.node_ft(dst)
        pair_vec=torch.concat([src_ft,dst_ft],dim=-1) # [B,embed_dim+embed_dim]
        pred_logit=self.decoder(pair_vec) # [B,1]
        return pred_logit
