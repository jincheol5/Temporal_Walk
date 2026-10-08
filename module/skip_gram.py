from multiprocessing import cpu_count
from gensim.models import Word2Vec

"""
DeepWalk github code 수정 버전

Gensim = CPU 학습
"""
class SkipGram(Word2Vec):
    def __init__(
            self,
            vector_size:int=128,
            window_size:int=10,
            workers:int|None=None,
            **kwargs,
        ):
        super().__init__(
            vector_size=vector_size, # 노드 임베딩 차원
            window=window_size, # 중심 노드에서 문맥 노드를 선택하는 최대 거리
            min_count=0, # sequence에서 일정 횟수보다 적게 등장한 토큰을 제거하는 기준, graph의 경우 0으로 고정
            # workers=workers or cpu_count(), # Word2Vec 학습에 사용할 CPU worker thread 수
            workers=1, # 멀티스레딩 환경에서 Gensim의 내부 연산(our_dot_float) 충돌 문제 해결을 위해 1로 설정
            sg=1, # skip-gram 설정
            hs=1, # Hierarchical Softmax 설정
            negative=0, # Negative Sampling 사용 안함 설정
            **kwargs,
        )