import numpy as np

class SamplingUtils:
    @staticmethod
    def compute_similarity(
            vec_a:np.ndarray,
            vec_b:np.ndarray
        )->float:
        """
        두 local structure vector의 cosine similarity를 계산합니다.
        결과를 0.0 ~ 1,0 범위로 제한합니다.
        두 벡터 중 하나가 zero vector이면 0을 반환합니다.
        """
        norm_a=np.linalg.norm(vec_a)
        norm_b=np.linalg.norm(vec_b)
        if norm_a==0 or norm_b==0:
            return 0.0
        similarity=np.dot(
            vec_a,
            vec_b
        ) / (norm_a * norm_b)
        # 부동소수점 오차 및 음수 가중치 방지
        return float(
            np.clip(
                similarity,
                0.0,
                1.0
            )
        )
