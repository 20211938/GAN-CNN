"""
연합학습 가중치 집계 알고리즘
Federated Averaging (FedAvg) 구현
"""

import torch
from typing import List, Dict
import copy


class FederatedAveraging:
    """
    Federated Averaging 알고리즘
    여러 클라이언트의 가중치를 평균화
    """
    
    @staticmethod
    def aggregate(
        client_weights: List[Dict],
        client_sizes: List[int] = None
    ) -> Dict:
        """
        클라이언트 가중치를 평균화
        
        Args:
            client_weights: 클라이언트별 모델 가중치 리스트
            client_sizes: 클라이언트별 데이터 크기 리스트 (가중 평균용)
            
        Returns:
            평균화된 가중치 딕셔너리
        """
        if len(client_weights) == 0:
            raise ValueError("클라이언트 가중치가 없습니다")
        
        # 가중치가 지정되지 않으면 균등 가중치 사용
        if client_sizes is None:
            client_sizes = [1] * len(client_weights)
        
        # 총 데이터 크기 계산
        total_size = sum(client_sizes)
        
        # 첫 번째 클라이언트의 가중치 구조 복사
        aggregated_weights = copy.deepcopy(client_weights[0])
        
        # 각 파라미터에 대해 가중 평균 계산
        for key in aggregated_weights.keys():
            # 가중 평균 초기화
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])
            
            # 각 클라이언트의 가중치를 가중치로 합산
            for client_idx, weights in enumerate(client_weights):
                weight = client_sizes[client_idx] / total_size
                aggregated_weights[key] += weight * weights[key].float()
        
        return aggregated_weights
    
    @staticmethod
    def weighted_average(
        client_weights: List[Dict],
        weights: List[float]
    ) -> Dict:
        """
        사용자 정의 가중치로 평균화
        
        Args:
            client_weights: 클라이언트별 모델 가중치 리스트
            weights: 각 클라이언트에 대한 가중치 리스트
            
        Returns:
            가중 평균화된 가중치 딕셔너리
        """
        if len(client_weights) != len(weights):
            raise ValueError("가중치와 클라이언트 수가 일치하지 않습니다")
        
        # 가중치 정규화
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # 첫 번째 클라이언트의 가중치 구조 복사
        aggregated_weights = copy.deepcopy(client_weights[0])
        
        # 각 파라미터에 대해 가중 평균 계산
        for key in aggregated_weights.keys():
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])
            
            for client_idx, client_weight in enumerate(client_weights):
                weight = normalized_weights[client_idx]
                aggregated_weights[key] += weight * client_weight[key].float()
        
        return aggregated_weights

