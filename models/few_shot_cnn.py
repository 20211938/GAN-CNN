"""
퓨샷 학습을 지원하는 CNN 모델
Prototypical Networks 방식 구현
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Tuple, Optional
import numpy as np


class FewShotCNN(nn.Module):
    """
    퓨샷 학습을 지원하는 CNN 모델
    Prototypical Networks 방식 사용
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        feature_dim: int = 512
    ):
        """
        Args:
            backbone: 백본 모델 ('resnet18', 'resnet34', 'resnet50')
            pretrained: 사전 학습된 가중치 사용 여부
            feature_dim: 특징 벡터 차원
        """
        super(FewShotCNN, self).__init__()
        
        self.backbone_name = backbone
        self.feature_dim = feature_dim
        
        # 백본 모델 선택
        if backbone == 'resnet18':
            backbone_model = models.resnet18(pretrained=pretrained)
            num_features = 512
        elif backbone == 'resnet34':
            backbone_model = models.resnet34(pretrained=pretrained)
            num_features = 512
        elif backbone == 'resnet50':
            backbone_model = models.resnet50(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"지원하지 않는 백본: {backbone}")
        
        # 특징 추출기 (마지막 FC 레이어 제거)
        self.features = nn.Sequential(*list(backbone_model.children())[:-1])
        
        # 특징 벡터 정규화를 위한 레이어
        self.feature_norm = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, feature_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        특징 벡터 추출
        
        Args:
            x: 입력 이미지 텐서 (B, C, H, W)
            
        Returns:
            특징 벡터 (B, feature_dim)
        """
        # 특징 추출
        features = self.features(x)
        
        # 특징 벡터로 변환 및 정규화
        feature_vector = self.feature_norm(features)
        feature_vector = F.normalize(feature_vector, p=2, dim=1)
        
        return feature_vector
    
    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Support set으로부터 각 클래스의 Prototype 계산
        
        Args:
            support_features: Support set 특징 벡터 (N, feature_dim)
            support_labels: Support set 레이블 (N,)
            
        Returns:
            각 클래스의 Prototype 딕셔너리 {class_id: prototype_vector}
        """
        prototypes = {}
        unique_labels = torch.unique(support_labels)
        
        for label in unique_labels:
            # 해당 클래스의 모든 샘플 선택
            mask = (support_labels == label)
            class_features = support_features[mask]
            
            # Prototype 계산 (평균)
            prototype = class_features.mean(dim=0)
            prototypes[int(label.item())] = prototype
        
        return prototypes
    
    def predict_few_shot(
        self,
        query_features: torch.Tensor,
        prototypes: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query를 Prototype과 비교하여 분류
        
        Args:
            query_features: Query 특징 벡터 (M, feature_dim)
            prototypes: 각 클래스의 Prototype 딕셔너리
            
        Returns:
            (예측 클래스, 거리)
        """
        # Prototype을 리스트로 변환
        prototype_labels = list(prototypes.keys())
        prototype_vectors = torch.stack([prototypes[label] for label in prototype_labels])
        prototype_vectors = prototype_vectors.to(query_features.device)
        
        # 거리 계산 (Euclidean distance)
        # query_features: (M, feature_dim)
        # prototype_vectors: (N_classes, feature_dim)
        distances = torch.cdist(query_features, prototype_vectors, p=2)
        
        # 가장 가까운 Prototype 선택
        _, predicted_indices = torch.min(distances, dim=1)
        predicted_labels = torch.tensor([prototype_labels[idx] for idx in predicted_indices])
        
        return predicted_labels.to(query_features.device), distances
    
    def few_shot_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        퓨샷 학습 에피소드 수행
        
        Args:
            support_images: Support set 이미지 (N, C, H, W)
            support_labels: Support set 레이블 (N,)
            query_images: Query set 이미지 (M, C, H, W)
            query_labels: Query set 레이블 (M,) - 선택적, 평가용
            
        Returns:
            {
                'predicted_labels': 예측된 레이블,
                'distances': 거리,
                'accuracy': 정확도 (query_labels가 제공된 경우)
            }
        """
        self.eval()
        
        with torch.no_grad():
            # 특징 벡터 추출
            support_features = self.forward(support_images)
            query_features = self.forward(query_images)
            
            # Prototype 계산
            prototypes = self.compute_prototypes(support_features, support_labels)
            
            # 분류
            predicted_labels, distances = self.predict_few_shot(query_features, prototypes)
            
            result = {
                'predicted_labels': predicted_labels,
                'distances': distances
            }
            
            # 정확도 계산 (레이블이 제공된 경우)
            if query_labels is not None:
                correct = (predicted_labels == query_labels).float()
                accuracy = correct.mean()
                result['accuracy'] = accuracy
                result['correct'] = correct.sum().item()
                result['total'] = len(query_labels)
        
        return result


class HybridCNN(nn.Module):
    """
    일반 학습과 퓨샷 학습을 모두 지원하는 하이브리드 CNN 모델
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        dropout: float = 0.5,
        feature_dim: int = 512
    ):
        """
        Args:
            num_classes: 분류할 결함 유형 수 (일반 학습 모드용)
            backbone: 백본 모델
            pretrained: 사전 학습된 가중치 사용 여부
            dropout: 드롭아웃 비율
            feature_dim: 특징 벡터 차원 (퓨샷 학습용)
        """
        super(HybridCNN, self).__init__()
        
        self.num_classes = num_classes
        self.backbone_name = backbone
        
        # 백본 모델 선택
        if backbone == 'resnet18':
            backbone_model = models.resnet18(pretrained=pretrained)
            num_features = 512
        elif backbone == 'resnet34':
            backbone_model = models.resnet34(pretrained=pretrained)
            num_features = 512
        elif backbone == 'resnet50':
            backbone_model = models.resnet50(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"지원하지 않는 백본: {backbone}")
        
        # 특징 추출기 (공통)
        self.features = nn.Sequential(*list(backbone_model.children())[:-1])
        
        # 일반 학습용 분류기
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # 퓨샷 학습용 특징 추출기
        self.few_shot_features = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, feature_dim)
        )
    
    def forward(self, x: torch.Tensor, mode: str = 'normal') -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 입력 이미지 텐서 (B, C, H, W)
            mode: 'normal' (일반 학습) 또는 'few_shot' (퓨샷 학습)
            
        Returns:
            mode='normal': 로그its (B, num_classes)
            mode='few_shot': 특징 벡터 (B, feature_dim)
        """
        # 특징 추출
        features = self.features(x)
        
        if mode == 'few_shot':
            # 퓨샷 학습 모드: 특징 벡터 반환
            feature_vector = self.few_shot_features(features)
            feature_vector = F.normalize(feature_vector, p=2, dim=1)
            return feature_vector
        else:
            # 일반 학습 모드: 분류 결과 반환
            logits = self.classifier(features)
            return logits
    
    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """Prototype 계산 (FewShotCNN과 동일)"""
        prototypes = {}
        unique_labels = torch.unique(support_labels)
        
        for label in unique_labels:
            mask = (support_labels == label)
            class_features = support_features[mask]
            prototype = class_features.mean(dim=0)
            prototypes[int(label.item())] = prototype
        
        return prototypes
    
    def predict_few_shot(
        self,
        query_features: torch.Tensor,
        prototypes: Dict[int, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """퓨샷 학습 예측 (FewShotCNN과 동일)"""
        prototype_labels = list(prototypes.keys())
        prototype_vectors = torch.stack([prototypes[label] for label in prototype_labels])
        prototype_vectors = prototype_vectors.to(query_features.device)
        
        distances = torch.cdist(query_features, prototype_vectors, p=2)
        _, predicted_indices = torch.min(distances, dim=1)
        predicted_labels = torch.tensor([prototype_labels[idx] for idx in predicted_indices])
        
        return predicted_labels.to(query_features.device), distances
    
    def few_shot_episode(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        """퓨샷 학습 에피소드 수행"""
        self.eval()
        
        with torch.no_grad():
            # 특징 추출
            support_features = self.forward(support_images, mode='few_shot')
            query_features = self.forward(query_images, mode='few_shot')
            
            # Prototype 계산
            prototypes = self.compute_prototypes(support_features, support_labels)
            
            if verbose:
                print(f"    ├─ Support 샘플 수: {len(support_images)}개")
                print(f"    ├─ Query 샘플 수: {len(query_images)}개")
                print(f"    ├─ Prototype 수: {len(prototypes)}개 클래스")
            
            # 분류
            predicted_labels, distances = self.predict_few_shot(query_features, prototypes)
            
            result = {
                'predicted_labels': predicted_labels,
                'distances': distances
            }
            
            if query_labels is not None:
                correct = (predicted_labels == query_labels).float()
                accuracy = correct.mean()
                result['accuracy'] = accuracy
                result['correct'] = correct.sum().item()
                result['total'] = len(query_labels)
                
                if verbose:
                    print(f"    ├─ 정확도: {accuracy:.4f}")
                    print(f"    └─ 정확히 분류: {result['correct']}/{result['total']}")
        
        return result
    
    def predict(self, x: torch.Tensor, mode: str = 'normal') -> Dict[str, torch.Tensor]:
        """
        예측 수행
        
        Args:
            x: 입력 이미지 텐서
            mode: 'normal' 또는 'few_shot'
            
        Returns:
            예측 결과 딕셔너리
        """
        self.eval()
        with torch.no_grad():
            if mode == 'few_shot':
                features = self.forward(x, mode='few_shot')
                return {'features': features}
            else:
                logits = self.forward(x, mode='normal')
                probs = F.softmax(logits, dim=1)
                pred_class = torch.argmax(probs, dim=1)
                
                return {
                    'logits': logits,
                    'probs': probs,
                    'pred_class': pred_class
                }
    
    def get_state_dict(self) -> Dict:
        """모델 가중치 반환 (연합학습용)"""
        return self.state_dict()
    
    def load_state_dict(self, state_dict: Dict):
        """모델 가중치 로드 (연합학습용)"""
        super().load_state_dict(state_dict)


def create_hybrid_cnn_model(
    num_classes: int,
    backbone: str = 'resnet18',
    pretrained: bool = True
) -> HybridCNN:
    """
    하이브리드 CNN 모델 생성 헬퍼 함수
    
    Args:
        num_classes: 분류할 결함 유형 수
        backbone: 백본 모델 이름
        pretrained: 사전 학습된 가중치 사용 여부
        
    Returns:
        생성된 하이브리드 CNN 모델
    """
    model = HybridCNN(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained
    )
    return model

