"""
CNN: 결함 유형 분류 모델
JSON 파일의 TagBoxes에서 추출한 결함 영역을 입력으로 받아 결함 유형을 분류
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional


class DefectClassifierCNN(nn.Module):
    """
    결함 유형 분류를 위한 CNN 모델
    ResNet 기반 백본 사용
    """
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = 'resnet18',
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Args:
            num_classes: 분류할 결함 유형 수
            backbone: 백본 모델 ('resnet18', 'resnet34', 'resnet50')
            pretrained: 사전 학습된 가중치 사용 여부
            dropout: 드롭아웃 비율
        """
        super(DefectClassifierCNN, self).__init__()
        
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
        
        # 특징 추출기 (마지막 FC 레이어 제거)
        self.features = nn.Sequential(*list(backbone_model.children())[:-1])
        
        # 분류기
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 입력 이미지 텐서 (B, C, H, W)
            
        Returns:
            로그its (B, num_classes)
        """
        # 특징 추출
        features = self.features(x)
        
        # 분류
        logits = self.classifier(features)
        
        return logits
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        예측 수행
        
        Args:
            x: 입력 이미지 텐서
            
        Returns:
            {
                'logits': torch.Tensor,
                'probs': torch.Tensor,
                'pred_class': torch.Tensor
            }
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
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


def create_cnn_model(
    num_classes: int,
    backbone: str = 'resnet18',
    pretrained: bool = True
) -> DefectClassifierCNN:
    """
    CNN 모델 생성 헬퍼 함수
    
    Args:
        num_classes: 분류할 결함 유형 수
        backbone: 백본 모델 이름
        pretrained: 사전 학습된 가중치 사용 여부
        
    Returns:
        생성된 CNN 모델
    """
    model = DefectClassifierCNN(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained
    )
    return model

