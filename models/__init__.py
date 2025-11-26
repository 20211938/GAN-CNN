"""
모델 모듈
- CLIPDefectDetector: CLIP 기반 결함 탐지 모델
- CNN: 결함 유형 분류 모델
- FewShotCNN: 퓨샷 학습 전용 모델
- HybridCNN: 일반 학습과 퓨샷 학습을 모두 지원하는 하이브리드 모델
"""

from .clip_defect_detector import CLIPDefectDetector
from .cnn import DefectClassifierCNN, create_cnn_model
from .few_shot_cnn import FewShotCNN, HybridCNN, create_hybrid_cnn_model

__all__ = [
    'CLIPDefectDetector',
    'DefectClassifierCNN',
    'create_cnn_model',
    'FewShotCNN',
    'HybridCNN',
    'create_hybrid_cnn_model'
]

