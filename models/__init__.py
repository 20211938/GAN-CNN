"""
모델 모듈
- AprilGAN: 제로샷 이상 탐지 모델
- CNN: 결함 유형 분류 모델
"""

from .aprilgan import AprilGAN
from .cnn import DefectClassifierCNN

__all__ = ['AprilGAN', 'DefectClassifierCNN']

