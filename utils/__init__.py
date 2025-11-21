"""
유틸리티 모듈
- 데이터 로딩 및 전처리
- 바운딩박스 처리
- 퓨샷 학습 데이터셋
"""

from .data_loader import DefectDataset, load_defect_data
from .bbox_utils import extract_bboxes_from_json, bbox_to_mask, match_anomaly_regions
from .few_shot_dataset import (
    FewShotDataset,
    create_few_shot_loader,
    prepare_few_shot_samples
)
from .client_data_loader import load_client_data
from .non_iid_distribution import distribute_non_iid, analyze_client_distribution

__all__ = [
    'DefectDataset',
    'load_defect_data',
    'extract_bboxes_from_json',
    'bbox_to_mask',
    'match_anomaly_regions',
    'FewShotDataset',
    'create_few_shot_loader',
    'prepare_few_shot_samples',
    'load_client_data',
    'distribute_non_iid',
    'analyze_client_distribution'
]

