"""
바운딩박스 처리 유틸리티
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def normalize_defect_type(defect_type: str) -> str:
    """
    결함 유형 이름을 정규화 (공백 제거, 오타 수정 등)
    
    Args:
        defect_type: 원본 결함 유형 이름
        
    Returns:
        정규화된 결함 유형 이름
    """
    if not defect_type:
        return 'Unknown'
    
    # 앞뒤 공백 제거
    normalized = defect_type.strip()
    
    # 연속된 공백을 하나로 통일
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # 알려진 오타 수정
    typo_corrections = {
        'Reocater Streaking': 'Recoater Streaking',
        'Reocater': 'Recoater',
        'Laser capture timing error ': 'Laser capture timing error',
        'Recoater capture timing error ': 'Recoater capture timing error',
    }
    
    # 오타 수정 적용
    for typo, correct in typo_corrections.items():
        if normalized == typo or normalized.startswith(typo):
            normalized = normalized.replace(typo, correct)
    
    # 다시 공백 정리 (오타 수정 후 공백이 생길 수 있음)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized if normalized else 'Unknown'


def extract_bboxes_from_json(json_path: Path) -> Tuple[List[Dict], List[str]]:
    """
    JSON 파일에서 바운딩박스와 결함 유형 추출
    
    Args:
        json_path: JSON 파일 경로
        
    Returns:
        bboxes: 바운딩박스 리스트 [{'x1': int, 'y1': int, 'x2': int, 'y2': int}, ...]
        defect_types: 결함 유형 리스트 ['Super Elevation', ...]
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    bboxes = []
    defect_types = []
    
    # DepositionImageModel에서 바운딩박스 추출
    if 'DepositionImageModel' in metadata:
        for tag in metadata['DepositionImageModel'].get('TagBoxes', []):
            bbox = {
                'x1': tag['StartPoint']['X'],
                'y1': tag['StartPoint']['Y'],
                'x2': tag['EndPoint']['X'],
                'y2': tag['EndPoint']['Y']
            }
            defect_type = tag.get('Comment', tag.get('Name', 'Unknown'))
            # 결함 유형 정규화 (공백 제거, 오타 수정)
            defect_type = normalize_defect_type(defect_type)
            bboxes.append(bbox)
            defect_types.append(defect_type)
    
    # ScanningImageModel에서 바운딩박스 추출
    if 'ScanningImageModel' in metadata:
        for tag in metadata['ScanningImageModel'].get('TagBoxes', []):
            bbox = {
                'x1': tag['StartPoint']['X'],
                'y1': tag['StartPoint']['Y'],
                'x2': tag['EndPoint']['X'],
                'y2': tag['EndPoint']['Y']
            }
            defect_type = tag.get('Comment', tag.get('Name', 'Unknown'))
            # 결함 유형 정규화 (공백 제거, 오타 수정)
            defect_type = normalize_defect_type(defect_type)
            bboxes.append(bbox)
            defect_types.append(defect_type)
    
    return bboxes, defect_types


def bbox_to_mask(image_shape: Tuple[int, int], bboxes: List[Dict]) -> np.ndarray:
    """
    바운딩박스를 이진 마스크로 변환
    
    Args:
        image_shape: 이미지 크기 (height, width)
        bboxes: 바운딩박스 리스트
        
    Returns:
        mask: 이진 마스크 (0: 정상, 1: 결함)
    """
    mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
    for bbox in bboxes:
        y1 = max(0, bbox['y1'])
        y2 = min(image_shape[0], bbox['y2'])
        x1 = max(0, bbox['x1'])
        x2 = min(image_shape[1], bbox['x2'])
        mask[y1:y2, x1:x2] = 1
    return mask


def calculate_iou(bbox1: Dict, bbox2: Dict) -> float:
    """
    두 바운딩박스의 IoU (Intersection over Union) 계산
    
    Args:
        bbox1: 첫 번째 바운딩박스 {'x1': int, 'y1': int, 'x2': int, 'y2': int}
        bbox2: 두 번째 바운딩박스
        
    Returns:
        IoU 값 (0.0 ~ 1.0)
    """
    # 교집합 영역 계산
    x1_inter = max(bbox1['x1'], bbox2['x1'])
    y1_inter = max(bbox1['y1'], bbox2['y1'])
    x2_inter = min(bbox1['x2'], bbox2['x2'])
    y2_inter = min(bbox1['y2'], bbox2['y2'])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    # 각 바운딩박스의 넓이
    bbox1_area = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
    bbox2_area = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])
    
    # 합집합 영역
    union_area = bbox1_area + bbox2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def match_anomaly_regions(
    anomaly_regions: List[Dict],
    ground_truth_bboxes: List[Dict],
    ground_truth_types: List[str],
    iou_threshold: float = 0.3
) -> List[Tuple[Dict, Optional[str]]]:
    """
    AprilGAN이 찾은 이상 영역과 실제 레이블을 매칭
    
    Args:
        anomaly_regions: AprilGAN이 찾은 이상 영역 리스트
        ground_truth_bboxes: 실제 바운딩박스 리스트
        ground_truth_types: 실제 결함 유형 리스트
        iou_threshold: IoU 임계값 (이 값 이상이면 매칭)
        
    Returns:
        매칭된 영역과 레이블 리스트 [(region, defect_type), ...]
    """
    matched = []
    used_gt_indices = set()
    
    for region in anomaly_regions:
        best_iou = 0.0
        best_match_idx = None
        
        # 가장 높은 IoU를 가진 실제 레이블 찾기
        for idx, gt_bbox in enumerate(ground_truth_bboxes):
            if idx in used_gt_indices:
                continue
            
            iou = calculate_iou(region, gt_bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match_idx = idx
        
        if best_match_idx is not None:
            matched.append((region, ground_truth_types[best_match_idx]))
            used_gt_indices.add(best_match_idx)
        else:
            # 매칭되지 않은 경우 (False Positive 또는 새로운 결함)
            matched.append((region, None))
    
    return matched

