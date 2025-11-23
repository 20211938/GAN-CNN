"""
AprilGAN 제로샷 이상 탐지 모델 평가 유틸리티
검출 결과와 Ground Truth를 비교하여 Precision, Recall, F1-Score, IoU 계산
"""

import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
import cv2
from collections import defaultdict

from .bbox_utils import extract_bboxes_from_json, calculate_iou


def evaluate_aprilgan_detection(
    aprilgan_model,
    image_paths: List[Path],
    json_paths: List[Path],
    iou_threshold: float = 0.5
) -> Dict:
    """
    AprilGAN 검출 성능 평가
    
    Args:
        aprilgan_model: AprilGAN 모델 인스턴스
        image_paths: 이미지 파일 경로 리스트
        json_paths: JSON 파일 경로 리스트
        iou_threshold: IoU 임계값 (이 값 이상이면 True Positive)
        
    Returns:
        평가 결과 딕셔너리
    """
    print(f"\n{'='*70}")
    print(f"[AprilGAN 제로샷 모델 평가]")
    print(f"{'='*70}")
    print(f"  ├─ 평가 이미지 수: {len(image_paths)}개")
    print(f"  ├─ IoU 임계값: {iou_threshold}")
    print(f"  └─ 평가 중...\n")
    
    # 통계 변수
    total_detections = 0  # AprilGAN이 검출한 총 영역 수
    total_ground_truth = 0  # Ground Truth 총 영역 수
    true_positives = 0  # 올바르게 검출된 영역 수
    false_positives = 0  # 잘못 검출된 영역 수 (Ground Truth에 없는 영역)
    false_negatives = 0  # 놓친 영역 수 (Ground Truth에 있지만 검출 못함)
    
    # IoU 점수 리스트
    iou_scores = []
    
    # 이미지별 상세 결과
    image_results = []
    
    from tqdm import tqdm
    pbar = tqdm(
        zip(image_paths, json_paths),
        total=len(image_paths),
        desc="AprilGAN 평가",
        unit="image",
        ncols=100
    )
    
    for img_path, json_path in pbar:
        # 이미지 로드
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # AprilGAN으로 이상 영역 검출
        anomaly_result = aprilgan_model.detect(image_rgb)
        detected_regions = anomaly_result.get('anomaly_regions', [])
        
        # JSON에서 Ground Truth 추출
        gt_bboxes, gt_types = extract_bboxes_from_json(json_path)
        
        total_detections += len(detected_regions)
        total_ground_truth += len(gt_bboxes)
        
        # 매칭 수행
        matched_gt_indices = set()
        matched_det_indices = set()
        image_iou_scores = []
        
        # 각 검출 영역에 대해 가장 높은 IoU를 가진 Ground Truth 찾기
        for det_idx, det_bbox in enumerate(detected_regions):
            best_iou = 0.0
            best_gt_idx = None
            
            for gt_idx, gt_bbox in enumerate(gt_bboxes):
                if gt_idx in matched_gt_indices:
                    continue
                
                iou = calculate_iou(det_bbox, gt_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx is not None:
                # True Positive
                true_positives += 1
                matched_gt_indices.add(best_gt_idx)
                matched_det_indices.add(det_idx)
                iou_scores.append(best_iou)
                image_iou_scores.append(best_iou)
            else:
                # False Positive
                false_positives += 1
        
        # False Negative 계산 (매칭되지 않은 Ground Truth)
        false_negatives += len(gt_bboxes) - len(matched_gt_indices)
        
        # 이미지별 결과 저장
        image_results.append({
            'image_path': str(img_path),
            'detections': len(detected_regions),
            'ground_truth': len(gt_bboxes),
            'true_positives': len(matched_det_indices),
            'false_positives': len(detected_regions) - len(matched_det_indices),
            'false_negatives': len(gt_bboxes) - len(matched_gt_indices),
            'avg_iou': np.mean(image_iou_scores) if image_iou_scores else 0.0
        })
        
        pbar.set_postfix({
            'TP': true_positives,
            'FP': false_positives,
            'FN': false_negatives
        })
    
    pbar.close()
    
    # 메트릭 계산
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(iou_scores) if iou_scores else 0.0
    
    # 결과 딕셔너리
    results = {
        'total_images': len(image_paths),
        'total_detections': total_detections,
        'total_ground_truth': total_ground_truth,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'mean_iou': float(mean_iou),
        'iou_threshold': iou_threshold,
        'image_results': image_results
    }
    
    # 결과 출력
    print(f"\n{'='*70}")
    print(f"[AprilGAN 평가 결과]")
    print(f"{'='*70}")
    print(f"\n[검출 통계]")
    print(f"  ├─ 총 검출 영역: {total_detections}개")
    print(f"  ├─ 총 Ground Truth 영역: {total_ground_truth}개")
    print(f"  ├─ True Positives (TP): {true_positives}개")
    print(f"  ├─ False Positives (FP): {false_positives}개")
    print(f"  └─ False Negatives (FN): {false_negatives}개")
    
    print(f"\n[성능 메트릭]")
    print(f"  ├─ Precision: {precision:.4f}")
    print(f"  ├─ Recall: {recall:.4f}")
    print(f"  ├─ F1-Score: {f1_score:.4f}")
    print(f"  └─ Mean IoU: {mean_iou:.4f}")
    print(f"{'='*70}\n")
    
    return results

