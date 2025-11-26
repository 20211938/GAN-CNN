"""
CLIP 모델 결함 검출 성능 평가 및 시각화 스크립트
"""

import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

from models.clip_defect_detector import CLIPDefectDetector
from utils.bbox_utils import extract_bboxes_from_json, normalize_defect_type, calculate_iou


def visualize_detection_results(
    image: np.ndarray,
    gt_bboxes: List[Dict],
    gt_types: List[str],
    detected_regions: List[Dict],
    defect_type_scores: Dict[str, float],
    save_path: Optional[Path] = None,
    show: bool = True
):
    """
    검출 결과를 시각화
    
    Args:
        image: 원본 이미지
        gt_bboxes: Ground Truth 바운딩 박스 리스트
        gt_types: Ground Truth 결함 유형 리스트
        detected_regions: 검출된 이상 영역 리스트
        defect_type_scores: 결함 유형별 점수
        save_path: 저장 경로 (None이면 저장 안 함)
        show: 화면에 표시 여부
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 원본 이미지에 GT 표시
    ax1 = axes[0]
    ax1.imshow(image)
    ax1.set_title('Ground Truth', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # GT 박스 표시
    for bbox, defect_type in zip(gt_bboxes, gt_types):
        x1, y1 = bbox['x1'], bbox['y1']
        w, h = bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1']
        rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='green', 
                       facecolor='none', label=f'GT: {defect_type}')
        ax1.add_patch(rect)
        ax1.text(x1, y1 - 5, defect_type, color='green', fontsize=10, 
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 검출 결과 표시
    ax2 = axes[1]
    ax2.imshow(image)
    ax2.set_title('CLIP Detection Results', fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # 검출된 박스 표시
    for i, region in enumerate(detected_regions):
        x1, y1 = region['x1'], region['y1']
        w, h = region['x2'] - region['x1'], region['y2'] - region['y1']
        score = region.get('score', 0.0)
        rect = Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', 
                        facecolor='none', alpha=0.7)
        ax2.add_patch(rect)
        ax2.text(x1, y1 - 5, f'Score: {score:.3f}', color='red', fontsize=10,
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # 결함 유형별 점수 표시
    score_text = "Defect Type Scores:\n"
    for defect_type, score in sorted(defect_type_scores.items(), key=lambda x: x[1], reverse=True):
        score_text += f"  {defect_type}: {score:.3f}\n"
    
    ax2.text(10, image.shape[0] - 20, score_text, color='black', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            verticalalignment='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  💾 저장: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def evaluate_clip_detector(
    data_dir: Path,
    clip_model: CLIPDefectDetector,
    output_dir: Optional[Path] = None,
    max_images: Optional[int] = None,
    visualize: bool = True,
    save_images: bool = True,
    iou_threshold: float = 0.5
) -> Dict:
    """
    CLIP 모델의 결함 검출 성능 평가
    
    Args:
        data_dir: 데이터 디렉토리
        clip_model: CLIP 모델
        output_dir: 결과 저장 디렉토리
        max_images: 평가할 최대 이미지 수 (None이면 전체)
        visualize: 시각화 여부
        save_images: 이미지 저장 여부
        iou_threshold: IoU 임계값
        
    Returns:
        평가 결과 딕셔너리
    """
    print(f"\n{'='*70}")
    print(f"CLIP 모델 결함 검출 성능 평가")
    print(f"{'='*70}")
    print(f"  ├─ 데이터 디렉토리: {data_dir}")
    print(f"  ├─ 최대 이미지 수: {max_images if max_images else '전체'}")
    print(f"  ├─ IoU 임계값: {iou_threshold}")
    print(f"  └─ 시각화: {'활성화' if visualize else '비활성화'}")
    print(f"{'='*70}\n")
    
    # 출력 디렉토리 생성
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        if save_images:
            vis_dir = output_dir / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 찾기
    image_files = list(data_dir.glob("*.jpg"))
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"[1단계] 이미지 파일 검색 완료: {len(image_files)}개\n")
    
    # 통계 변수
    total_images = 0
    images_with_gt = 0
    images_with_detections = 0
    
    # 검출 통계
    total_gt_boxes = 0
    total_detected_boxes = 0
    matched_pairs = []
    
    # 결함 유형별 통계
    defect_type_stats = defaultdict(lambda: {
        'gt_count': 0,
        'detected_count': 0,
        'matched_count': 0,
        'scores': []
    })
    
    # 이미지별 결과
    image_results = []
    
    from tqdm import tqdm
    pbar = tqdm(image_files, desc="평가 진행", unit="image", ncols=100)
    
    for img_path in pbar:
        json_path = img_path.with_suffix(".jpg.json")
        if not json_path.exists():
            continue
        
        total_images += 1
        
        # 이미지 로드
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ground Truth 추출
        gt_bboxes, gt_types = extract_bboxes_from_json(json_path)
        
        if len(gt_bboxes) == 0:
            continue
        
        images_with_gt += 1
        total_gt_boxes += len(gt_bboxes)
        
        # 결함 유형 수집
        defect_types = list(set(gt_types))
        defect_types = [normalize_defect_type(dt) for dt in defect_types]
        
        # CLIP으로 결함 검출
        try:
            detection_result = clip_model.detect(
                image_rgb,
                defect_types=defect_types,
                bboxes=None  # 박스 없이 전체 이미지 기반 검출
            )
        except Exception as e:
            print(f"\n⚠️  검출 실패 ({img_path.name}): {e}")
            continue
        
        detected_regions = detection_result.get('anomaly_regions', [])
        defect_type_scores = detection_result.get('defect_type_scores', {})
        
        if len(detected_regions) > 0:
            images_with_detections += 1
        
        total_detected_boxes += len(detected_regions)
        
        # GT와 검출 결과 매칭 (IoU 기반)
        matched = []
        unmatched_gt = list(range(len(gt_bboxes)))
        unmatched_det = list(range(len(detected_regions)))
        
        for i, gt_bbox in enumerate(gt_bboxes):
            gt_type = normalize_defect_type(gt_types[i])
            defect_type_stats[gt_type]['gt_count'] += 1
            
            best_iou = 0
            best_j = -1
            
            for j, det_region in enumerate(detected_regions):
                if j not in unmatched_det:
                    continue
                
                iou = calculate_iou(gt_bbox, det_region)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            
            if best_iou >= iou_threshold and best_j != -1:
                matched.append({
                    'gt_idx': i,
                    'det_idx': best_j,
                    'iou': best_iou,
                    'gt_type': gt_type
                })
                unmatched_gt.remove(i)
                unmatched_det.remove(best_j)
                defect_type_stats[gt_type]['matched_count'] += 1
        
        # 매칭되지 않은 검출 결과도 통계에 추가
        for j in unmatched_det:
            det_region = detected_regions[j]
            score = det_region.get('score', 0.0)
            # 가장 높은 점수의 결함 유형 찾기
            if defect_type_scores:
                best_type = max(defect_type_scores.items(), key=lambda x: x[1])[0]
                defect_type_stats[best_type]['detected_count'] += 1
                defect_type_stats[best_type]['scores'].append(score)
        
        matched_pairs.extend(matched)
        
        # 이미지별 결과 저장
        image_result = {
            'image_path': str(img_path),
            'gt_count': len(gt_bboxes),
            'detected_count': len(detected_regions),
            'matched_count': len(matched),
            'matched_pairs': matched,
            'unmatched_gt': unmatched_gt,
            'unmatched_det': unmatched_det
        }
        image_results.append(image_result)
        
        # 시각화
        if visualize and (save_images or len(image_results) <= 5):
            vis_path = None
            if save_images and output_dir:
                vis_path = vis_dir / f"{img_path.stem}_detection.png"
            
            visualize_detection_results(
                image_rgb,
                gt_bboxes,
                gt_types,
                detected_regions,
                defect_type_scores,
                save_path=vis_path,
                show=(len(image_results) <= 5)  # 처음 5개만 화면 표시
            )
        
        pbar.set_postfix({
            'GT': total_gt_boxes,
            '검출': total_detected_boxes,
            '매칭': len(matched_pairs)
        })
    
    pbar.close()
    
    # 성능 메트릭 계산
    print(f"\n{'='*70}")
    print(f"[2단계] 성능 메트릭 계산")
    print(f"{'='*70}\n")
    
    # 전체 메트릭
    precision = len(matched_pairs) / total_detected_boxes if total_detected_boxes > 0 else 0.0
    recall = len(matched_pairs) / total_gt_boxes if total_gt_boxes > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 평균 IoU
    avg_iou = np.mean([m['iou'] for m in matched_pairs]) if matched_pairs else 0.0
    
    print(f"📊 전체 성능 메트릭:")
    print(f"  ├─ 평가 이미지 수: {total_images}개")
    print(f"  ├─ GT 박스 수: {total_gt_boxes}개")
    print(f"  ├─ 검출 박스 수: {total_detected_boxes}개")
    print(f"  ├─ 매칭된 박스 수: {len(matched_pairs)}개")
    print(f"  ├─ Precision: {precision:.4f}")
    print(f"  ├─ Recall: {recall:.4f}")
    print(f"  ├─ F1-Score: {f1_score:.4f}")
    print(f"  └─ 평균 IoU: {avg_iou:.4f}")
    
    # 결함 유형별 성능
    print(f"\n📊 결함 유형별 성능:")
    for defect_type, stats in sorted(defect_type_stats.items()):
        type_precision = stats['matched_count'] / stats['detected_count'] if stats['detected_count'] > 0 else 0.0
        type_recall = stats['matched_count'] / stats['gt_count'] if stats['gt_count'] > 0 else 0.0
        type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0.0
        avg_score = np.mean(stats['scores']) if stats['scores'] else 0.0
        
        print(f"  ├─ {defect_type}:")
        print(f"  │   ├─ GT: {stats['gt_count']}개")
        print(f"  │   ├─ 검출: {stats['detected_count']}개")
        print(f"  │   ├─ 매칭: {stats['matched_count']}개")
        print(f"  │   ├─ Precision: {type_precision:.4f}")
        print(f"  │   ├─ Recall: {type_recall:.4f}")
        print(f"  │   ├─ F1-Score: {type_f1:.4f}")
        print(f"  │   └─ 평균 점수: {avg_score:.4f}")
    
    # 결과 저장
    results = {
        'total_images': total_images,
        'images_with_gt': images_with_gt,
        'images_with_detections': images_with_detections,
        'total_gt_boxes': total_gt_boxes,
        'total_detected_boxes': total_detected_boxes,
        'matched_pairs': len(matched_pairs),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_iou': avg_iou,
        'defect_type_stats': {
            k: {
                'gt_count': v['gt_count'],
                'detected_count': v['detected_count'],
                'matched_count': v['matched_count'],
                'avg_score': float(np.mean(v['scores'])) if v['scores'] else 0.0
            }
            for k, v in defect_type_stats.items()
        },
        'image_results': image_results[:10]  # 처음 10개만 저장
    }
    
    if output_dir:
        results_path = output_dir / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 결과 저장: {results_path}")
    
    print(f"\n{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='CLIP 모델 결함 검출 성능 평가')
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='데이터 디렉토리 경로 (기본값: data)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('clip_evaluation_results'),
        help='결과 저장 디렉토리 (기본값: clip_evaluation_results)'
    )
    parser.add_argument(
        '--clip-model',
        type=str,
        default='ViT-B/32',
        help='CLIP 모델 이름 (기본값: ViT-B/32)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=None,
        help='평가할 최대 이미지 수 (기본값: 전체)'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU 임계값 (기본값: 0.5)'
    )
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='시각화 비활성화'
    )
    parser.add_argument(
        '--no-save-images',
        action='store_true',
        help='이미지 저장 비활성화'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='사용할 디바이스 (기본값: 자동 감지)'
    )
    
    args = parser.parse_args()
    
    # 디바이스 설정
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"CLIP 모델 결함 검출 평가 도구")
    print(f"{'='*70}")
    print(f"  ├─ 데이터 디렉토리: {args.data_dir}")
    print(f"  ├─ 출력 디렉토리: {args.output_dir}")
    print(f"  ├─ CLIP 모델: {args.clip_model}")
    print(f"  ├─ 최대 이미지 수: {args.max_images if args.max_images else '전체'}")
    print(f"  ├─ IoU 임계값: {args.iou_threshold}")
    print(f"  ├─ 시각화: {'비활성화' if args.no_visualize else '활성화'}")
    print(f"  ├─ 이미지 저장: {'비활성화' if args.no_save_images else '활성화'}")
    print(f"  └─ 디바이스: {args.device}")
    print(f"{'='*70}\n")
    
    # CLIP 모델 초기화
    print("[CLIP 모델 초기화]")
    try:
        device = torch.device(args.device)
        clip_model = CLIPDefectDetector(
            model_name=args.clip_model,
            device=device
        )
        print("[CLIP 모델 초기화] ✅ 완료\n")
    except Exception as e:
        print(f"[CLIP 모델 초기화] ❌ 실패: {e}")
        print("\n💡 CLIP 라이브러리를 설치하세요:")
        print("   pip install git+https://github.com/openai/CLIP.git")
        return
    
    # 평가 실행
    results = evaluate_clip_detector(
        data_dir=args.data_dir,
        clip_model=clip_model,
        output_dir=args.output_dir,
        max_images=args.max_images,
        visualize=not args.no_visualize,
        save_images=not args.no_save_images,
        iou_threshold=args.iou_threshold
    )
    
    print("✅ 평가 완료!")


if __name__ == '__main__':
    main()

