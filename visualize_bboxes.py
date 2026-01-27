"""
JSON 파일의 바운딩 박스를 이미지 위에 그려서 저장하는 스크립트
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
from tqdm import tqdm

from utils.bbox_utils import extract_bboxes_from_json, normalize_defect_type


def draw_bboxes_on_image(
    image: np.ndarray,
    bboxes: List[Dict],
    defect_types: List[str],
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    box_thickness: int = 2,
    font_scale: float = 0.6,
    font_thickness: int = 2
) -> np.ndarray:
    """
    이미지에 바운딩 박스와 라벨을 그리기
    
    Args:
        image: 입력 이미지 (BGR 형식)
        bboxes: 바운딩 박스 리스트
        defect_types: 결함 유형 리스트
        box_color: 박스 색상 (BGR)
        text_color: 텍스트 색상 (BGR)
        box_thickness: 박스 두께
        font_scale: 폰트 크기
        font_thickness: 폰트 두께
        
    Returns:
        그려진 이미지
    """
    result_image = image.copy()
    h, w = image.shape[:2]
    
    # 색상 팔레트 (다양한 색상 사용)
    colors = [
        (0, 255, 0),    # 녹색
        (255, 0, 0),    # 파란색
        (0, 0, 255),    # 빨간색
        (255, 255, 0),  # 청록색
        (255, 0, 255),  # 자홍색
        (0, 255, 255),  # 노란색
        (128, 0, 128),  # 보라색
        (255, 165, 0),  # 주황색
    ]
    
    for idx, (bbox, defect_type) in enumerate(zip(bboxes, defect_types)):
        x1 = max(0, int(bbox['x1']))
        y1 = max(0, int(bbox['y1']))
        x2 = min(w, int(bbox['x2']))
        y2 = min(h, int(bbox['y2']))
        
        # 박스 색상 선택 (결함 유형별로 다른 색상)
        color = colors[idx % len(colors)]
        
        # 바운딩 박스 그리기
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, box_thickness)
        
        # 결함 유형 라벨 텍스트
        label = normalize_defect_type(defect_type)
        
        # 텍스트 배경 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )
        
        # 텍스트 배경 그리기
        text_bg_y1 = max(0, y1 - text_height - baseline - 5)
        text_bg_y2 = y1
        text_bg_x1 = x1
        text_bg_x2 = min(w, x1 + text_width + 10)
        
        cv2.rectangle(
            result_image,
            (text_bg_x1, text_bg_y1),
            (text_bg_x2, text_bg_y2),
            color,
            -1  # 채워진 사각형
        )
        
        # 텍스트 그리기
        cv2.putText(
            result_image,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )
    
    return result_image


def visualize_dataset_bboxes(
    data_dir: Path,
    output_dir: Path,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png'],
    overwrite: bool = False
):
    """
    데이터셋의 모든 이미지에 바운딩 박스를 그려서 저장
    
    Args:
        data_dir: 데이터 디렉토리
        output_dir: 출력 디렉토리
        image_extensions: 이미지 확장자 리스트
        overwrite: 기존 파일 덮어쓰기 여부
    """
    print(f"\n{'='*70}")
    print(f"바운딩 박스 시각화")
    print(f"{'='*70}")
    print(f"  ├─ 입력 디렉토리: {data_dir}")
    print(f"  ├─ 출력 디렉토리: {output_dir}")
    print(f"  └─ 이미지 확장자: {', '.join(image_extensions)}")
    print(f"{'='*70}\n")
    
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 이미지 파일 찾기
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_dir.glob(f"*{ext}"))
        image_files.extend(data_dir.glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        print(f"⚠️  이미지 파일을 찾을 수 없습니다: {data_dir}")
        return
    
    print(f"[1단계] 이미지 파일 검색 완료: {len(image_files)}개\n")
    
    # 통계 변수
    processed_count = 0
    skipped_count = 0
    error_count = 0
    total_bboxes = 0
    
    # 이미지 처리
    pbar = tqdm(image_files, desc="바운딩 박스 그리기", unit="image", ncols=100)
    
    for img_path in pbar:
        # JSON 파일 찾기
        json_path = img_path.with_suffix(".jpg.json")
        if not json_path.exists():
            # 다른 확장자도 시도
            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                skipped_count += 1
                continue
        
        # 출력 파일 경로
        output_path = output_dir / f"{img_path.stem}_bboxes{img_path.suffix}"
        
        # 이미 존재하는 경우
        if output_path.exists() and not overwrite:
            skipped_count += 1
            continue
        
        try:
            # 이미지 로드
            image = cv2.imread(str(img_path))
            if image is None:
                error_count += 1
                continue
            
            # JSON에서 바운딩 박스 추출
            bboxes, defect_types = extract_bboxes_from_json(json_path)
            
            if len(bboxes) == 0:
                # 바운딩 박스가 없는 경우 원본 이미지만 저장 (선택사항)
                skipped_count += 1
                continue
            
            total_bboxes += len(bboxes)
            
            # 바운딩 박스 그리기
            result_image = draw_bboxes_on_image(image, bboxes, defect_types)
            
            # 결과 저장
            cv2.imwrite(str(output_path), result_image)
            processed_count += 1
            
            pbar.set_postfix({
                '처리': processed_count,
                '바운딩박스': total_bboxes,
                '스킵': skipped_count
            })
            
        except Exception as e:
            error_count += 1
            print(f"\n⚠️  오류 발생 ({img_path.name}): {e}")
            continue
    
    pbar.close()
    
    # 결과 요약
    print(f"\n{'='*70}")
    print(f"처리 완료")
    print(f"{'='*70}")
    print(f"  ├─ 처리된 이미지: {processed_count}개")
    print(f"  ├─ 스킵된 이미지: {skipped_count}개")
    print(f"  ├─ 오류 발생: {error_count}개")
    print(f"  ├─ 총 바운딩 박스: {total_bboxes}개")
    print(f"  └─ 출력 디렉토리: {output_dir}")
    print(f"{'='*70}\n")


def visualize_single_image(
    image_path: Path,
    json_path: Path,
    output_path: Path,
    box_color: Tuple[int, int, int] = (0, 255, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255)
):
    """
    단일 이미지에 바운딩 박스 그리기
    
    Args:
        image_path: 이미지 파일 경로
        json_path: JSON 파일 경로
        output_path: 출력 파일 경로
        box_color: 박스 색상 (BGR)
        text_color: 텍스트 색상 (BGR)
    """
    # 이미지 로드
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
    
    # JSON에서 바운딩 박스 추출
    bboxes, defect_types = extract_bboxes_from_json(json_path)
    
    if len(bboxes) == 0:
        print(f"⚠️  바운딩 박스가 없습니다: {json_path}")
        return
    
    # 바운딩 박스 그리기
    result_image = draw_bboxes_on_image(image, bboxes, defect_types, box_color, text_color)
    
    # 결과 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_image)
    
    print(f"✅ 저장 완료: {output_path}")
    print(f"   바운딩 박스 수: {len(bboxes)}개")


def main():
    parser = argparse.ArgumentParser(
        description='JSON 파일의 바운딩 박스를 이미지 위에 그려서 저장'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='데이터 디렉토리 경로 (기본값: data)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('visualized_bboxes'),
        help='출력 디렉토리 경로 (기본값: visualized_bboxes)'
    )
    parser.add_argument(
        '--image',
        type=Path,
        default=None,
        help='단일 이미지 파일 경로 (지정 시 해당 이미지만 처리)'
    )
    parser.add_argument(
        '--json',
        type=Path,
        default=None,
        help='단일 JSON 파일 경로 (--image와 함께 사용)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='단일 이미지 출력 경로 (--image와 함께 사용)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='기존 파일 덮어쓰기'
    )
    
    args = parser.parse_args()
    
    # 단일 이미지 처리 모드
    if args.image is not None:
        if args.json is None:
            args.json = args.image.with_suffix(".jpg.json")
            if not args.json.exists():
                args.json = args.image.with_suffix(".json")
        
        if not args.json.exists():
            print(f"❌ JSON 파일을 찾을 수 없습니다: {args.json}")
            return
        
        if args.output is None:
            args.output = args.image.parent / f"{args.image.stem}_bboxes{args.image.suffix}"
        
        try:
            visualize_single_image(args.image, args.json, args.output)
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
        return
    
    # 전체 데이터셋 처리 모드
    if not args.data_dir.exists():
        print(f"❌ 데이터 디렉토리가 없습니다: {args.data_dir}")
        return
    
    visualize_dataset_bboxes(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite
    )


if __name__ == '__main__':
    main()

