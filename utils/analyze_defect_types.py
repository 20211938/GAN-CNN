"""
데이터셋의 결함 종류 식별 및 분석 스크립트
"""

import os
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set
import sys


def extract_defect_types_from_metadata(metadata: dict) -> List[str]:
    """JSON 메타데이터에서 결함 유형 추출"""
    defect_types = []
    
    # TagBoxes에서 결함 정보 추출
    if 'DepositionImageModel' in metadata:
        tag_boxes = metadata['DepositionImageModel'].get('TagBoxes', [])
        for tag in tag_boxes:
            name = tag.get('Name', '').strip()
            comment = tag.get('Comment', '').strip()
            if name:
                defect_type = comment if comment else name
                if defect_type and defect_type not in defect_types:
                    defect_types.append(defect_type)
    
    if 'ScanningImageModel' in metadata:
        tag_boxes = metadata['ScanningImageModel'].get('TagBoxes', [])
        for tag in tag_boxes:
            name = tag.get('Name', '').strip()
            comment = tag.get('Comment', '').strip()
            if name:
                defect_type = comment if comment else name
                if defect_type and defect_type not in defect_types:
                    defect_types.append(defect_type)
    
    return defect_types if defect_types else ["Normal"]


def analyze_defect_dataset(data_dir: str = "data/labeled_layers"):
    """데이터셋의 결함 종류 분석"""
    print("=" * 60)
    print("데이터셋 결함 종류 식별 및 분석")
    print("=" * 60)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"\n[오류] 데이터 디렉토리가 없습니다: {data_dir}")
        return
    
    print(f"\n[데이터 분석] 디렉토리: {data_dir}")
    
    # 통계 정보
    defect_type_counts = Counter()
    defect_type_samples = defaultdict(list)  # 각 결함 유형별 샘플 경로
    total_images = 0
    images_with_defects = 0
    images_without_tags = 0
    
    # 디렉토리별 통계
    dir_stats = defaultdict(lambda: {'total': 0, 'defect_types': Counter()})
    
    db_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"  - 디렉토리 수: {len(db_dirs)}개\n")
    
    for db_dir in db_dirs:
        dir_name = db_dir.name
        dir_total = 0
        
        for img_file in db_dir.glob("*.jpg"):
            json_file = img_file.with_suffix(".jpg.json")
            if not json_file.exists():
                continue
            
            total_images += 1
            dir_total += 1
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 결함 유형 추출
                defect_types = extract_defect_types_from_metadata(metadata)
                
                # TagBoxes가 없는지 확인
                has_deposition_tags = bool(metadata.get('DepositionImageModel', {}).get('TagBoxes', []))
                has_scanning_tags = bool(metadata.get('ScanningImageModel', {}).get('TagBoxes', []))
                has_no_tags = not has_deposition_tags and not has_scanning_tags
                
                if has_no_tags:
                    images_without_tags += 1
                
                if len(defect_types) > 0 and defect_types != ["Normal"]:
                    images_with_defects += 1
                
                # 각 결함 유형 카운트 및 샘플 저장
                for defect_type in defect_types:
                    defect_type_counts[defect_type] += 1
                    defect_type_samples[defect_type].append(str(img_file))
                    dir_stats[dir_name]['defect_types'][defect_type] += 1
                
                dir_stats[dir_name]['total'] = dir_total
                    
            except Exception as e:
                continue
        
        if dir_total > 0:
            print(f"  - {dir_name}: {dir_total}개 이미지")
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("[전체 통계]")
    print("=" * 60)
    print(f"  - 전체 이미지 수: {total_images}")
    print(f"  - 결함이 있는 이미지: {images_with_defects}개 ({images_with_defects/total_images*100:.2f}%)")
    print(f"  - 태그가 없는 이미지: {images_without_tags}개 ({images_without_tags/total_images*100:.2f}%)")
    
    # 결함 유형별 통계
    print("\n" + "=" * 60)
    print("[결함 유형별 통계]")
    print("=" * 60)
    
    sorted_types = sorted(defect_type_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n  총 {len(sorted_types)}개 결함 유형 발견\n")
    print(f"{'순위':<6} {'결함 유형':<40} {'샘플 수':<12} {'비율':<10}")
    print("-" * 70)
    
    for idx, (defect_type, count) in enumerate(sorted_types, 1):
        percentage = count / total_images * 100 if total_images > 0 else 0
        print(f"{idx:<6} {defect_type:<40} {count:<12} {percentage:>6.2f}%")
    
    # 샘플 수별 분류
    print("\n" + "=" * 60)
    print("[샘플 수별 분류]")
    print("=" * 60)
    
    categories = {
        '매우 많음 (100개 이상)': [],
        '많음 (50-99개)': [],
        '보통 (30-49개)': [],
        '적음 (10-29개)': [],
        '매우 적음 (1-9개)': []
    }
    
    for defect_type, count in sorted_types:
        if count >= 100:
            categories['매우 많음 (100개 이상)'].append((defect_type, count))
        elif count >= 50:
            categories['많음 (50-99개)'].append((defect_type, count))
        elif count >= 30:
            categories['보통 (30-49개)'].append((defect_type, count))
        elif count >= 10:
            categories['적음 (10-29개)'].append((defect_type, count))
        else:
            categories['매우 적음 (1-9개)'].append((defect_type, count))
    
    for category, items in categories.items():
        if items:
            print(f"\n  [{category}] {len(items)}개")
            for defect_type, count in sorted(items, key=lambda x: x[1], reverse=True):
                print(f"    - {defect_type}: {count}개")
    
    # 디렉토리별 결함 유형 분포
    print("\n" + "=" * 60)
    print("[디렉토리별 결함 유형 분포]")
    print("=" * 60)
    
    for dir_name in sorted(dir_stats.keys()):
        stats = dir_stats[dir_name]
        print(f"\n  {dir_name} ({stats['total']}개 이미지)")
        top_types = stats['defect_types'].most_common(5)
        for defect_type, count in top_types:
            print(f"    - {defect_type}: {count}개")
    
    # 각 결함 유형별 샘플 경로 (상위 3개)
    print("\n" + "=" * 60)
    print("[결함 유형별 샘플 이미지 경로] (상위 10개 유형, 각 3개씩)")
    print("=" * 60)
    
    for defect_type, count in sorted_types[:10]:
        samples = defect_type_samples[defect_type][:3]
        print(f"\n  {defect_type} ({count}개)")
        for idx, sample_path in enumerate(samples, 1):
            print(f"    {idx}. {Path(sample_path).name}")
    
    # 클래스 불균형 분석
    print("\n" + "=" * 60)
    print("[클래스 불균형 분석]")
    print("=" * 60)
    
    if len(sorted_types) > 0:
        max_count = sorted_types[0][1]
        min_count = sorted_types[-1][1]
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"  - 최대 샘플 수: {max_count}개 ({sorted_types[0][0]})")
        print(f"  - 최소 샘플 수: {min_count}개 ({sorted_types[-1][0]})")
        print(f"  - 불균형 비율: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            print(f"  - 경고: 클래스 불균형이 심합니다 (10:1 이상)")
        elif imbalance_ratio > 5:
            print(f"  - 주의: 클래스 불균형이 있습니다 (5:1 이상)")
        else:
            print(f"  - 양호: 클래스 불균형이 적습니다")
    
    # 학습 권장 사항
    print("\n" + "=" * 60)
    print("[학습 권장 사항]")
    print("=" * 60)
    
    # 30개 미만인 클래스
    minor_classes = [(dt, count) for dt, count in sorted_types if count < 30 and dt != "Normal"]
    if minor_classes:
        print(f"\n  [소수 클래스 제거 권장] (30개 미만)")
        for defect_type, count in minor_classes:
            print(f"    - {defect_type}: {count}개")
        print(f"    총 {len(minor_classes)}개 클래스, {sum(c for _, c in minor_classes)}개 샘플")
    
    # 30개 이상인 클래스
    major_classes = [(dt, count) for dt, count in sorted_types if count >= 30 or dt == "Normal"]
    if major_classes:
        print(f"\n  [학습 가능한 클래스] (30개 이상)")
        for defect_type, count in sorted(major_classes, key=lambda x: x[1], reverse=True):
            print(f"    - {defect_type}: {count}개")
        print(f"    총 {len(major_classes)}개 클래스, {sum(c for _, c in major_classes)}개 샘플")
    
    print("\n" + "=" * 60)
    print("분석 완료!")
    print("=" * 60)
    
    return {
        'defect_type_counts': dict(defect_type_counts),
        'defect_type_samples': dict(defect_type_samples),
        'total_images': total_images,
        'images_with_defects': images_with_defects,
        'dir_stats': dict(dir_stats)
    }


def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='데이터셋의 결함 종류 식별 및 분석')
    parser.add_argument('--data-dir', type=str, default='data/labeled_layers',
                       help='데이터 디렉토리 (기본값: data/labeled_layers)')
    
    args = parser.parse_args()
    
    analyze_defect_dataset(data_dir=args.data_dir)


if __name__ == "__main__":
    main()

