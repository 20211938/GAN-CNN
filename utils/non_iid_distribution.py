"""
Non-IID 데이터 분배 유틸리티
각 클라이언트가 서로 다른 결함 유형 분포를 가지도록 데이터 분배
"""

import random
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from pathlib import Path


def distribute_non_iid(
    image_paths: List[Path],
    json_paths: List[Path],
    num_clients: int,
    alpha: float = 0.5,
    min_samples_per_client: int = 10
) -> List[Tuple[List[Path], List[Path]]]:
    """
    Non-IID 분포로 데이터를 클라이언트별로 분배
    
    Args:
        image_paths: 이미지 파일 경로 리스트
        json_paths: JSON 파일 경로 리스트
        num_clients: 클라이언트 수
        alpha: Non-IID 정도 (0.1: 매우 편향, 1.0: 거의 균등, 10.0: 완전 균등)
        min_samples_per_client: 클라이언트당 최소 샘플 수
        
    Returns:
        클라이언트별 (이미지 경로 리스트, JSON 경로 리스트) 튜플 리스트
    """
    from .bbox_utils import extract_bboxes_from_json
    
    # 결함 유형별로 데이터 그룹화
    defect_type_to_files = defaultdict(list)
    
    print(f"\n[Non-IID 분배] 데이터 분석 중...")
    for img_path, json_path in zip(image_paths, json_paths):
        try:
            _, defect_types = extract_bboxes_from_json(json_path)
            # 각 이미지의 주요 결함 유형 결정 (첫 번째 결함 유형 또는 'Normal')
            main_defect_type = defect_types[0] if defect_types else 'Normal'
            defect_type_to_files[main_defect_type].append((img_path, json_path))
        except Exception:
            # 오류 발생 시 Normal로 분류
            defect_type_to_files['Normal'].append((img_path, json_path))
    
    defect_types = list(defect_type_to_files.keys())
    print(f"  └─ 발견된 결함 유형: {len(defect_types)}개")
    
    # 각 결함 유형별 샘플 수 출력
    for dtype in sorted(defect_types):
        count = len(defect_type_to_files[dtype])
        print(f"      - {dtype}: {count}개")
    
    # Dirichlet 분포를 사용하여 Non-IID 분배
    # alpha가 작을수록 더 편향된 분배
    print(f"\n[Non-IID 분배] Dirichlet 분포로 클라이언트별 분배 (alpha={alpha})...")
    
    # 각 결함 유형에 대해 클라이언트별 비율 생성
    client_distributions = {}
    for defect_type in defect_types:
        # Dirichlet 분포로 각 클라이언트의 비율 생성
        proportions = np.random.dirichlet([alpha] * num_clients)
        client_distributions[defect_type] = proportions
    
    # 클라이언트별 데이터 분배
    client_data = [[] for _ in range(num_clients)]
    
    for defect_type, files in defect_type_to_files.items():
        proportions = client_distributions[defect_type]
        n_samples = len(files)
        
        # 각 클라이언트에게 할당할 샘플 수 계산
        client_counts = []
        remaining = n_samples
        
        for i in range(num_clients - 1):
            count = max(0, int(n_samples * proportions[i]))
            count = min(count, remaining)
            client_counts.append(count)
            remaining -= count
        
        # 마지막 클라이언트는 남은 모든 샘플
        client_counts.append(remaining)
        
        # 샘플 셔플
        shuffled_files = files.copy()
        random.shuffle(shuffled_files)
        
        # 각 클라이언트에 샘플 할당
        idx = 0
        for client_id in range(num_clients):
            count = client_counts[client_id]
            client_data[client_id].extend(shuffled_files[idx:idx+count])
            idx += count
    
    # 결과 정리
    result = []
    for client_id in range(num_clients):
        client_images = [img for img, _ in client_data[client_id]]
        client_jsons = [json for _, json in client_data[client_id]]
        
        # 최소 샘플 수 확인
        if len(client_images) < min_samples_per_client:
            print(f"  ⚠️  클라이언트 {client_id}: 샘플 수 부족 ({len(client_images)} < {min_samples_per_client})")
        
        result.append((client_images, client_jsons))
    
    # 분배 통계 출력
    print(f"\n[Non-IID 분배] 분배 완료!")
    print(f"  └─ 클라이언트별 샘플 수:")
    for client_id, (images, _) in enumerate(result):
        print(f"      클라이언트 {client_id}: {len(images)}개")
    
    return result


def analyze_client_distribution(
    client_data: List[Tuple[List[Path], List[Path]]],
    defect_type_to_idx: Dict[str, int]
) -> None:
    """
    클라이언트별 데이터 분포 분석 및 출력
    
    Args:
        client_data: 클라이언트별 데이터 리스트
        defect_type_to_idx: 결함 유형 인덱스 매핑
    """
    from .bbox_utils import extract_bboxes_from_json
    
    print(f"\n{'='*70}")
    print(f"[Non-IID 분석] 클라이언트별 데이터 분포")
    print(f"{'='*70}")
    
    idx_to_defect_type = {idx: dtype for dtype, idx in defect_type_to_idx.items()}
    
    for client_id, (image_paths, json_paths) in enumerate(client_data):
        # 클라이언트별 결함 유형 통계
        defect_counts = defaultdict(int)
        
        for json_path in json_paths:
            try:
                _, defect_types = extract_bboxes_from_json(json_path)
                for dtype in defect_types:
                    defect_counts[dtype] += 1
            except Exception:
                defect_counts['Normal'] += 1
        
        total = sum(defect_counts.values())
        
        print(f"\n[클라이언트 {client_id}]")
        print(f"  ├─ 총 샘플 수: {total}개")
        print(f"  └─ 결함 유형별 분포:")
        
        # 비율 순으로 정렬
        sorted_defects = sorted(defect_counts.items(), key=lambda x: x[1], reverse=True)
        
        for dtype, count in sorted_defects[:10]:  # 상위 10개만 출력
            percentage = count / total * 100 if total > 0 else 0
            bar = '█' * int(percentage / 2)  # 간단한 바 차트
            print(f"      {dtype:<30} {count:>4}개 ({percentage:>5.1f}%) {bar}")
        
        if len(sorted_defects) > 10:
            print(f"      ... 외 {len(sorted_defects) - 10}개 유형")
    
    print(f"{'='*70}\n")

