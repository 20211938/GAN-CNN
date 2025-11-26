"""
클라이언트별 Non-IID 데이터 로더
각 클라이언트가 서로 다른 결함 유형 분포를 가지도록 데이터 분배
"""

import os
import multiprocessing
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader

from .data_loader import DefectDataset, load_defect_data
from .non_iid_distribution import distribute_non_iid, analyze_client_distribution


def get_optimal_num_workers() -> int:
    """
    최적의 num_workers 수 계산
    
    Returns:
        num_workers 수 (Windows에서는 0, Linux/Mac에서는 CPU 코어 수 기반)
    """
    # Windows에서는 multiprocessing이 spawn 방식을 사용하므로 0으로 설정
    if os.name == 'nt':  # Windows
        return 0
    
    # Linux/Mac에서는 CPU 코어 수의 절반 사용 (너무 많으면 오버헤드 발생)
    cpu_count = multiprocessing.cpu_count()
    return max(0, min(4, cpu_count // 2))  # 최대 4개로 제한


def get_optimal_batch_size(
    base_batch_size: int = 32,
    available_memory_gb: Optional[float] = None
) -> int:
    """
    GPU 메모리에 따라 최적의 배치 크기 계산
    
    Args:
        base_batch_size: 기본 배치 크기
        available_memory_gb: 사용 가능한 GPU 메모리 (GB, None이면 자동 감지)
        
    Returns:
        최적의 배치 크기
    """
    import torch
    
    if not torch.cuda.is_available():
        return base_batch_size
    
    try:
        if available_memory_gb is None:
            # GPU 메모리 자동 감지
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            gpu_memory_gb = available_memory_gb
        
        # GPU 메모리에 따라 배치 크기 조정
        # 8GB 이상: 기본값 유지 또는 증가
        # 4-8GB: 기본값 유지
        # 4GB 미만: 배치 크기 감소
        
        if gpu_memory_gb >= 8:
            optimal_batch_size = base_batch_size * 2  # 메모리 여유 있으면 증가
        elif gpu_memory_gb >= 4:
            optimal_batch_size = base_batch_size
        else:
            optimal_batch_size = max(8, base_batch_size // 2)  # 최소 8개는 보장
        
        return optimal_batch_size
    except Exception:
        # 오류 발생 시 기본값 반환
        return base_batch_size


def load_client_data(
    data_dir: Path,
    num_clients: int = 3,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    batch_size: int = 32,
    patch_size: Tuple[int, int] = (224, 224),
    non_iid_alpha: float = 0.5,
    num_workers: Optional[int] = None,
    auto_batch_size: bool = True,
    verbose: bool = True,
    clip_model: Optional[object] = None,
    use_clip: bool = False
) -> Tuple[List[DataLoader], List[DataLoader], Optional[DataLoader], Dict[str, int]]:
    """
    클라이언트별 Non-IID 데이터를 로드하고 DataLoader 생성
    
    Args:
        data_dir: 데이터 디렉토리 경로
        num_clients: 클라이언트 수
        clip_model: CLIP 모델 (use_clip=True일 때 사용)
        use_clip: CLIP 모델 사용 여부 (False면 JSON TagBoxes 직접 사용)
        train_ratio: 학습 데이터 비율 (train 데이터 중에서, 기본값: 0.8)
        val_ratio: 검증 데이터 비율 (train 데이터 중에서, 기본값: 0.2)
        test_ratio: 테스트 데이터 비율 (전체 데이터 중에서, 기본값: 0.1)
        batch_size: 배치 크기
        patch_size: CNN 입력 크기
        non_iid_alpha: Non-IID 정도 (0.1: 매우 편향, 1.0: 거의 균등, 10.0: 완전 균등)
        num_workers: 데이터 로딩 워커 수 (None이면 자동 설정)
        auto_batch_size: 배치 크기 자동 조정 여부
        verbose: 상세 출력 여부
        
    Returns:
        train_loaders: 클라이언트별 학습 데이터 로더 리스트
        val_loaders: 클라이언트별 검증 데이터 로더 리스트
        test_loader: 전체 테스트 데이터 로더 (None이면 생성 안 함)
        defect_type_to_idx: 결함 유형 인덱스 매핑
    """
    # num_workers 자동 설정
    if num_workers is None:
        num_workers = get_optimal_num_workers()
    
    # 배치 크기 자동 조정
    if auto_batch_size:
        optimal_batch_size = get_optimal_batch_size(batch_size)
        if optimal_batch_size != batch_size:
            if verbose:
                print(f"[최적화] 배치 크기 자동 조정: {batch_size} → {optimal_batch_size}")
            batch_size = optimal_batch_size
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"[클라이언트별 Non-IID 데이터 로딩]")
        print(f"{'='*70}")
        print(f"  ├─ 클라이언트 수: {num_clients}개")
        print(f"  ├─ Non-IID 정도 (alpha): {non_iid_alpha}")
        print(f"  ├─ 배치 크기: {batch_size}")
        print(f"  ├─ 데이터 로딩 워커 수: {num_workers}개")
        print(f"  └─ alpha 설명: 작을수록 편향됨 (0.1=매우편향, 1.0=보통, 10.0=균등)")
    
    # 이미지와 JSON 파일 찾기
    image_paths = []
    json_paths = []
    
    for img_path in data_dir.glob("*.jpg"):
        json_path = img_path.with_suffix(".jpg.json")
        if json_path.exists():
            image_paths.append(img_path)
            json_paths.append(json_path)
    
    if len(image_paths) == 0:
        raise ValueError(f"데이터 디렉토리에 이미지 파일이 없습니다: {data_dir}")
    
    # 결함 유형 수집 (정규화 적용)
    from .bbox_utils import extract_bboxes_from_json, normalize_defect_type
    defect_types = set()
    for json_path in json_paths:
        _, types = extract_bboxes_from_json(json_path)
        # 이미 extract_bboxes_from_json에서 정규화되지만, 중복 제거를 위해 한 번 더 정규화
        normalized_types = [normalize_defect_type(t) for t in types]
        defect_types.update(normalized_types)
    
    defect_types.add('Normal')
    defect_types = sorted(list(defect_types))
    
    # 결함 유형을 인덱스로 매핑
    defect_type_to_idx = {dtype: idx for idx, dtype in enumerate(defect_types)}
    
    if verbose:
        print(f"\n[1단계] 전체 데이터 분석")
        print(f"  └─ 발견된 결함 유형: {len(defect_types)}개")
        print(f"  └─ 전체 데이터 수: {len(image_paths)}개")
    
    # 먼저 전체 데이터를 결함 유형별로 그룹화
    if verbose:
        print(f"\n[2단계] 결함 유형별 데이터 그룹화 및 train/test 분할")
    
    import random
    from collections import defaultdict
    # 재현성을 위한 시드 설정 (선택사항)
    # random.seed(42)
    
    # 결함 유형별로 데이터 그룹화
    defect_type_groups = defaultdict(list)
    for img_path, json_path in zip(image_paths, json_paths):
        _, types = extract_bboxes_from_json(json_path)
        normalized_types = [normalize_defect_type(t) for t in types]
        # 각 이미지가 가진 결함 유형 중 첫 번째를 기준으로 그룹화
        # (여러 결함이 있으면 첫 번째로 분류)
        if normalized_types:
            primary_type = normalized_types[0]
        else:
            primary_type = 'Normal'
        defect_type_groups[primary_type].append((img_path, json_path))
    
    # 각 결함 유형별로 셔플
    for defect_type in defect_type_groups:
        random.shuffle(defect_type_groups[defect_type])
    
    # 각 결함 유형에서 test_ratio만큼 테스트 데이터로 추출
    test_images_all = []
    test_jsons_all = []
    train_images_all = []
    train_jsons_all = []
    test_defect_types = set()
    
    for defect_type, samples in defect_type_groups.items():
        n_samples = len(samples)
        n_test = max(1, int(n_samples * test_ratio))  # 최소 1개는 보장
        
        # 테스트 데이터 추출
        test_samples = samples[:n_test]
        test_images_all.extend([s[0] for s in test_samples])
        test_jsons_all.extend([s[1] for s in test_samples])
        test_defect_types.add(defect_type)  # 테스트에 포함된 결함 유형 기록
        
        # 학습 데이터 추출
        train_samples = samples[n_test:]
        train_images_all.extend([s[0] for s in train_samples])
        train_jsons_all.extend([s[1] for s in train_samples])
    
    # Train 데이터 셔플 (Non-IID 분배 전에)
    combined_train = list(zip(train_images_all, train_jsons_all))
    random.shuffle(combined_train)
    train_images_all, train_jsons_all = zip(*combined_train)
    train_images_all = list(train_images_all)
    train_jsons_all = list(train_jsons_all)
    
    n_total = len(image_paths)
    if verbose:
        print(f"  ├─ Train 데이터: {len(train_images_all)}개 ({len(train_images_all)/n_total*100:.1f}%)")
        print(f"  ├─ Test 데이터: {len(test_images_all)}개 ({len(test_images_all)/n_total*100:.1f}%)")
        print(f"  └─ Test 데이터에 포함된 결함 유형: {len(test_defect_types)}개 (모든 결함 유형 포함)")
    
    # Train 데이터만 클라이언트별 Non-IID 분배
    if verbose:
        print(f"\n[3단계] Train 데이터 클라이언트별 Non-IID 분배 수행 중...")
    
    client_data = distribute_non_iid(
        image_paths=train_images_all,
        json_paths=train_jsons_all,
        num_clients=num_clients,
        alpha=non_iid_alpha
    )
    
    # 클라이언트별 분포 분석
    if verbose:
        analyze_client_distribution(client_data, defect_type_to_idx)
    
    # 각 클라이언트별로 학습/검증 데이터셋 생성
    train_loaders = []
    val_loaders = []
    
    if verbose:
        print(f"\n[4단계] 클라이언트별 데이터셋 생성 중...")
    
    for client_id, (client_images, client_jsons) in enumerate(client_data):
        if verbose:
            print(f"\n  [클라이언트 {client_id}] 데이터셋 생성 중...")
        
        # 학습/검증 분할 (train 데이터 중에서)
        n_total_client = len(client_images)
        n_train = int(n_total_client * train_ratio)
        
        train_images = client_images[:n_train]
        train_jsons = client_jsons[:n_train]
        val_images = client_images[n_train:]
        val_jsons = client_jsons[n_train:]
        
        if verbose:
            print(f"    ├─ 학습용: {n_train}개 ({n_train/n_total_client*100:.1f}%)")
            print(f"    └─ 검증용: {n_total_client - n_train}개 ({(n_total_client-n_train)/n_total_client*100:.1f}%)")
        
        # 데이터셋 생성
        train_dataset = DefectDataset(
            train_images,
            train_jsons,
            defect_type_to_idx,
            patch_size,
            clip_model=clip_model,
            use_clip=use_clip
        )
        
        val_dataset = DefectDataset(
            val_images,
            val_jsons,
            defect_type_to_idx,
            patch_size,
            clip_model=clip_model,
            use_clip=use_clip
        )
        
        # DataLoader 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False  # GPU 사용 시 메모리 고정
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False
        )
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        
        if verbose:
            print(f"    └─ 완료! (학습 샘플: {len(train_dataset)}개, 검증 샘플: {len(val_dataset)}개)")
    
    # 전체 테스트 데이터셋 생성
    test_loader = None
    if len(test_images_all) > 0:
        if verbose:
            print(f"\n[5단계] 전체 테스트 데이터셋 생성 중...")
        
        test_dataset = DefectDataset(
            test_images_all,
            test_jsons_all,
            defect_type_to_idx,
            patch_size,
            clip_model=clip_model,
            use_clip=use_clip
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True if num_workers > 0 else False
        )
        
        if verbose:
            print(f"  └─ 테스트 샘플: {len(test_dataset)}개")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"[클라이언트별 Non-IID 데이터 로딩] ✅ 완료!")
        print(f"{'='*70}\n")
    
    return train_loaders, val_loaders, test_loader, defect_type_to_idx

