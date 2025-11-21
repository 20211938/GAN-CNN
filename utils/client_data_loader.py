"""
클라이언트별 Non-IID 데이터 로더
각 클라이언트가 서로 다른 결함 유형 분포를 가지도록 데이터 분배
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import DataLoader

from .data_loader import DefectDataset, load_defect_data
from .non_iid_distribution import distribute_non_iid, analyze_client_distribution


def load_client_data(
    data_dir: Path,
    aprilgan_model,
    num_clients: int = 3,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    patch_size: Tuple[int, int] = (224, 224),
    non_iid_alpha: float = 0.5,
    verbose: bool = True
) -> Tuple[List[DataLoader], List[DataLoader], Dict[str, int]]:
    """
    클라이언트별 Non-IID 데이터를 로드하고 DataLoader 생성
    
    Args:
        data_dir: 데이터 디렉토리 경로
        aprilgan_model: AprilGAN 모델
        num_clients: 클라이언트 수
        train_ratio: 학습 데이터 비율
        batch_size: 배치 크기
        patch_size: CNN 입력 크기
        non_iid_alpha: Non-IID 정도 (0.1: 매우 편향, 1.0: 거의 균등, 10.0: 완전 균등)
        verbose: 상세 출력 여부
        
    Returns:
        train_loaders: 클라이언트별 학습 데이터 로더 리스트
        val_loaders: 클라이언트별 검증 데이터 로더 리스트
        defect_type_to_idx: 결함 유형 인덱스 매핑
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"[클라이언트별 Non-IID 데이터 로딩]")
        print(f"{'='*70}")
        print(f"  ├─ 클라이언트 수: {num_clients}개")
        print(f"  ├─ Non-IID 정도 (alpha): {non_iid_alpha}")
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
    
    # Non-IID 분배
    if verbose:
        print(f"\n[2단계] Non-IID 분배 수행 중...")
    
    client_data = distribute_non_iid(
        image_paths=image_paths,
        json_paths=json_paths,
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
        print(f"[3단계] 클라이언트별 데이터셋 생성 중...")
    
    for client_id, (client_images, client_jsons) in enumerate(client_data):
        if verbose:
            print(f"\n  [클라이언트 {client_id}] 데이터셋 생성 중...")
        
        # 학습/검증 분할
        n_total = len(client_images)
        n_train = int(n_total * train_ratio)
        
        train_images = client_images[:n_train]
        train_jsons = client_jsons[:n_train]
        val_images = client_images[n_train:]
        val_jsons = client_jsons[n_train:]
        
        if verbose:
            print(f"    ├─ 학습용: {n_train}개")
            print(f"    └─ 검증용: {n_total - n_train}개")
        
        # 데이터셋 생성
        train_dataset = DefectDataset(
            train_images,
            train_jsons,
            aprilgan_model,
            defect_type_to_idx,
            patch_size
        )
        
        val_dataset = DefectDataset(
            val_images,
            val_jsons,
            aprilgan_model,
            defect_type_to_idx,
            patch_size
        )
        
        # DataLoader 생성
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        
        if verbose:
            print(f"    └─ 완료! (학습 샘플: {len(train_dataset)}개, 검증 샘플: {len(val_dataset)}개)")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"[클라이언트별 Non-IID 데이터 로딩] ✅ 완료!")
        print(f"{'='*70}\n")
    
    return train_loaders, val_loaders, defect_type_to_idx

