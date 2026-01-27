"""
데이터 로딩 및 전처리 유틸리티
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms

from .bbox_utils import extract_bboxes_from_json, match_anomaly_regions
from .non_iid_distribution import distribute_non_iid, analyze_client_distribution


class DefectDataset(Dataset):
    """
    결함 데이터셋 클래스
    CLIP 모델을 사용하여 결함 영역을 검출하거나 JSON TagBoxes에서 직접 추출하여 CNN 학습 데이터 제공
    """
    
    def __init__(
        self,
        image_paths: List[Path],
        json_paths: List[Path],
        defect_type_to_idx: Dict[str, int],
        patch_size: Tuple[int, int] = (224, 224),
        transform: Optional[transforms.Compose] = None,
        clip_model: Optional[object] = None,
        use_clip: bool = False
    ):
        """
        Args:
            image_paths: 이미지 파일 경로 리스트
            json_paths: JSON 파일 경로 리스트
            defect_type_to_idx: 결함 유형을 인덱스로 매핑하는 딕셔너리
            patch_size: CNN 입력 크기
            transform: 이미지 변환 (augmentation)
            clip_model: CLIP 모델 (use_clip=True일 때 사용)
            use_clip: CLIP 모델 사용 여부 (False면 JSON TagBoxes 직접 사용)
        """
        self.image_paths = image_paths
        self.json_paths = json_paths
        self.defect_type_to_idx = defect_type_to_idx
        self.patch_size = patch_size
        self.transform = transform or self._default_transform()
        self.clip_model = clip_model
        self.use_clip = use_clip
        
        # 데이터 샘플 생성
        if use_clip and clip_model is not None:
            self.samples = self._prepare_samples_with_clip()
        else:
            self.samples = self._prepare_samples()
    
    def _default_transform(self):
        """기본 이미지 변환"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _prepare_samples(self) -> List[Dict]:
        """
        JSON 파일의 TagBoxes에서 직접 결함 영역을 추출하여 샘플 생성
        """
        print(f"\n[데이터 준비] JSON TagBoxes에서 결함 영역 추출 중...")
        print(f"  ├─ 총 이미지 수: {len(self.image_paths)}개")
        
        samples = []
        processed_images = 0
        total_regions = 0
        
        from tqdm import tqdm
        pbar = tqdm(
            zip(self.image_paths, self.json_paths),
            total=len(self.image_paths),
            desc="결함 영역 추출",
            unit="image",
            ncols=100
        )
        
        for img_path, json_path in pbar:
            # 이미지 로드
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # JSON에서 결함 영역과 레이블 직접 추출
            gt_bboxes, gt_types = extract_bboxes_from_json(json_path)
            total_regions += len(gt_bboxes)
            
            # 각 결함 영역을 샘플로 추가
            for bbox, defect_type in zip(gt_bboxes, gt_types):
                # 결함 유형 정규화
                from .bbox_utils import normalize_defect_type
                normalized_defect_type = normalize_defect_type(defect_type)
                
                # 영역 추출
                x1 = max(0, bbox['x1'])
                y1 = max(0, bbox['y1'])
                x2 = min(image_rgb.shape[1], bbox['x2'])
                y2 = min(image_rgb.shape[0], bbox['y2'])
                
                patch = image_rgb[y1:y2, x1:x2]
                
                if patch.size == 0:
                    continue
                
                samples.append({
                    'patch': patch,
                    'label': normalized_defect_type,
                    'bbox': bbox,
                    'image_path': str(img_path)
                })
            
            processed_images += 1
            pbar.set_postfix({
                '처리': f'{processed_images}/{len(self.image_paths)}',
                '샘플': len(samples)
            })
        
        pbar.close()
        
        print(f"\n[데이터 준비] ✅ 완료!")
        print(f"  ├─ 처리된 이미지: {processed_images}개")
        print(f"  ├─ 추출된 결함 영역: {total_regions}개")
        print(f"  └─ 생성된 샘플: {len(samples)}개\n")
        
        return samples
    
    def _prepare_samples_with_clip(self) -> List[Dict]:
        """
        CLIP 모델을 사용하여 결함 영역을 검출하고 실제 레이블과 매칭하여 샘플 생성
        """
        print(f"\n[데이터 준비] CLIP 모델로 결함 영역 검출 중...")
        print(f"  ├─ 총 이미지 수: {len(self.image_paths)}개")
        
        samples = []
        processed_images = 0
        total_regions = 0
        matched_regions = 0
        
        # 결함 유형 리스트 생성
        defect_types = list(self.defect_type_to_idx.keys())
        defect_types = [d for d in defect_types if d != 'Normal']
        
        from tqdm import tqdm
        pbar = tqdm(
            zip(self.image_paths, self.json_paths),
            total=len(self.image_paths),
            desc="CLIP 결함 검출",
            unit="image",
            ncols=100
        )
        
        for img_path, json_path in pbar:
            # 이미지 로드
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # JSON에서 실제 레이블 추출
            gt_bboxes, gt_types = extract_bboxes_from_json(json_path)
            
            # CLIP으로 결함 영역 검출
            clip_result = self.clip_model.detect(
                image_rgb,
                defect_types=defect_types,
                bboxes=gt_bboxes  # 바운딩 박스가 있으면 박스별 검출
            )
            anomaly_regions = clip_result.get('anomaly_regions', [])
            total_regions += len(anomaly_regions)
            
            # 이상 영역과 실제 레이블 매칭
            matched_regions_list = match_anomaly_regions(
                anomaly_regions,
                gt_bboxes,
                gt_types
            )
            
            # 각 매칭된 영역을 샘플로 추가
            for region, defect_type in matched_regions_list:
                if defect_type is None:
                    # 레이블이 없는 경우 스킵
                    continue
                
                # 결함 유형 정규화
                from .bbox_utils import normalize_defect_type
                normalized_defect_type = normalize_defect_type(defect_type)
                
                matched_regions += 1
                
                # 영역 추출
                x1 = max(0, region['x1'])
                y1 = max(0, region['y1'])
                x2 = min(image_rgb.shape[1], region['x2'])
                y2 = min(image_rgb.shape[0], region['y2'])
                
                patch = image_rgb[y1:y2, x1:x2]
                
                if patch.size == 0:
                    continue
                
                samples.append({
                    'patch': patch,
                    'label': normalized_defect_type,
                    'bbox': region,
                    'image_path': str(img_path)
                })
            
            processed_images += 1
            pbar.set_postfix({
                '처리': f'{processed_images}/{len(self.image_paths)}',
                '샘플': len(samples)
            })
        
        pbar.close()
        
        print(f"\n[데이터 준비] ✅ 완료!")
        print(f"  ├─ 처리된 이미지: {processed_images}개")
        print(f"  ├─ CLIP 검출 영역: {total_regions}개")
        print(f"  ├─ 매칭된 영역: {matched_regions}개")
        print(f"  └─ 생성된 샘플: {len(samples)}개\n")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # 패치 리사이즈
        patch = cv2.resize(sample['patch'], self.patch_size)
        patch = patch.astype(np.float32) / 255.0
        
        # PIL Image로 변환 (transforms 호환)
        from PIL import Image
        patch_pil = Image.fromarray((patch * 255).astype(np.uint8))
        
        # 변환 적용
        patch_tensor = self.transform(patch_pil)
        
        # 레이블 정규화 및 인덱스 변환
        from .bbox_utils import normalize_defect_type
        normalized_label = normalize_defect_type(sample['label'])
        label_idx = self.defect_type_to_idx.get(normalized_label, 0)
        
        return {
            'image': patch_tensor,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'defect_type': normalized_label
        }


def load_defect_data(
    data_dir: Path,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    patch_size: Tuple[int, int] = (224, 224),
    clip_model: Optional[object] = None,
    use_clip: bool = False
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    결함 데이터를 로드하고 DataLoader 생성
    
    Args:
        data_dir: 데이터 디렉토리 경로
        train_ratio: 학습 데이터 비율
        batch_size: 배치 크기
        patch_size: CNN 입력 크기
        clip_model: CLIP 모델 (use_clip=True일 때 사용)
        use_clip: CLIP 모델 사용 여부 (False면 JSON TagBoxes 직접 사용)
        
    Returns:
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        defect_type_to_idx: 결함 유형 인덱스 매핑
    """
    print(f"\n{'='*70}")
    print(f"[데이터 로딩] 데이터셋 준비 시작")
    print(f"{'='*70}")
    
    # 이미지와 JSON 파일 찾기
    print(f"[1단계] 이미지 및 JSON 파일 검색 중...")
    image_paths = []
    json_paths = []
    
    for img_path in data_dir.glob("*.jpg"):
        json_path = img_path.with_suffix(".jpg.json")
        if json_path.exists():
            image_paths.append(img_path)
            json_paths.append(json_path)
    
    print(f"  └─ 찾은 이미지-JSON 쌍: {len(image_paths)}개")
    
    # 결함 유형 수집
    print(f"\n[2단계] 결함 유형 수집 중...")
    defect_types = set()
    for json_path in json_paths:
        _, types = extract_bboxes_from_json(json_path)
        defect_types.update(types)
    
    # 'Normal' 추가 (결함이 없는 경우)
    defect_types.add('Normal')
    defect_types = sorted(list(defect_types))
    
    print(f"  └─ 발견된 결함 유형: {len(defect_types)}개")
    for idx, dtype in enumerate(defect_types[:10], 1):  # 상위 10개만 출력
        print(f"      {idx}. {dtype}")
    if len(defect_types) > 10:
        print(f"      ... 외 {len(defect_types) - 10}개")
    
    # 결함 유형을 인덱스로 매핑
    defect_type_to_idx = {dtype: idx for idx, dtype in enumerate(defect_types)}
    idx_to_defect_type = {idx: dtype for dtype, idx in defect_type_to_idx.items()}
    
    # 데이터셋 분할
    n_total = len(image_paths)
    n_train = int(n_total * train_ratio)
    
    train_image_paths = image_paths[:n_train]
    train_json_paths = json_paths[:n_train]
    val_image_paths = image_paths[n_train:]
    val_json_paths = json_paths[n_train:]
    
    print(f"\n[3단계] 데이터셋 분할")
    print(f"  ├─ 전체 이미지: {n_total}개")
    print(f"  ├─ 학습용: {n_train}개 ({train_ratio*100:.1f}%)")
    print(f"  └─ 검증용: {n_total - n_train}개 ({(1-train_ratio)*100:.1f}%)")
    print(f"\n[4단계] 데이터셋 생성 중...")
    print(f"  ├─ 학습 데이터셋 생성 중...")
    
    # 데이터셋 생성
    train_dataset = DefectDataset(
        train_image_paths,
        train_json_paths,
        defect_type_to_idx,
        patch_size,
        clip_model=clip_model,
        use_clip=use_clip
    )
    
    print(f"  └─ 검증 데이터셋 생성 중...")
    
    val_dataset = DefectDataset(
        val_image_paths,
        val_json_paths,
        defect_type_to_idx,
        patch_size,
        clip_model=clip_model,
        use_clip=use_clip
    )
    
    print(f"\n[5단계] DataLoader 생성")
    print(f"  ├─ 학습 샘플 수: {len(train_dataset)}개")
    print(f"  ├─ 검증 샘플 수: {len(val_dataset)}개")
    print(f"  └─ 배치 크기: {batch_size}")
    print(f"{'='*70}\n")
    
    # DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows에서는 0으로 설정
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, defect_type_to_idx

