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


class DefectDataset(Dataset):
    """
    결함 데이터셋 클래스
    AprilGAN이 찾은 이상 영역과 CNN 레이블을 제공
    """
    
    def __init__(
        self,
        image_paths: List[Path],
        json_paths: List[Path],
        aprilgan_model,
        defect_type_to_idx: Dict[str, int],
        patch_size: Tuple[int, int] = (224, 224),
        transform: Optional[transforms.Compose] = None
    ):
        """
        Args:
            image_paths: 이미지 파일 경로 리스트
            json_paths: JSON 파일 경로 리스트
            aprilgan_model: AprilGAN 모델 (제로샷)
            defect_type_to_idx: 결함 유형을 인덱스로 매핑하는 딕셔너리
            patch_size: CNN 입력 크기
            transform: 이미지 변환 (augmentation)
        """
        self.image_paths = image_paths
        self.json_paths = json_paths
        self.aprilgan_model = aprilgan_model
        self.defect_type_to_idx = defect_type_to_idx
        self.patch_size = patch_size
        self.transform = transform or self._default_transform()
        
        # 데이터 샘플 생성 (AprilGAN으로 이상 영역 찾기)
        self.samples = self._prepare_samples()
    
    def _default_transform(self):
        """기본 이미지 변환"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _prepare_samples(self) -> List[Dict]:
        """
        AprilGAN으로 이상 영역을 찾고 실제 레이블과 매칭하여 샘플 생성
        """
        samples = []
        
        for img_path, json_path in zip(self.image_paths, self.json_paths):
            # 이미지 로드
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # AprilGAN으로 이상 영역 검출
            anomaly_result = self.aprilgan_model.detect(image_rgb)
            anomaly_regions = anomaly_result.get('anomaly_regions', [])
            
            # JSON에서 실제 레이블 추출
            gt_bboxes, gt_types = extract_bboxes_from_json(json_path)
            
            # 이상 영역과 실제 레이블 매칭
            matched_regions = match_anomaly_regions(
                anomaly_regions,
                gt_bboxes,
                gt_types
            )
            
            # 각 매칭된 영역을 샘플로 추가
            for region, defect_type in matched_regions:
                if defect_type is None:
                    # 레이블이 없는 경우 스킵 (또는 'Unknown'으로 처리)
                    continue
                
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
                    'label': defect_type,
                    'bbox': region,
                    'image_path': str(img_path)
                })
        
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
        
        # 레이블 인덱스 변환
        label_idx = self.defect_type_to_idx.get(sample['label'], 0)
        
        return {
            'image': patch_tensor,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'defect_type': sample['label']
        }


def load_defect_data(
    data_dir: Path,
    aprilgan_model,
    train_ratio: float = 0.8,
    batch_size: int = 32,
    patch_size: Tuple[int, int] = (224, 224)
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    결함 데이터를 로드하고 DataLoader 생성
    
    Args:
        data_dir: 데이터 디렉토리 경로
        aprilgan_model: AprilGAN 모델
        train_ratio: 학습 데이터 비율
        batch_size: 배치 크기
        patch_size: CNN 입력 크기
        
    Returns:
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        defect_type_to_idx: 결함 유형 인덱스 매핑
    """
    # 이미지와 JSON 파일 찾기
    image_paths = []
    json_paths = []
    
    for img_path in data_dir.glob("*.jpg"):
        json_path = img_path.with_suffix(".jpg.json")
        if json_path.exists():
            image_paths.append(img_path)
            json_paths.append(json_path)
    
    # 결함 유형 수집
    defect_types = set()
    for json_path in json_paths:
        _, types = extract_bboxes_from_json(json_path)
        defect_types.update(types)
    
    # 'Normal' 추가 (결함이 없는 경우)
    defect_types.add('Normal')
    defect_types = sorted(list(defect_types))
    
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
    
    # 데이터셋 생성
    train_dataset = DefectDataset(
        train_image_paths,
        train_json_paths,
        aprilgan_model,
        defect_type_to_idx,
        patch_size
    )
    
    val_dataset = DefectDataset(
        val_image_paths,
        val_json_paths,
        aprilgan_model,
        defect_type_to_idx,
        patch_size
    )
    
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

