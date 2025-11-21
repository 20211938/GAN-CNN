"""
퓨샷 학습용 데이터셋 유틸리티
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict


class FewShotDataset(Dataset):
    """
    퓨샷 학습용 데이터셋
    N-way K-shot 에피소드 생성
    """
    
    def __init__(
        self,
        samples: List[Dict],
        n_way: int = 5,
        k_shot: int = 3,
        n_query: int = 5,
        episodes: int = 100
    ):
        """
        Args:
            samples: 샘플 리스트 [{'patch': image, 'label': label, ...}, ...]
            n_way: N-way (몇 개의 클래스)
            k_shot: K-shot (각 클래스당 Support 샘플 수)
            n_query: Query 샘플 수 (각 클래스당)
            episodes: 생성할 에피소드 수
        """
        self.samples = samples
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.episodes = episodes
        
        # 클래스별로 샘플 그룹화
        self.class_samples = defaultdict(list)
        for sample in samples:
            label = sample['label']
            self.class_samples[label].append(sample)
        
        # 사용 가능한 클래스 리스트
        self.available_classes = list(self.class_samples.keys())
        
        # 각 클래스당 최소 샘플 수 확인
        min_samples = min(len(samples) for samples in self.class_samples.values())
        if min_samples < k_shot + n_query:
            raise ValueError(
                f"각 클래스당 최소 {k_shot + n_query}개의 샘플이 필요합니다. "
                f"현재 최소 샘플 수: {min_samples}"
            )
    
    def __len__(self) -> int:
        return self.episodes
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        에피소드 생성
        
        Returns:
            {
                'support_images': (n_way * k_shot, C, H, W),
                'support_labels': (n_way * k_shot,),
                'query_images': (n_way * n_query, C, H, W),
                'query_labels': (n_way * n_query,)
            }
        """
        # N개 클래스 랜덤 선택
        selected_classes = np.random.choice(
            self.available_classes,
            size=min(self.n_way, len(self.available_classes)),
            replace=False
        )
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for class_idx, class_label in enumerate(selected_classes):
            class_samples = self.class_samples[class_label]
            
            # 랜덤으로 샘플 선택
            selected_samples = np.random.choice(
                class_samples,
                size=self.k_shot + self.n_query,
                replace=False
            )
            
            # Support와 Query로 분할
            support_samples = selected_samples[:self.k_shot]
            query_samples = selected_samples[self.k_shot:]
            
            # Support set 추가
            for sample in support_samples:
                patch = sample['patch']
                # 이미 텐서인 경우 그대로 사용, 아니면 변환
                if isinstance(patch, torch.Tensor):
                    support_images.append(patch)
                else:
                    # numpy array를 tensor로 변환
                    patch_tensor = torch.from_numpy(patch).float()
                    if patch_tensor.dim() == 2:
                        patch_tensor = patch_tensor.unsqueeze(0)  # Grayscale
                    support_images.append(patch_tensor)
                support_labels.append(class_idx)
            
            # Query set 추가
            for sample in query_samples:
                patch = sample['patch']
                if isinstance(patch, torch.Tensor):
                    query_images.append(patch)
                else:
                    patch_tensor = torch.from_numpy(patch).float()
                    if patch_tensor.dim() == 2:
                        patch_tensor = patch_tensor.unsqueeze(0)
                    query_images.append(patch_tensor)
                query_labels.append(class_idx)
        
        # 리스트를 텐서로 변환
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        
        return {
            'support_images': support_images,
            'support_labels': support_labels,
            'query_images': query_images,
            'query_labels': query_labels
        }


def create_few_shot_loader(
    samples: List[Dict],
    n_way: int = 5,
    k_shot: int = 3,
    n_query: int = 5,
    episodes: int = 100,
    batch_size: int = 1
) -> DataLoader:
    """
    퓨샷 학습용 DataLoader 생성
    
    Args:
        samples: 샘플 리스트
        n_way: N-way
        k_shot: K-shot
        n_query: Query 샘플 수
        episodes: 에피소드 수
        batch_size: 배치 크기 (보통 1)
        
    Returns:
        DataLoader
    """
    dataset = FewShotDataset(
        samples=samples,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query,
        episodes=episodes
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )


def prepare_few_shot_samples(
    defect_dataset_samples: List[Dict],
    defect_type_to_idx: Dict[str, int],
    min_samples_per_class: int = 5
) -> Tuple[List[Dict], Dict[str, int]]:
    """
    일반 데이터셋 샘플을 퓨샷 학습용으로 변환
    
    Args:
        defect_dataset_samples: DefectDataset의 샘플 리스트
        defect_type_to_idx: 결함 유형 인덱스 매핑
        min_samples_per_class: 클래스당 최소 샘플 수
        
    Returns:
        (퓨샷 학습용 샘플 리스트, 클래스 매핑)
    """
    # 클래스별 샘플 수 확인
    class_counts = defaultdict(int)
    for sample in defect_dataset_samples:
        defect_type = sample['defect_type']
        class_counts[defect_type] += 1
    
    # 충분한 샘플이 있는 클래스만 선택
    valid_classes = [
        cls for cls, count in class_counts.items()
        if count >= min_samples_per_class
    ]
    
    # 새로운 클래스 매핑 생성
    few_shot_class_to_idx = {cls: idx for idx, cls in enumerate(valid_classes)}
    
    # 샘플 필터링 및 변환
    few_shot_samples = []
    for sample in defect_dataset_samples:
        defect_type = sample['defect_type']
        if defect_type in valid_classes:
            few_shot_samples.append({
                'patch': sample['image'],  # 이미 전처리된 텐서
                'label': few_shot_class_to_idx[defect_type],
                'defect_type': defect_type,
                'bbox': sample.get('bbox'),
                'image_path': sample.get('image_path')
            })
    
    return few_shot_samples, few_shot_class_to_idx

