"""
클래스별 성능 평가 메트릭 계산 유틸리티
각 결함 유형별 Precision, Recall, F1-Score 계산
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_fscore_support
)


def calculate_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    클래스별 성능 메트릭 계산
    
    Args:
        y_true: 실제 레이블 (numpy array)
        y_pred: 예측 레이블 (numpy array)
        num_classes: 클래스 수 (None이면 자동 감지)
        class_names: 클래스 이름 리스트 (선택사항)
        
    Returns:
        메트릭 딕셔너리
    """
    if num_classes is None:
        num_classes = max(len(np.unique(y_true)), len(np.unique(y_pred))) + 1
    
    # 전체 Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # 클래스별 메트릭 계산
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # 클래스별 성능 딕셔너리
    per_class_metrics = {}
    for i in range(num_classes):
        class_name = class_names[i] if class_names and i < len(class_names) else f"Class_{i}"
        per_class_metrics[class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(support[i])
        }
    
    metrics = {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'per_class': per_class_metrics,
        'num_classes': num_classes
    }
    
    return metrics


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    모델 평가 및 클래스별 메트릭 계산
    
    Args:
        model: 평가할 모델
        data_loader: 데이터 로더
        device: 디바이스
        num_classes: 클래스 수 (None이면 자동 감지)
        class_names: 클래스 이름 리스트 (선택사항)
        
    Returns:
        평가 결과 딕셔너리
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item() * images.size(0)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 평균 손실 계산
    avg_loss = total_loss / len(all_labels) if len(all_labels) > 0 else 0.0
    
    # 클래스별 메트릭 계산
    metrics = calculate_per_class_metrics(
        all_labels,
        all_preds,
        num_classes=num_classes,
        class_names=class_names
    )
    
    metrics['loss'] = avg_loss
    metrics['total_samples'] = len(all_labels)
    
    return metrics


def print_per_class_metrics(metrics: Dict, title: str = "클래스별 성능 평가"):
    """
    클래스별 성능 메트릭 출력
    
    Args:
        metrics: calculate_per_class_metrics 또는 evaluate_model의 반환값
        title: 출력 제목
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    # 전체 Accuracy
    print(f"\n[전체 성능]")
    print(f"  └─ Accuracy: {metrics['accuracy']:.4f}")
    
    if 'loss' in metrics:
        print(f"\n[손실]")
        print(f"  └─ Average Loss: {metrics['loss']:.6f}")
    
    # 클래스별 성능
    if 'per_class' in metrics and metrics['per_class']:
        print(f"\n[클래스별 성능]")
        print(f"{'클래스명':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 70)
        
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"{class_name:<30} {class_metrics['precision']:>11.4f}  {class_metrics['recall']:>11.4f}  {class_metrics['f1_score']:>11.4f}  {class_metrics['support']:>9}개")
    
    print(f"{'='*70}\n")

