"""
연합학습 시각화 도구
학습 곡선, 클라이언트별 성능 비교, Non-IID 분포, 결함 검출 결과 시각화
"""

import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용 가능하도록
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import seaborn as sns

# 한글 폰트 설정 (Windows)
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
    except:
        pass

# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_training_curves(
    log_data: Dict[str, Any],
    save_path: Path,
    title: str = "연합학습 학습 곡선"
):
    """
    실시간 학습 곡선 시각화 (Loss, Accuracy)
    
    Args:
        log_data: 로거의 log_data 딕셔너리
        save_path: 저장 경로
        title: 그래프 제목
    """
    rounds = log_data.get('rounds', [])
    if not rounds:
        print("[시각화] 라운드 데이터가 없습니다.")
        return
    
    rounds_list = []
    avg_losses = []
    avg_accuracies = []
    
    for round_data in rounds:
        clients = round_data.get('clients', [])
        if clients:
            rounds_list.append(round_data['round'])
            avg_loss = sum(c.get('loss', 0) for c in clients) / len(clients)
            avg_acc = sum(c.get('accuracy', 0) for c in clients) / len(clients)
            avg_losses.append(avg_loss)
            avg_accuracies.append(avg_acc)
    
    if not rounds_list:
        print("[시각화] 시각화할 데이터가 없습니다.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss 곡선
    ax1.plot(rounds_list, avg_losses, 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_xlabel('라운드', fontsize=12)
    ax1.set_ylabel('평균 손실 (Loss)', fontsize=12)
    ax1.set_title('평균 손실 변화', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(rounds_list)
    
    # Accuracy 곡선
    ax2.plot(rounds_list, avg_accuracies, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax2.set_xlabel('라운드', fontsize=12)
    ax2.set_ylabel('평균 정확도 (Accuracy)', fontsize=12)
    ax2.set_title('평균 정확도 변화', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(rounds_list)
    ax2.set_ylim([0, 1])
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[시각화] 학습 곡선 저장: {save_path}")


def plot_client_performance_comparison(
    log_data: Dict[str, Any],
    save_path: Path,
    title: str = "클라이언트별 성능 비교"
):
    """
    클라이언트별 성능 비교 차트
    
    Args:
        log_data: 로거의 log_data 딕셔너리
        save_path: 저장 경로
        title: 그래프 제목
    """
    rounds = log_data.get('rounds', [])
    if not rounds:
        print("[시각화] 라운드 데이터가 없습니다.")
        return
    
    # 클라이언트별 데이터 수집
    client_data = {}
    
    for round_data in rounds:
        clients = round_data.get('clients', [])
        for client_stat in clients:
            client_id = client_stat.get('client_id', 'unknown')
            if client_id not in client_data:
                client_data[client_id] = {
                    'rounds': [],
                    'losses': [],
                    'accuracies': []
                }
            
            client_data[client_id]['rounds'].append(round_data['round'])
            client_data[client_id]['losses'].append(client_stat.get('loss', 0))
            client_data[client_id]['accuracies'].append(client_stat.get('accuracy', 0))
    
    if not client_data:
        print("[시각화] 클라이언트 데이터가 없습니다.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 클라이언트별 Loss 비교
    for client_id, data in sorted(client_data.items()):
        ax1.plot(data['rounds'], data['losses'], 'o-', linewidth=2, 
                markersize=6, label=f'클라이언트 {client_id}', alpha=0.8)
    
    ax1.set_xlabel('라운드', fontsize=12)
    ax1.set_ylabel('손실 (Loss)', fontsize=12)
    ax1.set_title('클라이언트별 손실 비교', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 클라이언트별 Accuracy 비교
    for client_id, data in sorted(client_data.items()):
        ax2.plot(data['rounds'], data['accuracies'], 'o-', linewidth=2,
                markersize=6, label=f'클라이언트 {client_id}', alpha=0.8)
    
    ax2.set_xlabel('라운드', fontsize=12)
    ax2.set_ylabel('정확도 (Accuracy)', fontsize=12)
    ax2.set_title('클라이언트별 정확도 비교', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[시각화] 클라이언트별 성능 비교 저장: {save_path}")


def plot_non_iid_distribution(
    client_distributions: Dict[int, Dict[str, Any]],
    defect_type_to_idx: Dict[str, int],
    save_path: Path,
    title: str = "Non-IID 데이터 분포"
):
    """
    Non-IID 분포 시각화
    
    Args:
        client_distributions: 클라이언트별 분포 정보
        defect_type_to_idx: 결함 유형 인덱스 매핑
        save_path: 저장 경로
        title: 그래프 제목
    """
    if not client_distributions:
        print("[시각화] 클라이언트 분포 데이터가 없습니다.")
        return
    
    # 클라이언트별 결함 유형 분포 수집
    num_clients = len(client_distributions)
    defect_types = sorted(defect_type_to_idx.keys())
    
    # 클라이언트별 각 결함 유형의 샘플 수
    distribution_matrix = []
    client_labels = []
    
    for client_id in sorted(client_distributions.keys()):
        client_labels.append(f'클라이언트 {client_id}')
        dist = client_distributions[client_id]
        defect_dist = dist.get('defect_distribution', {})
        
        # 각 결함 유형별 샘플 수
        counts = [defect_dist.get(dtype, 0) for dtype in defect_types]
        distribution_matrix.append(counts)
    
    if not distribution_matrix:
        print("[시각화] 분포 데이터가 없습니다.")
        return
    
    distribution_matrix = np.array(distribution_matrix)
    
    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(max(12, len(defect_types) * 0.8), max(6, num_clients * 0.6)))
    
    im = ax.imshow(distribution_matrix, cmap='YlOrRd', aspect='auto')
    
    # 축 레이블 설정
    ax.set_xticks(np.arange(len(defect_types)))
    ax.set_yticks(np.arange(num_clients))
    ax.set_xticklabels(defect_types, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(client_labels, fontsize=10)
    
    # 값 표시
    for i in range(num_clients):
        for j in range(len(defect_types)):
            value = distribution_matrix[i, j]
            if value > 0:
                text = ax.text(j, i, int(value), ha="center", va="center",
                             color="black" if value < distribution_matrix.max() * 0.5 else "white",
                             fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('결함 유형', fontsize=12)
    ax.set_ylabel('클라이언트', fontsize=12)
    
    # 컬러바
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('샘플 수', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[시각화] Non-IID 분포 저장: {save_path}")


def plot_defect_detection_results(
    detection_results: List[Dict[str, Any]],
    save_path: Path,
    title: str = "결함 검출 결과 시각화",
    max_samples: int = 16
):
    """
    결함 검출 결과 시각화 (이미지 그리드)
    
    Args:
        detection_results: 검출 결과 리스트 [{'image': np.ndarray, 'anomaly_regions': [...], ...}, ...]
        save_path: 저장 경로
        title: 그래프 제목
        max_samples: 최대 표시 샘플 수
    """
    if not detection_results:
        print("[시각화] 검출 결과 데이터가 없습니다.")
        return
    
    # 샘플 수 제한
    samples = detection_results[:max_samples]
    n_samples = len(samples)
    
    if n_samples == 0:
        return
    
    # 그리드 크기 계산
    cols = 4
    rows = (n_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, result in enumerate(samples):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        image = result.get('image')
        anomaly_regions = result.get('anomaly_regions', [])
        anomaly_score = result.get('anomaly_score', 0.0)
        
        if image is not None:
            # 이미지 표시
            if len(image.shape) == 3:
                ax.imshow(image)
            else:
                ax.imshow(image, cmap='gray')
            
            # 이상 영역 표시
            for region in anomaly_regions:
                x1 = region.get('x1', 0)
                y1 = region.get('y1', 0)
                x2 = region.get('x2', 0)
                y2 = region.get('y2', 0)
                
                # 바운딩박스 그리기
                rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            
            ax.set_title(f'Score: {anomaly_score:.3f}', fontsize=10)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No Image', ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    # 빈 서브플롯 숨기기
    for idx in range(n_samples, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[시각화] 결함 검출 결과 저장: {save_path}")


def plot_class_performance(
    metrics: Dict[str, Any],
    save_path: Path,
    title: str = "클래스별 성능 분석"
):
    """
    클래스별 성능 분석 차트 (Precision, Recall, F1-Score)
    
    Args:
        metrics: evaluate_model의 반환값
        save_path: 저장 경로
        title: 그래프 제목
    """
    per_class = metrics.get('per_class', {})
    if not per_class:
        print("[시각화] 클래스별 데이터가 없습니다.")
        return
    
    classes = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for class_name, class_metrics in sorted(per_class.items()):
        classes.append(class_name)
        precisions.append(class_metrics.get('precision', 0))
        recalls.append(class_metrics.get('recall', 0))
        f1_scores.append(class_metrics.get('f1_score', 0))
    
    if not classes:
        return
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 0.8), 6))
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x, recalls, width, label='Recall', alpha=0.8, color='#2ecc71')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='#e74c3c')
    
    ax.set_xlabel('클래스 (결함 유형)', fontsize=12)
    ax.set_ylabel('성능 점수', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # 값 표시
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:  # 너무 작은 값은 표시하지 않음
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[시각화] 클래스별 성능 분석 저장: {save_path}")


def create_all_visualizations(
    logger,
    client_distributions: Optional[Dict[int, Dict[str, Any]]] = None,
    defect_type_to_idx: Optional[Dict[str, int]] = None,
    final_metrics: Optional[Dict[str, Any]] = None
):
    """
    모든 시각화 생성
    
    Args:
        logger: FederatedLearningLogger 인스턴스
        client_distributions: 클라이언트별 분포 정보
        defect_type_to_idx: 결함 유형 인덱스 매핑
        final_metrics: 최종 평가 메트릭
    """
    if logger is None:
        return
    
    vis_dir = logger.get_log_path() / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    log_data = logger.log_data
    
    # 1. 학습 곡선
    plot_training_curves(
        log_data,
        vis_dir / "training_curves.png",
        title=f"{logger.experiment_name} - 학습 곡선"
    )
    
    # 2. 클라이언트별 성능 비교
    plot_client_performance_comparison(
        log_data,
        vis_dir / "client_performance_comparison.png",
        title=f"{logger.experiment_name} - 클라이언트별 성능 비교"
    )
    
    # 3. 클래스별 성능 분석
    if final_metrics:
        plot_class_performance(
            final_metrics,
            vis_dir / "class_performance.png",
            title=f"{logger.experiment_name} - 클래스별 성능 분석"
        )
    
    print(f"\n[시각화] 모든 그래프 저장 완료: {vis_dir}")

