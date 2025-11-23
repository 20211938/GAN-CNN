"""
모델 체크포인트 저장/로드 유틸리티
학습 중 모델 상태를 저장하고 복원하는 기능 제공
"""

import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
import json


class CheckpointManager:
    """
    모델 체크포인트 관리 클래스
    라운드별 체크포인트, 최고 성능 모델, 최신 모델을 관리
    """
    
    def __init__(
        self,
        checkpoint_dir: Path = Path("checkpoints"),
        experiment_name: Optional[str] = None,
        save_best: bool = True,
        save_latest: bool = True,
        save_rounds: bool = True
    ):
        """
        Args:
            checkpoint_dir: 체크포인트 저장 디렉토리
            experiment_name: 실험 이름 (None이면 타임스탬프 사용)
            save_best: 최고 성능 모델 저장 여부
            save_latest: 최신 모델 저장 여부
            save_rounds: 라운드별 체크포인트 저장 여부
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험 이름 생성
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.checkpoint_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best = save_best
        self.save_latest = save_latest
        self.save_rounds = save_rounds
        
        # 최고 성능 추적
        self.best_accuracy = 0.0
        self.best_round = 0
        
        print(f"[체크포인트] 저장 디렉토리: {self.experiment_dir}")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        round_num: int,
        metrics: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        is_best: bool = False
    ) -> Path:
        """
        체크포인트 저장
        
        Args:
            model: 저장할 모델
            round_num: 라운드 번호
            metrics: 성능 메트릭 (accuracy, loss 등)
            config: 하이퍼파라미터 설정 (선택사항)
            optimizer: 옵티마이저 상태 (선택사항)
            is_best: 최고 성능 모델인지 여부
            
        Returns:
            저장된 체크포인트 파일 경로
        """
        checkpoint = {
            'round': round_num,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': config or {},
            'timestamp': datetime.now().isoformat(),
            'experiment_name': self.experiment_name
        }
        
        # 옵티마이저 상태 추가 (있는 경우)
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        saved_paths = []
        
        # 라운드별 체크포인트 저장
        if self.save_rounds:
            round_path = self.experiment_dir / f"round_{round_num:03d}.pth"
            torch.save(checkpoint, round_path)
            saved_paths.append(round_path)
            print(f"[체크포인트] 라운드 {round_num} 저장: {round_path.name}")
        
        # 최신 모델 저장
        if self.save_latest:
            latest_path = self.experiment_dir / "latest_model.pth"
            torch.save(checkpoint, latest_path)
            saved_paths.append(latest_path)
            print(f"[체크포인트] 최신 모델 저장: {latest_path.name}")
        
        # 최고 성능 모델 저장
        if self.save_best and is_best:
            best_path = self.experiment_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            saved_paths.append(best_path)
            print(f"[체크포인트] ✅ 최고 성능 모델 저장: {best_path.name} (정확도: {metrics.get('accuracy', 0):.4f})")
        
        # 메타데이터 JSON 저장
        metadata_path = self.experiment_dir / "checkpoint_metadata.json"
        self._save_metadata(metadata_path, checkpoint)
        
        return saved_paths[0] if saved_paths else None
    
    def _save_metadata(self, metadata_path: Path, checkpoint: Dict):
        """체크포인트 메타데이터를 JSON으로 저장"""
        metadata = {
            'round': checkpoint['round'],
            'timestamp': checkpoint['timestamp'],
            'metrics': checkpoint['metrics'],
            'config': checkpoint.get('config', {})
        }
        
        # 기존 메타데이터 로드 (있는 경우)
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
                    # 라운드별 메타데이터 리스트로 저장
                    if 'rounds' not in existing_metadata:
                        existing_metadata['rounds'] = []
                    existing_metadata['rounds'].append(metadata)
                    metadata = existing_metadata
            except Exception as e:
                print(f"[체크포인트] ⚠️  메타데이터 로드 실패: {e}")
                metadata = {'rounds': [metadata]}
        else:
            metadata = {'rounds': [metadata]}
        
        # 메타데이터 저장
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        체크포인트 로드
        
        Args:
            checkpoint_path: 체크포인트 파일 경로
            model: 가중치를 로드할 모델
            optimizer: 옵티마이저 (선택사항)
            device: 디바이스 (선택사항)
            
        Returns:
            체크포인트 딕셔너리 (metrics, config 등 포함)
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 모델 가중치 로드
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 옵티마이저 상태 로드 (있는 경우)
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"[체크포인트] ✅ 체크포인트 로드 완료: {checkpoint_path.name}")
        print(f"  - 라운드: {checkpoint.get('round', 'N/A')}")
        print(f"  - 정확도: {checkpoint.get('metrics', {}).get('accuracy', 0):.4f}")
        
        return checkpoint
    
    def load_best(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """최고 성능 모델 로드"""
        best_path = self.experiment_dir / "best_model.pth"
        return self.load_checkpoint(best_path, model, optimizer)
    
    def load_latest(self, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """최신 모델 로드"""
        latest_path = self.experiment_dir / "latest_model.pth"
        return self.load_checkpoint(latest_path, model, optimizer)
    
    def load_round(self, round_num: int, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """특정 라운드 체크포인트 로드"""
        round_path = self.experiment_dir / f"round_{round_num:03d}.pth"
        return self.load_checkpoint(round_path, model, optimizer)
    
    def update_best(self, accuracy: float, round_num: int) -> bool:
        """
        최고 성능 업데이트
        
        Args:
            accuracy: 현재 정확도
            round_num: 라운드 번호
            
        Returns:
            최고 성능이 갱신되었는지 여부
        """
        if accuracy > self.best_accuracy:
            old_best = self.best_accuracy
            self.best_accuracy = accuracy
            self.best_round = round_num
            print(f"[체크포인트] 🎯 최고 성능 갱신: {old_best:.4f} → {accuracy:.4f} (라운드 {round_num})")
            return True
        return False
    
    def get_checkpoint_dir(self) -> Path:
        """체크포인트 디렉토리 경로 반환"""
        return self.experiment_dir


def create_checkpoint_manager(
    checkpoint_dir: Path = Path("checkpoints"),
    experiment_name: Optional[str] = None,
    **kwargs
) -> CheckpointManager:
    """
    체크포인트 매니저 생성 헬퍼 함수
    
    Args:
        checkpoint_dir: 체크포인트 저장 디렉토리
        experiment_name: 실험 이름
        **kwargs: 추가 옵션
        
    Returns:
        CheckpointManager 인스턴스
    """
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        experiment_name=experiment_name,
        **kwargs
    )

