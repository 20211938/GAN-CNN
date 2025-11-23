"""
학습 로그 기록 유틸리티
학습 과정, 평가 결과, 연합학습 통계를 파일로 저장
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import os


class FederatedLearningLogger:
    """
    연합학습 로그 기록 클래스
    학습 과정과 결과를 JSON 및 CSV 형식으로 저장
    """
    
    def __init__(
        self,
        log_dir: Path = Path("logs"),
        experiment_name: Optional[str] = None,
        save_json: bool = True,
        save_csv: bool = True
    ):
        """
        Args:
            log_dir: 로그 저장 디렉토리
            experiment_name: 실험 이름 (None이면 타임스탬프 사용)
            save_json: JSON 형식으로 저장 여부
            save_csv: CSV 형식으로 저장 여부
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 실험 이름 생성
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"federated_learning_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_json = save_json
        self.save_csv = save_csv
        
        # 로그 데이터 저장
        self.log_data = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': {},
            'aprilgan_evaluation': {},
            'rounds': [],
            'client_distributions': {},
            'final_results': {}
        }
        
        # CSV 파일 경로
        self.csv_rounds_path = self.experiment_dir / "rounds.csv"
        self.csv_clients_path = self.experiment_dir / "clients.csv"
        
        # CSV 헤더 작성
        if self.save_csv:
            self._init_csv_files()
        
        print(f"\n[로거] 로그 디렉토리: {self.experiment_dir}")
    
    def _init_csv_files(self):
        """CSV 파일 초기화 및 헤더 작성"""
        # 라운드별 통계 CSV
        with open(self.csv_rounds_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'timestamp', 'total_clients', 'aggregated',
                'avg_loss', 'avg_accuracy', 'total_samples'
            ])
        
        # 클라이언트별 통계 CSV
        with open(self.csv_clients_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'round', 'client_id', 'timestamp', 'loss', 'accuracy',
                'samples', 'data_size'
            ])
    
    def log_config(self, config: Dict[str, Any]):
        """
        실험 설정 기록
        
        Args:
            config: 실험 설정 딕셔너리
        """
        self.log_data['config'] = config
        print(f"[로거] 실험 설정 기록 완료")
    
    def log_aprilgan_evaluation(self, eval_results: Dict[str, Any]):
        """
        AprilGAN 제로샷 모델 평가 결과 기록
        
        Args:
            eval_results: evaluate_aprilgan_detection의 반환값
        """
        self.log_data['aprilgan_evaluation'] = {
            'timestamp': datetime.now().isoformat(),
            'total_images': eval_results.get('total_images', 0),
            'total_detections': eval_results.get('total_detections', 0),
            'total_ground_truth': eval_results.get('total_ground_truth', 0),
            'true_positives': eval_results.get('true_positives', 0),
            'false_positives': eval_results.get('false_positives', 0),
            'false_negatives': eval_results.get('false_negatives', 0),
            'precision': eval_results.get('precision', 0.0),
            'recall': eval_results.get('recall', 0.0),
            'f1_score': eval_results.get('f1_score', 0.0),
            'mean_iou': eval_results.get('mean_iou', 0.0),
            'iou_threshold': eval_results.get('iou_threshold', 0.5)
        }
        print(f"[로거] AprilGAN 평가 결과 기록 완료")
    
    def log_client_distribution(
        self,
        client_distributions: Dict[int, Dict[str, Any]]
    ):
        """
        클라이언트별 데이터 분포 기록
        
        Args:
            client_distributions: 클라이언트별 분포 정보
        """
        self.log_data['client_distributions'] = client_distributions
        
        # 분포 정보 출력
        print(f"\n[로거] 클라이언트별 데이터 분포 기록:")
        for client_id, dist in client_distributions.items():
            print(f"  클라이언트 {client_id}: {dist.get('total_samples', 0)}개 샘플")
    
    def log_round(
        self,
        round_num: int,
        client_stats: List[Dict[str, Any]],
        server_stats: Optional[Dict[str, Any]] = None
    ):
        """
        연합학습 라운드 기록
        
        Args:
            round_num: 라운드 번호
            client_stats: 클라이언트별 통계 리스트
            server_stats: 서버 집계 통계 (선택사항)
        """
        round_data = {
            'round': round_num,
            'timestamp': datetime.now().isoformat(),
            'clients': client_stats,
            'server': server_stats
        }
        
        self.log_data['rounds'].append(round_data)
        
        # CSV에 기록
        if self.save_csv:
            self._write_round_to_csv(round_num, client_stats, server_stats)
            self._write_clients_to_csv(round_num, client_stats)
        
        print(f"[로거] 라운드 {round_num} 기록 완료")
    
    def _write_round_to_csv(
        self,
        round_num: int,
        client_stats: List[Dict[str, Any]],
        server_stats: Optional[Dict[str, Any]]
    ):
        """라운드 통계를 CSV에 기록"""
        avg_loss = sum(c.get('loss', 0) for c in client_stats) / len(client_stats) if client_stats else 0
        avg_accuracy = sum(c.get('accuracy', 0) for c in client_stats) / len(client_stats) if client_stats else 0
        total_samples = sum(c.get('samples', 0) for c in client_stats)
        
        with open(self.csv_rounds_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                round_num,
                datetime.now().isoformat(),
                len(client_stats),
                'yes' if server_stats else 'no',
                f"{avg_loss:.6f}",
                f"{avg_accuracy:.4f}",
                total_samples
            ])
    
    def _write_clients_to_csv(
        self,
        round_num: int,
        client_stats: List[Dict[str, Any]]
    ):
        """클라이언트별 통계를 CSV에 기록"""
        with open(self.csv_clients_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for client_stat in client_stats:
                writer.writerow([
                    round_num,
                    client_stat.get('client_id', 'unknown'),
                    datetime.now().isoformat(),
                    f"{client_stat.get('loss', 0):.6f}",
                    f"{client_stat.get('accuracy', 0):.4f}",
                    client_stat.get('samples', 0),
                    client_stat.get('data_size', 0)
                ])
    
    def log_final_results(self, results: Dict[str, Any]):
        """
        최종 결과 기록
        
        Args:
            results: 최종 결과 딕셔너리
        """
        self.log_data['final_results'] = results
        self.log_data['end_time'] = datetime.now().isoformat()
        
        # 실행 시간 계산
        start_time = datetime.fromisoformat(self.log_data['start_time'])
        end_time = datetime.fromisoformat(self.log_data['end_time'])
        duration = (end_time - start_time).total_seconds()
        self.log_data['duration_seconds'] = duration
        
        print(f"\n[로거] 최종 결과 기록 완료")
        print(f"  실행 시간: {duration:.2f}초 ({duration/60:.2f}분)")
    
    def save(self, create_visualizations: bool = True):
        """
        로그를 파일로 저장
        
        Args:
            create_visualizations: 시각화 생성 여부
        """
        if self.save_json:
            json_path = self.experiment_dir / "experiment_log.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.log_data, f, indent=2, ensure_ascii=False)
            print(f"[로거] JSON 로그 저장: {json_path}")
        
        # 요약 파일 생성
        self._save_summary()
        
        # 시각화 생성
        if create_visualizations:
            try:
                from .visualization import create_all_visualizations
                create_all_visualizations(self)
            except Exception as e:
                print(f"[로거] ⚠️  시각화 생성 실패: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"[로거] 모든 로그 저장 완료: {self.experiment_dir}")
    
    def _save_summary(self):
        """실험 요약 저장"""
        summary = {
            'experiment_name': self.experiment_name,
            'start_time': self.log_data['start_time'],
            'end_time': self.log_data.get('end_time', 'N/A'),
            'duration_seconds': self.log_data.get('duration_seconds', 0),
            'config': self.log_data['config'],
            'total_rounds': len(self.log_data['rounds']),
            'num_clients': len(self.log_data.get('client_distributions', {})),
            'final_results': self.log_data.get('final_results', {})
        }
        
        # 라운드별 요약
        if self.log_data['rounds']:
            round_summaries = []
            for round_data in self.log_data['rounds']:
                clients = round_data.get('clients', [])
                if clients:
                    avg_loss = sum(c.get('loss', 0) for c in clients) / len(clients)
                    avg_acc = sum(c.get('accuracy', 0) for c in clients) / len(clients)
                    round_summaries.append({
                        'round': round_data['round'],
                        'avg_loss': avg_loss,
                        'avg_accuracy': avg_acc,
                        'num_clients': len(clients)
                    })
            summary['round_summaries'] = round_summaries
        
        summary_path = self.experiment_dir / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"[로거] 실험 요약 저장: {summary_path}")
    
    def get_log_path(self) -> Path:
        """로그 디렉토리 경로 반환"""
        return self.experiment_dir


def create_logger(
    log_dir: Path = Path("logs"),
    experiment_name: Optional[str] = None,
    **kwargs
) -> FederatedLearningLogger:
    """
    로거 생성 헬퍼 함수
    
    Args:
        log_dir: 로그 저장 디렉토리
        experiment_name: 실험 이름
        **kwargs: 추가 옵션
        
    Returns:
        FederatedLearningLogger 인스턴스
    """
    return FederatedLearningLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        **kwargs
    )

