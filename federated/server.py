"""
연합학습 서버
클라이언트로부터 가중치를 수신하고 평균화하여 배포
"""

import json
import time
from typing import Dict, List, Optional
from flask import Flask, request, jsonify
import torch
import threading

from .aggregator import FederatedAveraging


class FederatedServer:
    """
    연합학습 서버 클래스
    Flask 기반 REST API로 클라이언트와 통신
    """
    
    def __init__(
        self,
        port: int = 5000,
        num_clients: int = 3,
        min_clients: int = 2
    ):
        """
        Args:
            port: 서버 포트
            num_clients: 전체 클라이언트 수
            min_clients: 최소 참여 클라이언트 수
        """
        self.port = port
        self.num_clients = num_clients
        self.min_clients = min_clients
        
        self.app = Flask(__name__)
        self.setup_routes()
        
        # 서버 상태
        self.current_round = 0
        self.client_weights: Dict[int, Dict] = {}
        self.client_sizes: Dict[int, int] = {}
        self.aggregated_weights: Optional[Dict] = None
        
        # 동기화
        self.lock = threading.Lock()
        
        # Aggregator
        self.aggregator = FederatedAveraging()
    
    def setup_routes(self):
        """Flask 라우트 설정"""
        
        @self.app.route('/health', methods=['GET'])
        def health():
            return jsonify({'status': 'healthy'})
        
        @self.app.route('/get_weights', methods=['GET'])
        def get_weights():
            """클라이언트가 최신 가중치를 요청"""
            if self.aggregated_weights is None:
                return jsonify({'error': '가중치가 아직 없습니다'}), 404
            
            # 가중치를 JSON 직렬화 가능한 형태로 변환
            weights_serialized = self._serialize_weights(self.aggregated_weights)
            return jsonify({
                'round': self.current_round,
                'weights': weights_serialized
            })
        
        @self.app.route('/upload_weights', methods=['POST'])
        def upload_weights():
            """클라이언트가 학습된 가중치를 업로드"""
            try:
                data = request.json
                client_id = data.get('client_id')
                weights = data.get('weights')
                data_size = data.get('data_size', 1)
                round_num = data.get('round', self.current_round)
                
                if client_id is None or weights is None:
                    return jsonify({'error': '필수 필드가 없습니다'}), 400
                
                # 가중치 역직렬화
                weights_deserialized = self._deserialize_weights(weights)
                
                # 가중치 크기 계산
                total_params = sum(p.numel() for p in weights_deserialized.values())
                
                with self.lock:
                    self.client_weights[client_id] = weights_deserialized
                    self.client_sizes[client_id] = data_size
                    received_count = len(self.client_weights)
                
                print(f"[서버] ✅ 클라이언트 {client_id}로부터 가중치 수신")
                print(f"  ├─ 라운드: {round_num}")
                print(f"  ├─ 파라미터 수: {total_params:,}개")
                print(f"  ├─ 데이터 크기: {data_size}개 샘플")
                print(f"  └─ 수신된 클라이언트: {received_count}/{self.min_clients}")
                
                # 충분한 클라이언트가 모이면 집계
                if received_count >= self.min_clients:
                    self._aggregate_weights()
                
                return jsonify({
                    'status': 'success',
                    'received_clients': received_count,
                    'min_clients': self.min_clients
                })
            
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/reset', methods=['POST'])
        def reset():
            """서버 상태 초기화 (테스트용)"""
            with self.lock:
                self.current_round = 0
                self.client_weights = {}
                self.client_sizes = {}
                self.aggregated_weights = None
            return jsonify({'status': 'reset'})
    
    def _serialize_weights(self, weights: Dict) -> Dict:
        """가중치를 JSON 직렬화 가능한 형태로 변환"""
        serialized = {}
        for key, value in weights.items():
            # 텐서를 리스트로 변환
            if isinstance(value, torch.Tensor):
                serialized[key] = {
                    'data': value.cpu().numpy().tolist(),
                    'shape': list(value.shape),
                    'dtype': str(value.dtype)
                }
            else:
                serialized[key] = value
        return serialized
    
    def _deserialize_weights(self, weights: Dict) -> Dict:
        """JSON에서 가중치 역직렬화"""
        deserialized = {}
        for key, value in weights.items():
            if isinstance(value, dict) and 'data' in value:
                # 텐서로 복원
                tensor = torch.tensor(value['data'], dtype=torch.float32)
                tensor = tensor.reshape(value['shape'])
                deserialized[key] = tensor
            else:
                deserialized[key] = value
        return deserialized
    
    def _aggregate_weights(self):
        """가중치 집계"""
        if len(self.client_weights) < self.min_clients:
            print(f"[서버] ⚠️  클라이언트 수 부족 ({len(self.client_weights)}/{self.min_clients})")
            return
        
        print(f"\n{'='*70}")
        print(f"[서버] 가중치 집계 시작")
        print(f"{'='*70}")
        print(f"  ├─ 참여 클라이언트 수: {len(self.client_weights)}개")
        print(f"  ├─ 최소 요구 클라이언트: {self.min_clients}개")
        
        # 클라이언트별 데이터 크기 출력
        total_data_size = 0
        for cid in self.client_weights.keys():
            data_size = self.client_sizes[cid]
            total_data_size += data_size
            print(f"  │  └─ 클라이언트 {cid}: {data_size}개 샘플")
        
        print(f"  ├─ 총 데이터 크기: {total_data_size}개 샘플")
        
        # 클라이언트 가중치와 크기 리스트 생성
        weights_list = list(self.client_weights.values())
        sizes_list = [self.client_sizes[cid] for cid in self.client_weights.keys()]
        
        # 가중치 통계 계산
        sample_weights = [w / total_data_size for w in sizes_list]
        print(f"  ├─ 가중치 분포:")
        for idx, (cid, weight) in enumerate(zip(self.client_weights.keys(), sample_weights)):
            print(f"  │  └─ 클라이언트 {cid}: {weight:.4f} ({sizes_list[idx]}개)")
        
        print(f"  └─ Federated Averaging 수행 중...")
        
        # Federated Averaging 수행
        self.aggregated_weights = self.aggregator.aggregate(
            weights_list,
            sizes_list
        )
        
        self.current_round += 1
        
        # 집계된 가중치 통계
        if self.aggregated_weights:
            total_params = sum(p.numel() for p in self.aggregated_weights.values())
            print(f"\n[서버] ✅ 가중치 집계 완료!")
            print(f"  ├─ 라운드: {self.current_round}")
            print(f"  ├─ 총 파라미터 수: {total_params:,}개")
            print(f"  └─ 집계 방식: Federated Averaging (가중 평균)")
        
        print(f"{'='*70}\n")
        
        # 클라이언트 가중치 초기화 (다음 라운드 준비)
        self.client_weights = {}
        self.client_sizes = {}
    
    def start(self, host: str = '0.0.0.0', debug: bool = False):
        """서버 시작"""
        print(f"[서버] 연합학습 서버 시작 (포트: {self.port})")
        self.app.run(host=host, port=self.port, debug=debug, threaded=True)
    
    def get_aggregated_weights(self) -> Optional[Dict]:
        """집계된 가중치 반환"""
        return self.aggregated_weights
    
    def set_initial_weights(self, weights: Dict):
        """초기 가중치 설정"""
        self.aggregated_weights = weights
        print("[서버] 초기 가중치 설정 완료")

