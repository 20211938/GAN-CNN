"""
연합학습 클라이언트
로컬 데이터로 학습하고 가중치를 서버로 전송
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional
import requests
import time
import copy

from models.cnn import DefectClassifierCNN


class FederatedClient:
    """
    연합학습 클라이언트 클래스
    로컬 데이터로 CNN 모델을 학습하고 가중치를 서버와 교환
    """
    
    def __init__(
        self,
        client_id: int,
        server_url: str = 'http://localhost:5000',
        model: Optional[DefectClassifierCNN] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            client_id: 클라이언트 ID
            server_url: 서버 URL
            model: CNN 모델 (None이면 서버에서 가져옴)
            device: 학습 디바이스
        """
        self.client_id = client_id
        self.server_url = server_url
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화
        if model is None:
            # 서버에서 초기 가중치 가져오기
            self.model = None
            self._fetch_initial_weights()
        else:
            self.model = model.to(self.device)
        
        # 학습 설정
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
    
    def _fetch_initial_weights(self):
        """서버에서 초기 가중치 가져오기"""
        try:
            response = requests.get(f'{self.server_url}/get_weights', timeout=10)
            if response.status_code == 200:
                data = response.json()
                weights = self._deserialize_weights(data['weights'])
                
                # 모델 구조는 서버에서 알 수 없으므로, 임시로 작은 모델 생성
                # 실제로는 서버에서 모델 구조 정보도 함께 전송해야 함
                num_classes = self._infer_num_classes(weights)
                self.model = DefectClassifierCNN(num_classes=num_classes).to(self.device)
                self.model.load_state_dict(weights)
                print(f"[클라이언트 {self.client_id}] 서버에서 초기 가중치 수신 완료")
            else:
                print(f"[클라이언트 {self.client_id}] 서버에 가중치가 없습니다. 랜덤 초기화")
                # 기본 모델 생성 (실제로는 서버와 동일한 구조여야 함)
                self.model = DefectClassifierCNN(num_classes=10).to(self.device)
        except Exception as e:
            print(f"[클라이언트 {self.client_id}] 서버 연결 실패: {e}")
            # 기본 모델 생성
            self.model = DefectClassifierCNN(num_classes=10).to(self.device)
    
    def _infer_num_classes(self, weights: Dict) -> int:
        """가중치에서 클래스 수 추론"""
        # 마지막 레이어의 출력 크기에서 클래스 수 추론
        for key in reversed(list(weights.keys())):
            if 'classifier' in key and 'weight' in key:
                return weights[key].shape[0]
        return 10  # 기본값
    
    def train_local(
        self,
        train_loader: DataLoader,
        epochs: int = 1,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        로컬 데이터로 모델 학습
        
        Args:
            train_loader: 학습 데이터 로더
            epochs: 에폭 수
            learning_rate: 학습률
            
        Returns:
            학습 통계 딕셔너리
        """
        if self.model is None:
            raise ValueError("모델이 초기화되지 않았습니다")
        
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        total_loss = 0.0
        total_samples = 0
        correct = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            
            for batch in train_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # 통계
                epoch_loss += loss.item() * images.size(0)
                epoch_samples += images.size(0)
                
                # 정확도 계산
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            avg_loss = epoch_loss / epoch_samples
            accuracy = correct / total_samples if total_samples > 0 else 0.0
            
            print(f"[클라이언트 {self.client_id}] Epoch {epoch+1}/{epochs} - "
                  f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return {
            'loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'accuracy': accuracy,
            'samples': total_samples
        }
    
    def upload_weights(self, round_num: int, data_size: int) -> bool:
        """
        학습된 가중치를 서버로 업로드
        
        Args:
            round_num: 현재 라운드 번호
            data_size: 사용한 데이터 크기
            
        Returns:
            업로드 성공 여부
        """
        if self.model is None:
            raise ValueError("모델이 없습니다")
        
        # 가중치 가져오기
        weights = self.model.get_state_dict()
        
        # 직렬화
        weights_serialized = self._serialize_weights(weights)
        
        # 서버로 전송
        try:
            payload = {
                'client_id': self.client_id,
                'weights': weights_serialized,
                'data_size': data_size,
                'round': round_num
            }
            
            response = requests.post(
                f'{self.server_url}/upload_weights',
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"[클라이언트 {self.client_id}] 가중치 업로드 성공")
                return True
            else:
                print(f"[클라이언트 {self.client_id}] 가중치 업로드 실패: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"[클라이언트 {self.client_id}] 가중치 업로드 오류: {e}")
            return False
    
    def fetch_aggregated_weights(self, round_num: int) -> bool:
        """
        서버에서 집계된 가중치 가져오기
        
        Args:
            round_num: 라운드 번호
            
        Returns:
            가중치 가져오기 성공 여부
        """
        try:
            response = requests.get(f'{self.server_url}/get_weights', timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # 라운드 확인
                if data.get('round', 0) < round_num:
                    print(f"[클라이언트 {self.client_id}] 서버 가중치가 아직 업데이트되지 않았습니다")
                    return False
                
                # 가중치 역직렬화 및 로드
                weights = self._deserialize_weights(data['weights'])
                self.model.load_state_dict(weights)
                
                print(f"[클라이언트 {self.client_id}] 집계된 가중치 수신 완료 (라운드 {data['round']})")
                return True
            else:
                print(f"[클라이언트 {self.client_id}] 가중치 가져오기 실패: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"[클라이언트 {self.client_id}] 가중치 가져오기 오류: {e}")
            return False
    
    def _serialize_weights(self, weights: Dict) -> Dict:
        """가중치를 JSON 직렬화 가능한 형태로 변환"""
        serialized = {}
        for key, value in weights.items():
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
                tensor = torch.tensor(value['data'], dtype=torch.float32)
                tensor = tensor.reshape(value['shape'])
                deserialized[key] = tensor
            else:
                deserialized[key] = value
        return deserialized
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """
        모델 평가
        
        Args:
            test_loader: 테스트 데이터 로더
            
        Returns:
            평가 결과 딕셔너리
        """
        if self.model is None:
            raise ValueError("모델이 없습니다")
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total_samples
        }

