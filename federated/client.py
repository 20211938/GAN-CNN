"""
연합학습 클라이언트
로컬 데이터로 학습하고 가중치를 서버로 전송
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional
import requests
import time
import copy
from tqdm import tqdm

from models.cnn import DefectClassifierCNN
from models.few_shot_cnn import HybridCNN


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
        learning_rate: float = 0.001,
        use_few_shot: bool = False,
        few_shot_loader: Optional[DataLoader] = None
    ) -> Dict:
        """
        로컬 데이터로 모델 학습
        
        Args:
            train_loader: 학습 데이터 로더 (일반 학습용)
            epochs: 에폭 수
            learning_rate: 학습률
            use_few_shot: 퓨샷 학습 모드 사용 여부
            few_shot_loader: 퓨샷 학습용 데이터 로더
            
        Returns:
            학습 통계 딕셔너리
        """
        if self.model is None:
            raise ValueError("모델이 초기화되지 않았습니다")
        
        # 퓨샷 학습 모드
        if use_few_shot and few_shot_loader is not None:
            return self._train_few_shot(few_shot_loader, epochs)
        
        # 일반 학습 모드
        print(f"\n{'='*70}")
        print(f"[클라이언트 {self.client_id}] 일반 학습 모드 시작")
        print(f"{'='*70}")
        print(f"  - 총 에폭: {epochs}")
        print(f"  - 학습률: {learning_rate}")
        print(f"  - 배치 수: {len(train_loader)}")
        print(f"  - 디바이스: {self.device}")
        print(f"{'='*70}\n")
        
        self.model.train()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        total_loss = 0.0
        total_samples = 0
        correct = 0
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            # 진행 바 생성
            pbar = tqdm(
                train_loader,
                desc=f"[클라이언트 {self.client_id}] Epoch {epoch+1}/{epochs}",
                unit="batch",
                ncols=100
            )
            
            for batch_idx, batch in enumerate(pbar):
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
                batch_size = images.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size
                
                # 정확도 계산
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                epoch_correct += batch_correct
                correct += batch_correct
                
                # 진행 바 업데이트
                current_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
                current_acc = epoch_correct / epoch_samples if epoch_samples > 0 else 0.0
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}',
                    'Batch': f'{batch_idx+1}/{len(train_loader)}'
                })
            
            pbar.close()
            
            total_loss += epoch_loss
            total_samples += epoch_samples
            
            avg_loss = epoch_loss / epoch_samples if epoch_samples > 0 else 0.0
            accuracy = correct / total_samples if total_samples > 0 else 0.0
            
            print(f"\n[클라이언트 {self.client_id}] Epoch {epoch+1}/{epochs} 완료")
            print(f"  ├─ 평균 손실: {avg_loss:.6f}")
            print(f"  ├─ 정확도: {accuracy:.4f} ({correct}/{total_samples})")
            print(f"  └─ 처리 샘플 수: {epoch_samples}개\n")
        
        final_stats = {
            'loss': total_loss / total_samples if total_samples > 0 else 0.0,
            'accuracy': accuracy,
            'samples': total_samples
        }
        
        print(f"[클라이언트 {self.client_id}] 학습 완료!")
        print(f"  ├─ 최종 손실: {final_stats['loss']:.6f}")
        print(f"  ├─ 최종 정확도: {final_stats['accuracy']:.4f}")
        print(f"  └─ 총 샘플 수: {final_stats['samples']}개\n")
        
        return final_stats
    
    def _train_few_shot(
        self,
        few_shot_loader: DataLoader,
        epochs: int = 1
    ) -> Dict:
        """
        퓨샷 학습 수행
        
        Args:
            few_shot_loader: 퓨샷 학습용 데이터 로더
            epochs: 에피소드 수
            
        Returns:
            학습 통계 딕셔너리
        """
        # HybridCNN 모델인지 확인
        if not hasattr(self.model, 'few_shot_episode'):
            raise ValueError("모델이 퓨샷 학습을 지원하지 않습니다. HybridCNN을 사용하세요.")
        
        print(f"\n{'='*70}")
        print(f"[클라이언트 {self.client_id}] 퓨샷 학습 모드 시작")
        print(f"{'='*70}")
        print(f"  - 총 에폭: {epochs}")
        print(f"  - 에피소드 수: {len(few_shot_loader)}")
        print(f"  - 디바이스: {self.device}")
        print(f"{'='*70}\n")
        
        total_accuracy = 0.0
        total_episodes = 0
        total_correct = 0
        total_query_samples = 0
        
        for epoch in range(epochs):
            epoch_accuracy = 0.0
            epoch_episodes = 0
            epoch_correct = 0
            epoch_query_samples = 0
            
            # 진행 바 생성
            pbar = tqdm(
                few_shot_loader,
                desc=f"[클라이언트 {self.client_id}] Few-shot Epoch {epoch+1}/{epochs}",
                unit="episode",
                ncols=100
            )
            
            for episode_idx, batch in enumerate(pbar):
                support_images = batch['support_images'][0].to(self.device)
                support_labels = batch['support_labels'][0].to(self.device)
                query_images = batch['query_images'][0].to(self.device)
                query_labels = batch['query_labels'][0].to(self.device)
                
                # 퓨샷 학습 에피소드 수행
                result = self.model.few_shot_episode(
                    support_images=support_images,
                    support_labels=support_labels,
                    query_images=query_images,
                    query_labels=query_labels
                )
                
                accuracy = result['accuracy'].item()
                episode_correct = result.get('correct', 0)
                episode_total = result.get('total', len(query_labels))
                
                epoch_accuracy += accuracy
                epoch_episodes += 1
                epoch_correct += episode_correct
                epoch_query_samples += episode_total
                
                # 진행 바 업데이트
                current_acc = epoch_accuracy / epoch_episodes if epoch_episodes > 0 else 0.0
                pbar.set_postfix({
                    'Acc': f'{current_acc:.4f}',
                    'Ep': f'{episode_idx+1}/{len(few_shot_loader)}'
                })
            
            pbar.close()
            
            avg_accuracy = epoch_accuracy / epoch_episodes if epoch_episodes > 0 else 0.0
            total_accuracy += epoch_accuracy
            total_episodes += epoch_episodes
            total_correct += epoch_correct
            total_query_samples += epoch_query_samples
            
            print(f"\n[클라이언트 {self.client_id}] Few-shot Epoch {epoch+1}/{epochs} 완료")
            print(f"  ├─ 평균 정확도: {avg_accuracy:.4f}")
            print(f"  ├─ 정확히 분류: {epoch_correct}/{epoch_query_samples}")
            print(f"  └─ 처리 에피소드 수: {epoch_episodes}개\n")
        
        final_accuracy = total_accuracy / total_episodes if total_episodes > 0 else 0.0
        
        final_stats = {
            'loss': 0.0,  # 퓨샷 학습은 loss 대신 accuracy 사용
            'accuracy': final_accuracy,
            'samples': total_episodes
        }
        
        print(f"[클라이언트 {self.client_id}] 퓨샷 학습 완료!")
        print(f"  ├─ 최종 정확도: {final_stats['accuracy']:.4f}")
        print(f"  ├─ 총 정확히 분류: {total_correct}/{total_query_samples}")
        print(f"  └─ 총 에피소드 수: {total_episodes}개\n")
        
        return final_stats
    
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
        
        print(f"[클라이언트 {self.client_id}] 가중치 업로드 준비 중...")
        
        # 가중치 가져오기
        weights = self.model.get_state_dict()
        
        # 가중치 크기 계산
        total_params = sum(p.numel() for p in weights.values())
        total_size_mb = sum(p.numel() * 4 / (1024 * 1024) for p in weights.values())  # float32 기준
        
        print(f"  ├─ 총 파라미터 수: {total_params:,}개")
        print(f"  ├─ 예상 크기: {total_size_mb:.2f} MB")
        print(f"  └─ 데이터 크기: {data_size}개 샘플")
        
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
            
            print(f"  └─ 서버로 전송 중... ({self.server_url})")
            response = requests.post(
                f'{self.server_url}/upload_weights',
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                response_data = response.json()
                received_clients = response_data.get('received_clients', 0)
                min_clients = response_data.get('min_clients', 0)
                print(f"[클라이언트 {self.client_id}] ✅ 가중치 업로드 성공!")
                print(f"  └─ 서버 수신 클라이언트: {received_clients}/{min_clients}")
                return True
            else:
                print(f"[클라이언트 {self.client_id}] ❌ 가중치 업로드 실패: {response.status_code}")
                return False
        
        except Exception as e:
            print(f"[클라이언트 {self.client_id}] ❌ 가중치 업로드 오류: {e}")
            return False
    
    def fetch_aggregated_weights(self, round_num: int) -> bool:
        """
        서버에서 집계된 가중치 가져오기
        
        Args:
            round_num: 라운드 번호
            
        Returns:
            가중치 가져오기 성공 여부
        """
        print(f"[클라이언트 {self.client_id}] 서버에서 가중치 요청 중...")
        try:
            # 타임아웃을 60초로 증가 (큰 가중치 전송 시간 고려)
            response = requests.get(f'{self.server_url}/get_weights', timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                server_round = data.get('round', 0)
                weight_format = data.get('format', 'json')
                
                # 라운드 확인
                if server_round < round_num:
                    print(f"  ⚠️  서버 가중치가 아직 업데이트되지 않았습니다 (서버 라운드: {server_round}, 요청 라운드: {round_num})")
                    return False
                
                # 가중치 역직렬화 및 로드
                weights = self._deserialize_weights(data['weights'], format=weight_format)
                self.model.load_state_dict(weights)
                
                print(f"[클라이언트 {self.client_id}] ✅ 집계된 가중치 수신 완료!")
                print(f"  ├─ 서버 라운드: {server_round}")
                print(f"  └─ 전송 형식: {weight_format}")
                return True
            else:
                print(f"[클라이언트 {self.client_id}] ❌ 가중치 가져오기 실패: {response.status_code}")
                return False
        
        except requests.exceptions.Timeout:
            print(f"[클라이언트 {self.client_id}] ❌ 가중치 가져오기 타임아웃 (60초 초과)")
            print(f"  💡 서버가 응답하는 데 시간이 오래 걸립니다. 네트워크 상태를 확인하세요.")
            return False
        except Exception as e:
            print(f"[클라이언트 {self.client_id}] ❌ 가중치 가져오기 오류: {e}")
            return False
    
    def _serialize_weights(self, weights: Dict) -> Dict:
        """
        가중치를 JSON 직렬화 가능한 형태로 변환
        큰 모델의 경우 매우 느리므로 바이너리 압축 방식 권장
        """
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
    
    def _deserialize_weights(self, weights, format: str = 'json') -> Dict:
        """
        가중치 역직렬화
        바이너리 압축 형식 또는 JSON 형식 모두 지원
        """
        import base64
        import pickle
        import gzip
        
        # 바이너리 압축 형식인 경우
        if format == 'binary_compressed' or (isinstance(weights, str) and len(weights) > 1000):
            try:
                # Base64 디코딩
                decoded = base64.b64decode(weights.encode('utf-8'))
                # 압축 해제
                decompressed = gzip.decompress(decoded)
                # Pickle 역직렬화
                deserialized = pickle.loads(decompressed)
                # 텐서로 변환 (이미 텐서인 경우 그대로 사용)
                result = {}
                for k, v in deserialized.items():
                    if isinstance(v, torch.Tensor):
                        result[k] = v
                    elif isinstance(v, (list, tuple, np.ndarray)):
                        result[k] = torch.tensor(v)
                    else:
                        result[k] = v
                return result
            except Exception as e:
                print(f"[클라이언트 {self.client_id}] ⚠️  바이너리 역직렬화 실패, JSON 형식으로 시도: {e}")
        
        # JSON 형식 (기존 방식)
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
        
        print(f"\n[클라이언트 {self.client_id}] 모델 평가 시작...")
        print(f"  └─ 평가 배치 수: {len(test_loader)}")
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        pbar = tqdm(
            test_loader,
            desc=f"[클라이언트 {self.client_id}] 평가 중",
            unit="batch",
            ncols=100
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                batch_size = images.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                _, predicted = torch.max(outputs.data, 1)
                batch_correct = (predicted == labels).sum().item()
                correct += batch_correct
                
                # 진행 바 업데이트
                current_acc = correct / total_samples if total_samples > 0 else 0.0
                current_loss = total_loss / total_samples if total_samples > 0 else 0.0
                pbar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.4f}'
                })
        
        pbar.close()
        
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        
        print(f"\n[클라이언트 {self.client_id}] ✅ 평가 완료!")
        print(f"  ├─ 평균 손실: {avg_loss:.6f}")
        print(f"  ├─ 정확도: {accuracy:.4f} ({correct}/{total_samples})")
        print(f"  └─ 평가 샘플 수: {total_samples}개\n")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'samples': total_samples
        }

