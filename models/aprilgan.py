"""
AprilGAN: 제로샷 비전 이상탐지 모델
사전 학습된 모델로 추가 학습 없이 이상 영역을 탐지
"""

import numpy as np
import cv2
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class AprilGAN:
    """
    AprilGAN 제로샷 이상탐지 모델 래퍼
    
    실제 AprilGAN 모델이 구현되어 있지 않은 경우를 대비한
    시뮬레이션/스텁 구현입니다.
    실제 프로젝트에서는 실제 AprilGAN 모델로 교체해야 합니다.
    """
    
<<<<<<< HEAD
    def __init__(self, model_path: Optional[str] = None, model_name: str = "dinov2_vits14"):
        """
        Args:
            model_path: 사전 학습된 모델 경로 (None이면 timm에서 자동 다운로드)
            model_name: 사용할 DINOv2 모델 이름 (기본값: dinov2_vits14)
                        옵션: dinov2_vits14 (small), dinov2_vitb14_reg (base), dinov2_vitl14_reg (large)
=======
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: 사전 학습된 모델 경로 (None이면 기본 모델 사용)
>>>>>>> parent of f73de4e (이것저것)
        """
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model()
    
    def _load_model(self):
        """
        사전 학습된 모델 로드
        실제 구현에서는 실제 AprilGAN 모델을 로드해야 합니다.
        """
<<<<<<< HEAD
        try:
            if self.model_path:
                # 사용자 지정 모델 경로에서 로드
                model_path_obj = Path(self.model_path)
                if not model_path_obj.exists():
                    raise RuntimeError(f"AprilGAN 모델 파일을 찾을 수 없습니다: {self.model_path}")
                
                print(f"[AprilGAN] 모델 로드 중: {self.model_path}")
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # timm으로 모델 구조 생성
                self.model = timm.create_model(self.model_name, pretrained=False)
                
                # Checkpoint 로드
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['state_dict'])
                    elif 'model' in checkpoint:
                        self.model.load_state_dict(checkpoint['model'])
                    else:
                        self.model.load_state_dict(checkpoint)
                else:
                    self.model.load_state_dict(checkpoint)
                
                print("[AprilGAN] ✅ 모델 로드 완료")
            else:
                # timm에서 사전 학습된 모델 자동 다운로드
                print(f"[AprilGAN] 사전 학습된 DINOv2 모델 로드 중: {self.model_name}")
                
                # timm에서 사용 가능한 모든 DINOv2 모델 찾기
                try:
                    all_models = timm.list_models(pretrained=True)
                    dinov2_models = [m for m in all_models if 'dinov2' in m.lower()]
                    
                    if dinov2_models:
                        print(f"[AprilGAN] 사용 가능한 DINOv2 모델: {', '.join(dinov2_models[:5])}")
                    else:
                        print("[AprilGAN] 경고: timm에서 DINOv2 모델을 찾을 수 없습니다.")
                except Exception:
                    dinov2_models = []
                
                # 사용 가능한 DINOv2 모델 이름 목록 (우선순위 순)
                fallback_models = [
                    "dinov2_vits14",  # Small (21M params)
                    "dinov2_vits14_reg",  # Small with reg
                    "dinov2_vitb14",  # Base
                    "dinov2_vitb14_reg",  # Base with reg
                    "dinov2_vitb14_reg4",  # Base variant
                    "dinov2_vitl14",  # Large
                    "dinov2_vitl14_reg",  # Large with reg
                    "dinov2_vitg14",  # Giant
                ]
                
                # 모델 로드 시도: 실제 사용 가능한 모델을 우선 시도
                model_loaded = False
                
                # 1. 실제 사용 가능한 DINOv2 모델이 있으면 그것을 먼저 시도
                if dinov2_models:
                    # 사용 가능한 모델 중에서 우선순위가 높은 것 선택
                    preferred_from_available = None
                    for preferred in fallback_models:
                        if preferred in dinov2_models:
                            preferred_from_available = preferred
                            break
                    
                    # 사용 가능한 모델이 있으면 그것을 먼저 시도
                    if preferred_from_available:
                        models_to_try = [preferred_from_available] + [m for m in dinov2_models if m != preferred_from_available]
                    else:
                        # 사용 가능한 모델 중 첫 번째 것 사용
                        models_to_try = dinov2_models
                else:
                    # 사용 가능한 모델이 없으면 fallback 모델 시도
                    models_to_try = [self.model_name] + [m for m in fallback_models if m != self.model_name]
                
                for model_name in models_to_try:
                    try:
                        self.model = timm.create_model(model_name, pretrained=True)
                        self.model_name = model_name  # 실제 로드된 모델 이름으로 업데이트
                        print(f"[AprilGAN] ✅ 사전 학습된 모델 로드 완료: {model_name}")
                        model_loaded = True
                        break
                    except (RuntimeError, ValueError, KeyError) as e:
                        # 모델을 찾을 수 없는 경우 다음 모델 시도
                        continue
                
                if not model_loaded:
                    # 모든 모델 시도 실패
                    available_info = ""
                    if dinov2_models:
                        available_info = f"\n사용 가능한 DINOv2 모델: {', '.join(dinov2_models[:10])}"
                    raise RuntimeError(
                        f"AprilGAN 모델 로드 실패: '{self.model_name}' 및 대체 모델들을 찾을 수 없습니다.{available_info}\n"
                        f"timm 버전을 확인하거나 'pip install --upgrade timm' 명령으로 업그레이드하세요."
                    )
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 모델 입력 크기 확인 (timm 모델의 default_cfg에서 확인)
            if hasattr(self.model, 'default_cfg'):
                # timm 모델의 입력 크기 확인
                input_size = self.model.default_cfg.get('input_size', None)
                if input_size:
                    if isinstance(input_size, (list, tuple)):
                        # (3, H, W) 형식이면 H, W 추출
                        if len(input_size) == 3:
                            self.input_size = input_size[1]  # H 또는 W (정사각형 가정)
                        else:
                            self.input_size = input_size[0] if len(input_size) > 0 else 224
                    else:
                        self.input_size = input_size
                else:
                    self.input_size = 224  # 기본값
            else:
                self.input_size = 224  # 기본값
            
            # 패치 크기 확인 (모델에 따라 다를 수 있음)
            if hasattr(self.model, 'patch_embed'):
                patch_size = self.model.patch_embed.patch_size
                if isinstance(patch_size, (list, tuple)):
                    self.patch_size = patch_size[0]
                else:
                    self.patch_size = patch_size
            else:
                self.patch_size = 14  # DINOv2 기본 패치 크기
            
            print(f"[AprilGAN] 제로샷 모델 준비 완료 (입력 크기: {self.input_size}x{self.input_size}, 패치 크기: {self.patch_size})")
            
        except RuntimeError:
            # RuntimeError는 그대로 전파
            raise
        except Exception as e:
            raise RuntimeError(f"AprilGAN 모델 로드 실패: {e}. 학습을 중단합니다.")
=======
        # TODO: 실제 AprilGAN 모델 로드
        # 예시: self.model = load_aprilgan_model(self.model_path)
        
        # 현재는 시뮬레이션용 더미 모델
        print("[AprilGAN] 제로샷 모델 로드 완료 (시뮬레이션 모드)")
        self.model = None
>>>>>>> parent of f73de4e (이것저것)
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        이미지에서 이상 영역 검출
        
        Args:
            image: 입력 이미지 (H, W, 3) RGB 형식
            
        Returns:
            {
                'anomaly_mask': np.ndarray,  # 이상 영역 이진 마스크
                'anomaly_regions': List[Dict],  # 이상 영역 바운딩박스 리스트
                'anomaly_score': float,  # 전체 이상 점수
                'confidence_map': np.ndarray  # 각 픽셀의 이상 확률
            }
        """
        if self.model is None:
            # 시뮬레이션 모드: 간단한 이상 탐지 시뮬레이션
            return self._simulate_detection(image)
        
        # 실제 모델 사용
        return self._real_detection(image)
    
    def _simulate_detection(self, image: np.ndarray) -> Dict:
        """
        시뮬레이션용 이상 탐지
        실제 프로젝트에서는 이 부분을 실제 AprilGAN 모델로 교체해야 합니다.
        """
        h, w = image.shape[:2]
        
        # 간단한 시뮬레이션: 이미지의 일부 영역을 이상으로 표시
        # 실제로는 AprilGAN 모델이 정교한 이상 탐지를 수행합니다
        
        # 예시: 이미지의 밝기나 텍스처 변화를 기반으로 이상 탐지 시뮬레이션
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
<<<<<<< HEAD
        Args:
            anomaly_scores: 패치별 이상 점수 배열 (N,)
            image_shape: 원본 이미지 크기 (height, width)
            
        Returns:
            이상 탐지 결과 딕셔너리
        """
        h, w = image_shape
        
        # 패치 수 계산 (실제 입력 크기에 따라 패치 수가 달라짐)
        input_size = getattr(self, 'input_size', 224)  # 모델의 실제 입력 크기 사용
        num_patches_h = input_size // self.patch_size
        num_patches_w = input_size // self.patch_size
        
        # 패치 점수를 2D 맵으로 변환
        if len(anomaly_scores) == num_patches_h * num_patches_w:
            anomaly_map_2d = anomaly_scores.reshape(num_patches_h, num_patches_w)
        else:
            # 패치 수가 맞지 않으면 1D로 처리
            num_patches = len(anomaly_scores)
            # 정사각형에 가까운 형태로 변환
            side = int(np.sqrt(num_patches))
            if side * side == num_patches:
                anomaly_map_2d = anomaly_scores.reshape(side, side)
            else:
                # 정사각형이 아니면 패딩
                side = int(np.ceil(np.sqrt(num_patches)))
                padded = np.zeros(side * side)
                padded[:num_patches] = anomaly_scores
                anomaly_map_2d = padded.reshape(side, side)
        
        # 원본 이미지 크기로 리사이즈
        anomaly_map_resized = cv2.resize(anomaly_map_2d, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # 이상 마스크 생성 (임계값 기반)
        threshold = np.percentile(anomaly_map_resized, 90)  # 상위 10%를 이상으로 판단
        anomaly_mask = (anomaly_map_resized > threshold).astype(np.uint8)
=======
        # 간단한 이상 탐지 시뮬레이션 (실제로는 복잡한 딥러닝 모델)
        # 여기서는 예시로 밝기 변화가 큰 영역을 이상으로 표시
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        diff = np.abs(gray.astype(float) - blur.astype(float))
        threshold = np.percentile(diff, 90)
        anomaly_mask = (diff > threshold).astype(np.uint8)
>>>>>>> parent of f73de4e (이것저것)
        
        # 연결된 컴포넌트 찾기
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            anomaly_mask, connectivity=8
        )
        
        # 바운딩박스 생성
        anomaly_regions = []
        for i in range(1, num_labels):  # 0은 배경
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 최소 크기 필터링
            if w * h < 100:  # 너무 작은 영역 제외
                continue
            
            anomaly_regions.append({
                'x1': x,
                'y1': y,
                'x2': x + w,
                'y2': y + h
            })
        
        # 이상 점수 계산
        anomaly_score = np.sum(anomaly_mask) / (h * w)
        
        return {
            'anomaly_mask': anomaly_mask,
            'anomaly_regions': anomaly_regions,
            'anomaly_score': float(anomaly_score),
            'confidence_map': diff / (diff.max() + 1e-8)  # 정규화
        }
    
    def _real_detection(self, image: np.ndarray) -> Dict:
        """
        실제 AprilGAN 모델을 사용한 이상 탐지
        실제 구현에서는 이 메서드를 사용합니다.
        """
        # TODO: 실제 AprilGAN 모델 추론 코드
        # 예시:
        # preprocessed = self._preprocess(image)
        # with torch.no_grad():
        #     result = self.model(preprocessed)
        # return self._postprocess(result)
        
        raise NotImplementedError("실제 AprilGAN 모델 구현 필요")
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
<<<<<<< HEAD
        """
        이미지 전처리 (DINOv2 입력 형식에 맞춤)
        모델의 입력 크기에 맞게 동적으로 리사이즈
        """
        # RGB로 변환 (이미 RGB면 그대로 사용)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 모델 입력 크기로 리사이즈 (동적으로 결정됨)
        input_size = getattr(self, 'input_size', 224)  # 기본값 224
        image_resized = cv2.resize(image_rgb, (input_size, input_size))
        
        # [0, 255] -> [0, 1]로 정규화
=======
        """이미지 전처리"""
        # 리사이즈, 정규화 등
        image_resized = cv2.resize(image, (512, 512))
>>>>>>> parent of f73de4e (이것저것)
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        return image_tensor.unsqueeze(0).to(self.device)
    
    def __call__(self, image: np.ndarray) -> Dict:
        """직접 호출 가능하도록"""
        return self.detect(image)

