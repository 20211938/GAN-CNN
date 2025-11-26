"""
CLIP 기반 결함 탐지 모델
바운딩 박스와 텍스트 프롬프트를 사용한 지역-텍스트 정렬 기반 결함 탐지
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("[경고] CLIP 라이브러리가 설치되지 않았습니다. pip install git+https://github.com/openai/CLIP.git 명령으로 설치하세요.")


class CLIPDefectDetector:
    """
    CLIP 기반 결함 탐지 모델
    텍스트 프롬프트와 이미지 지역 간 유사도를 계산하여 결함 영역을 탐지합니다.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model_name: CLIP 모델 이름 (기본값: ViT-B/32)
                        옵션: RN50, RN101, RN50x4, RN50x16, RN50x64, 
                              ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px
            device: 사용할 디바이스 (None이면 자동 감지)
        """
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP 라이브러리가 필요합니다. 'pip install git+https://github.com/openai/CLIP.git' 명령으로 설치하세요.")
        
        self.model_name = model_name
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.preprocess = None
        self._load_model()
    
    def _load_model(self):
        """CLIP 모델 로드"""
        try:
            print(f"[CLIP] 모델 로드 중: {self.model_name}")
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            print(f"[CLIP] ✅ 모델 로드 완료")
        except Exception as e:
            raise RuntimeError(f"CLIP 모델 로드 실패: {e}")
    
    def _create_text_prompts(self, defect_type: str) -> List[str]:
        """
        결함 유형에 대한 텍스트 프롬프트 생성
        여러 템플릿을 사용하여 앙상블 효과를 얻습니다.
        
        Args:
            defect_type: 결함 유형 이름
            
        Returns:
            텍스트 프롬프트 리스트
        """
        # 기본 프롬프트 템플릿
        templates = [
            "{}",
            "a photo of {}",
            "a photo of {} defect",
            "an image of {}",
            "an image of {} defect",
            "a picture of {}",
            "a picture of {} defect",
            "{} in metal 3D printing",
            "{} defect in metal 3D printing",
        ]
        
        prompts = [template.format(defect_type) for template in templates]
        
        # "정상" 프롬프트 추가
        if defect_type.lower() != "normal":
            normal_prompts = [
                "normal surface",
                "a photo of normal surface",
                "normal metal surface",
                "defect-free surface",
            ]
            prompts.extend(normal_prompts)
        
        return prompts
    
    def detect(
        self,
        image: np.ndarray,
        defect_types: Optional[List[str]] = None,
        bboxes: Optional[List[Dict]] = None
    ) -> Dict:
        """
        이미지에서 결함 영역 검출
        
        Args:
            image: 입력 이미지 (H, W, 3) RGB 형식
            defect_types: 검출할 결함 유형 리스트 (None이면 일반적인 결함 유형 사용)
            bboxes: 바운딩 박스 리스트 (선택사항, 제공되면 해당 영역만 검출)
            
        Returns:
            {
                'anomaly_mask': np.ndarray,  # 이상 영역 이진 마스크
                'anomaly_regions': List[Dict],  # 이상 영역 바운딩박스 리스트
                'anomaly_score': float,  # 전체 이상 점수
                'confidence_map': np.ndarray,  # 각 픽셀의 이상 확률
                'defect_type_scores': Dict[str, float]  # 결함 유형별 점수
            }
        """
        if self.model is None:
            raise RuntimeError("CLIP 모델이 로드되지 않았습니다.")
        
        # 기본 결함 유형 (제공되지 않은 경우)
        if defect_types is None:
            defect_types = [
                "Recoater Streaking",
                "Super Elevation",
                "Crack",
                "Porosity",
                "Keyhole",
                "Spatter",
                "Lack of Fusion",
                "Normal"
            ]
        
        # 이미지 전처리
        image_rgb = image if len(image.shape) == 3 and image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]
        
        # PIL Image로 변환
        pil_image = Image.fromarray(image_rgb)
        
        # 이미지 임베딩 추출
        image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 이미지 특징 추출
            image_features = self.model.encode_image(image_tensor)
            image_features = F.normalize(image_features, dim=-1)
            
            # 텍스트 프롬프트 생성 및 인코딩
            all_prompts = []
            prompt_to_type = {}
            
            for defect_type in defect_types:
                prompts = self._create_text_prompts(defect_type)
                all_prompts.extend(prompts)
                prompt_to_type.update({prompt: defect_type for prompt in prompts})
            
            # 텍스트 특징 추출
            text_tokens = clip.tokenize(all_prompts, truncate=True).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            
            # 이미지-텍스트 유사도 계산
            similarity = (image_features @ text_features.T).cpu().numpy()[0]  # (num_prompts,)
            
            # 결함 유형별 최대 유사도 계산
            defect_type_scores = {}
            for defect_type in defect_types:
                type_prompts = [p for p, t in prompt_to_type.items() if t == defect_type]
                type_indices = [i for i, p in enumerate(all_prompts) if p in type_prompts]
                if type_indices:
                    defect_type_scores[defect_type] = float(np.max(similarity[type_indices]))
            
            # 정상이 아닌 결함 유형 중 최대 유사도
            defect_scores = {k: v for k, v in defect_type_scores.items() if k.lower() != "normal"}
            normal_score = defect_type_scores.get("Normal", 0.0)
            
            if defect_scores:
                max_defect_score = max(defect_scores.values())
                # 결함 점수 = 최대 결함 유사도 - 정상 유사도
                anomaly_score_global = max_defect_score - normal_score
            else:
                anomaly_score_global = 0.0
        
        # 바운딩 박스가 제공된 경우: 박스별 검출
        if bboxes is not None and len(bboxes) > 0:
            return self._detect_with_bboxes(image_rgb, bboxes, defect_types, defect_type_scores)
        
        # 바운딩 박스가 없는 경우: 전체 이미지 기반 검출
        return self._detect_without_bboxes(
            image_rgb,
            defect_types,
            defect_type_scores,
            anomaly_score_global
        )
    
    def _detect_with_bboxes(
        self,
        image: np.ndarray,
        bboxes: List[Dict],
        defect_types: List[str],
        defect_type_scores: Dict[str, float]
    ) -> Dict:
        """
        바운딩 박스가 제공된 경우: 박스별 결함 검출
        """
        h, w = image.shape[:2]
        
        # 각 박스에 대해 결함 유형 매칭
        anomaly_regions = []
        anomaly_mask = np.zeros((h, w), dtype=np.uint8)
        confidence_map = np.zeros((h, w), dtype=np.float32)
        
        for bbox in bboxes:
            x1 = max(0, bbox['x1'])
            y1 = max(0, bbox['y1'])
            x2 = min(w, bbox['x2'])
            y2 = min(h, bbox['y2'])
            
            # 박스 영역 추출
            patch = image[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            
            # 박스 영역에 대한 결함 점수 계산
            patch_pil = Image.fromarray(patch)
            patch_tensor = self.preprocess(patch_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                patch_features = self.model.encode_image(patch_tensor)
                patch_features = F.normalize(patch_features, dim=-1)
                
                # 텍스트 특징과 유사도 계산
                all_prompts = []
                for defect_type in defect_types:
                    all_prompts.extend(self._create_text_prompts(defect_type))
                
                text_tokens = clip.tokenize(all_prompts, truncate=True).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = F.normalize(text_features, dim=-1)
                
                similarity = (patch_features @ text_features.T).cpu().numpy()[0]
                
                # 결함 유형별 점수 계산
                max_score = float(np.max(similarity))
                normal_indices = [i for i, p in enumerate(all_prompts) if "normal" in p.lower()]
                normal_score = float(np.max(similarity[normal_indices])) if normal_indices else 0.0
                
                patch_anomaly_score = max_score - normal_score
            
            # 박스 영역을 마스크에 추가
            anomaly_mask[y1:y2, x1:x2] = 255
            confidence_map[y1:y2, x1:x2] = np.maximum(
                confidence_map[y1:y2, x1:x2],
                patch_anomaly_score
            )
            
            anomaly_regions.append({
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
                'score': patch_anomaly_score
            })
        
        # 전체 이상 점수 계산
        anomaly_score = np.sum(anomaly_mask > 0) / (h * w)
        
        return {
            'anomaly_mask': anomaly_mask,
            'anomaly_regions': anomaly_regions,
            'anomaly_score': float(anomaly_score),
            'confidence_map': confidence_map / (confidence_map.max() + 1e-8),
            'defect_type_scores': defect_type_scores
        }
    
    def _detect_without_bboxes(
        self,
        image: np.ndarray,
        defect_types: List[str],
        defect_type_scores: Dict[str, float],
        anomaly_score_global: float
    ) -> Dict:
        """
        바운딩 박스가 없는 경우: 전체 이미지 기반 결함 검출
        이미지를 패치로 나누어 각 패치의 결함 점수를 계산합니다.
        """
        h, w = image.shape[:2]
        
        # 이미지를 패치로 분할 (CLIP 입력 크기에 맞춤)
        patch_size = 224  # CLIP 기본 입력 크기
        stride = patch_size // 2  # 50% 오버랩
        
        anomaly_regions = []
        confidence_map = np.zeros((h, w), dtype=np.float32)
        
        # 패치별 결함 점수 계산
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(y + patch_size, h)
                x2 = min(x + patch_size, w)
                
                if y2 - y < patch_size // 2 or x2 - x < patch_size // 2:
                    continue
                
                patch = image[y:y2, x:x2]
                patch_resized = cv2.resize(patch, (patch_size, patch_size))
                
                patch_pil = Image.fromarray(patch_resized)
                patch_tensor = self.preprocess(patch_pil).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    patch_features = self.model.encode_image(patch_tensor)
                    patch_features = F.normalize(patch_features, dim=-1)
                    
                    # 텍스트 특징과 유사도 계산
                    all_prompts = []
                    for defect_type in defect_types:
                        all_prompts.extend(self._create_text_prompts(defect_type))
                    
                    text_tokens = clip.tokenize(all_prompts, truncate=True).to(self.device)
                    text_features = self.model.encode_text(text_tokens)
                    text_features = F.normalize(text_features, dim=-1)
                    
                    similarity = (patch_features @ text_features.T).cpu().numpy()[0]
                    
                    max_score = float(np.max(similarity))
                    normal_indices = [i for i, p in enumerate(all_prompts) if "normal" in p.lower()]
                    normal_score = float(np.max(similarity[normal_indices])) if normal_indices else 0.0
                    
                    patch_score = max_score - normal_score
                
                # 신뢰도 맵에 점수 추가 (가중 평균)
                confidence_map[y:y2, x:x2] = np.maximum(
                    confidence_map[y:y2, x:x2],
                    patch_score
                )
        
        # 임계값 기반 이상 마스크 생성
        threshold = np.percentile(confidence_map, 90)  # 상위 10%
        anomaly_mask = (confidence_map > threshold).astype(np.uint8) * 255
        
        # 연결된 컴포넌트로 바운딩 박스 생성
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            anomaly_mask, connectivity=8
        )
        
        for i in range(1, num_labels):  # 0은 배경
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w_box = stats[i, cv2.CC_STAT_WIDTH]
            h_box = stats[i, cv2.CC_STAT_HEIGHT]
            
            if w_box * h_box < 100:  # 최소 크기 필터링
                continue
            
            anomaly_regions.append({
                'x1': x,
                'y1': y,
                'x2': x + w_box,
                'y2': y + h_box,
                'score': float(np.mean(confidence_map[y:y+h_box, x:x+w_box]))
            })
        
        anomaly_score = np.sum(anomaly_mask > 0) / (h * w)
        
        return {
            'anomaly_mask': anomaly_mask,
            'anomaly_regions': anomaly_regions,
            'anomaly_score': float(anomaly_score),
            'confidence_map': confidence_map / (confidence_map.max() + 1e-8),
            'defect_type_scores': defect_type_scores
        }
    
    def __call__(self, image: np.ndarray, defect_types: Optional[List[str]] = None, bboxes: Optional[List[Dict]] = None) -> Dict:
        """직접 호출 가능하도록"""
        return self.detect(image, defect_types, bboxes)

