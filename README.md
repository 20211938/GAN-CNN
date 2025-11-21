# 금속 3D 프린팅 결함 검출 및 분류 프로젝트

## 📋 프로젝트 개요

금속 3D 프린팅 공정 중 발생하는 결함을 **AprilGAN + CNN** 구조를 활용하여 검출하고 분류하는 프로젝트입니다.

## 🚀 시작하기

### 필수 요구사항

- **Python**: 3.8 이상 (권장: 3.9 이상)
- **pip**: Python 패키지 관리자
- **NVIDIA GPU**: CUDA를 지원하는 NVIDIA GPU
- **CUDA Toolkit**: GPU 가속을 위한 CUDA Toolkit (필수: CUDA 12.8)

### 설치 방법

자세한 설치 가이드는 [INSTALLATION.md](INSTALLATION.md)를 참고하세요.

1. **CUDA Toolkit 12.8 설치** (필수)
   - [NVIDIA CUDA Toolkit 다운로드](https://developer.nvidia.com/cuda-downloads)
   - 필수 버전: CUDA 12.8

2. **가상 환경 생성 및 활성화**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **패키지 설치**
   ```powershell
   pip install -r requirements.txt
   ```

4. **PyTorch CUDA 12.8 버전 설치**
   ```powershell
   # CUDA 12.8 직접 지원 버전
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

5. **환경 변수 설정**
   - 프로젝트 루트에 `.env` 파일 생성
   - MongoDB 연결 정보 입력 (자세한 내용은 [INSTALLATION.md](INSTALLATION.md) 참고)

자세한 설치 가이드는 [INSTALLATION.md](INSTALLATION.md)를 참고하세요.

## 🎯 목표

- **결함 검출**: AprilGAN을 통한 합성 데이터 생성 및 결함 탐지
- **결함 분류**: CNN을 통한 결함 클래스 분류
- 제한된 데이터 환경에서의 성능 개선 및 실용성 확보

## 🏗️ 모델 구조

### AprilGAN + CNN 파이프라인

```
AprilGAN → 합성 데이터 생성/결함 탐지 → CNN → 결함 클래스 분류
```

- **AprilGAN**: 데이터 부족 문제 해결을 위한 합성 데이터 생성 및 결함 탐지
- **CNN**: 결함 유형별 분류 성능 극대화


## 🔒 데이터 프라이버시 중심 학습 전략 핵심 요약

본 프로젝트는 금속 3D 프린팅 결함 검출 환경에서 발생하는 **데이터 부족 문제**와  
여러 기관·클라이언트 간의 **데이터 프라이버시 요구사항**을 동시에 해결하기 위한  
프라이버시 중심 학습 전략을 기반으로 합니다.

### 📌 핵심 원칙

- **GAN은 클라이언트 내부(Local)에서만 학습**하여 민감 데이터가 외부로 유출되지 않습니다.
- **GAN이 생성한 합성 데이터 또한 공유되지 않으며**, 오직 로컬 학습에만 사용됩니다.
- **CNN 분류 모델은 Federated Learning(FedAvg) 방식으로 통합 학습**됩니다.
- 중앙 서버는 **이미지가 아닌, 모델의 가중치 변화(ΔW)만 수신**합니다.
- 모든 클라이언트의 **원본 데이터는 절대 서버로 전달되지 않습니다.**
- Federated Learning은 **네트워크 구조(CNN, U-Net, GAN)에 독립적**이므로 다양한 모델 적용이 가능합니다.
- 전체 과정은 아래의 **파이프라인 이미지**를 통해 시각적으로 정리됩니다.
- 이러한 조합은 **데이터 부족 + 프라이버시 보호 + 결함 검출 성능 향상**을 동시에 달성하기 위한 최적의 구조입니다.

### 구조 변경 이유

1. **데이터 제약**: 제한된 실제 데이터셋으로 인한 U-Net 기반 분할 모델의 한계
2. **효율성**: 분류 중심 과제에 적합한 구조로 재설계
3. **확장성**: GAN 기반 데이터 증강을 통한 모델 일반화 능력 향상

## 🗂️ 프로젝트 구조

프로젝트는 다음과 같이 4부분으로 구성됩니다:

1. **데이터 처리 (Data Processing)**: 데이터 가져오기 및 전처리
2. **AprilGAN (Defect Detection)**: 결함 검출 및 합성 데이터 생성
3. **CNN (Defect Classification)**: 결함 유형별 분류
4. **연합 학습 (Federated Learning)**: 서버-클라이언트 구조 구현
   - 각 모델(AprilGAN, CNN)을 독립적으로 연합 학습 가능
   - 필요 시 각 파일로 분리 구현

```
데이터 처리 → AprilGAN (검출) → CNN (분류) → 연합 학습 (서버-클라이언트)
```

## 📊 데이터셋

- 금속 3D 프린팅 공정 결함 이미지
- 결함 유형별 샘플 불균형 문제 해결을 위한 AprilGAN 기반 증강


## 📈 기대 효과

- **데이터 확장**: GAN 기반 합성 데이터로 훈련 세트 대폭 확장
- **성능 개선**: CNN 분류 모델의 정확도 향상
- **불균형 해소**: 결함 유형별 데이터 불균형 문제 완화

## 📝 참고문헌

- 논문 2: GAN 기반 합성 데이터 생성 방법론
- 베이스라인: U-Net 기반 연합 학습 결함 분할

---

---

## 📚 문서

- [설치 가이드](INSTALLATION.md) - 환경 설정 및 패키지 설치 방법
- [요구사항](requirements.txt) - 필수 패키지 목록

---

**Last Updated**: 2025년
