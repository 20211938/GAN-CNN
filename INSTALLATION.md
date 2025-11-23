# 설치 가이드

이 문서는 GAN-CNN 프로젝트의 환경 설정 및 패키지 설치 방법을 안내합니다.

**프로젝트 개요:**
- **AprilGAN**: DINOv2 Vision Transformer 기반 제로샷 이상 탐지 모델
- **CNN**: ResNet 기반 결함 유형 분류 모델 (연합학습)
- **연합학습**: 여러 클라이언트가 가중치만 공유하여 협력 학습

## 📋 목차

1. [필수 요구사항](#필수-요구사항)
2. [CUDA 설치](#cuda-설치)
3. [가상 환경 설정 및 패키지 설치](#가상-환경-설정-및-패키지-설치)
4. [환경 변수 설정](#환경-변수-설정)
5. [설치 확인](#설치-확인)

---

## 필수 요구사항

- **Python**: 3.8 이상 (권장: 3.9 이상, 최대: 3.11)
  - PyTorch 2.0+ 호환성을 위해 Python 3.9 이상을 권장합니다
- **pip**: Python 패키지 관리자 (일반적으로 Python과 함께 설치됨)
- **NVIDIA GPU**: CUDA를 지원하는 NVIDIA GPU가 필요합니다
- **CUDA Toolkit**: GPU 가속을 위한 CUDA Toolkit 설치가 필요합니다
  - 필수 버전: CUDA 12.8

### Python 버전 확인

```powershell
python --version
```

### CUDA 설치 확인

시스템에 CUDA 12.8이 설치되어 있는지 확인합니다:

```powershell
nvcc --version
```

CUDA 12.8이 설치되어 있지 않은 경우, 다음 단계를 따라 CUDA Toolkit 12.8을 설치합니다.

---

## CUDA 12.8 설치

### 1단계: CUDA 12.8 설치 확인

시스템에 CUDA 12.8이 이미 설치되어 있는지 확인합니다:

```powershell
nvcc --version
```

CUDA 12.8 버전이 표시되면 다음 섹션(가상 환경 설정)으로 진행하세요.

CUDA 12.8이 설치되어 있지 않은 경우, 아래 단계를 따라 설치합니다.

### 2단계: NVIDIA GPU 확인

시스템에 NVIDIA GPU가 설치되어 있는지 확인합니다:

```powershell
nvidia-smi
```

GPU 정보가 표시되면 다음 단계로 진행합니다.

### 3단계: CUDA Toolkit 12.8 다운로드

1. [NVIDIA CUDA Toolkit 다운로드 페이지](https://developer.nvidia.com/cuda-downloads)에 접속합니다.
2. 운영체제를 선택합니다 (Windows).
3. 아키텍처를 선택합니다 (x86_64).
4. 버전을 선택합니다 (CUDA 12.8).
5. 설치 유형을 선택합니다 (exe [local] 권장).
6. 다운로드 버튼을 클릭하여 설치 파일을 다운로드합니다.

### 4단계: CUDA Toolkit 12.8 설치

1. 다운로드한 설치 파일을 실행합니다.
2. 설치 마법사의 지시를 따릅니다.
3. 기본 설치 경로를 사용하는 것을 권장합니다.
4. 설치가 완료되면 시스템을 재시작합니다.

### 5단계: CUDA 12.8 설치 확인

시스템 재시작 후, 다음 명령어로 CUDA 설치를 확인합니다:

```powershell
nvcc --version
```

## 가상 환경 설정 및 패키지 설치

프로젝트 루트 디렉토리에서 다음 단계를 순서대로 실행하세요.

### Windows (PowerShell)

#### 1단계: 가상 환경 생성

```powershell
python -m venv venv
```

#### 2단계: 가상 환경 활성화

```powershell
.\venv\Scripts\Activate.ps1
```

**참고**: PowerShell 실행 정책 오류가 발생하는 경우, 다음 명령어를 먼저 실행하세요:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 3단계: pip 업그레이드

```powershell
python -m pip install --upgrade pip
```



#### 4단계: PyTorch CUDA 12.8 버전 설치

기본 패키지 설치 후, CUDA 12.8에 최적화된 PyTorch를 설치합니다.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

#### 5단계: 기본 패키지 설치

```powershell
pip install -r requirements.txt
```

#### 6단계: Vision Transformer 모델 패키지 설치 (DINOv2용)

AprilGAN 제로샷 모델이 DINOv2 Vision Transformer를 사용하므로 추가 패키지가 필요합니다:

```powershell
pip install timm
```


### 가상 환경 비활성화

작업이 끝나면 가상 환경을 비활성화합니다:

```powershell
deactivate
```

## 환경 변수 설정

MongoDB 연결 정보를 `.env` 파일에 설정합니다.

### 1단계: .env 파일 생성

프로젝트 루트 디렉토리에서 실행:

**Windows (PowerShell)**:
```powershell
New-Item -Path .env -ItemType File
```

**Windows (CMD)**:
```cmd
type nul > .env
```

### 2단계: .env 파일 내용 작성

생성된 `.env` 파일을 열고 다음 내용을 작성합니다:

```env
# MongoDB 연결 설정
MONGODB_HOST=localhost
MONGODB_PORT=50002
MONGODB_USER=your_username
MONGODB_PASSWORD=your_password
MONGODB_AUTH_DB=admin
```

**중요**: `your_username`, `your_password` 등을 실제 MongoDB 연결 정보로 변경하세요.

### 3단계: .env 파일 보안 확인

`.env` 파일은 민감한 정보를 포함하므로 Git에 커밋하지 않습니다. `.gitignore` 파일에 `.env`가 포함되어 있는지 확인하세요.

---

## 설치 확인

### 1단계: 패키지 설치 확인

```powershell
pip list
```

다음 주요 패키지가 설치되어 있는지 확인합니다:
- **데이터베이스**: pymongo, python-dotenv
- **머신러닝**: torch, torchvision
- **Vision Transformer**: timm (DINOv2용)
- **데이터 처리**: numpy, pandas
- **이미지 처리**: Pillow, opencv-python
- **시각화**: matplotlib, seaborn
- **유틸리티**: scikit-learn, scikit-image, tqdm

### 2단계: PyTorch 및 CUDA 설치 확인

```powershell
python -c "import torch; print('PyTorch 버전:', torch.__version__); print('CUDA 사용 가능:', torch.cuda.is_available()); print('CUDA 버전:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
```

정상적으로 설치된 경우 다음이 표시됩니다:
- `CUDA 사용 가능: True`
- `CUDA 버전: 12.8` (또는 설치한 PyTorch CUDA 버전)

**참고**: CUDA 12.8용 PyTorch를 설치했으므로, 시스템에 설치된 CUDA 12.8과 완벽하게 호환됩니다.

**중요**: `CUDA 사용 가능: False`가 표시되면 CUDA 설치 또는 PyTorch CUDA 버전 설치에 문제가 있는 것입니다.

### 3단계: 스크립트 실행 테스트

#### download_labeled_layers.py 테스트

```powershell
python utils/dataset/download_labeled_layers.py --help
```

```powershell
python utils/dataset/download_labeled_layers.py --dry-run
```

#### analyze_defect_types.py 테스트

```powershell
python utils/dataset/analyze_defect_types.py --help
```

#### cleanup_dataset.py 테스트

```powershell
python utils/dataset/cleanup_dataset.py --help
```

---

## 문제 해결

### 가상 환경이 활성화되지 않는 경우

**Windows PowerShell**에서 실행 정책 오류가 발생하는 경우:

```powershell
Get-ExecutionPolicy
```

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

그 후 다시 가상 환경을 활성화합니다:

```powershell
.\venv\Scripts\Activate.ps1
```

### 패키지 설치 오류

1. 인터넷 연결을 확인합니다.

2. pip를 업그레이드합니다:
   ```powershell
   python -m pip install --upgrade pip
   ```

3. 캐시를 클리어하고 재설치합니다:
   ```powershell
   pip cache purge
   pip install -r requirements.txt
   ```

### MongoDB 연결 오류

1. MongoDB 서버가 실행 중인지 확인합니다.

2. `.env` 파일의 연결 정보를 확인합니다.

3. 방화벽 설정을 확인합니다.

4. 명령줄에서 직접 연결 정보를 지정합니다:
   ```powershell
   python utils/dataset/download_labeled_layers.py --host localhost --port 50002 --username your_user --password your_password
   ```

### Python 버전 오류

Python 3.8 이상이 필요합니다 (권장: 3.9 이상). Python 버전을 확인합니다:

```powershell
python --version
```


### CUDA 설치 오류

1. **nvidia-smi가 작동하지 않는 경우**:
   - NVIDIA GPU 드라이버가 설치되어 있는지 확인합니다
   - [NVIDIA 드라이버 다운로드 페이지](https://www.nvidia.com/Download/index.aspx)에서 최신 드라이버를 설치합니다

2. **nvcc 명령어를 찾을 수 없는 경우**:
   - CUDA Toolkit 12.8이 제대로 설치되지 않았을 수 있습니다
   - 환경 변수 PATH에 CUDA bin 디렉토리가 추가되었는지 확인합니다
   - 일반 경로: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin`

3. **CUDA 버전 확인**:
   ```powershell
   nvcc --version
   ```
   CUDA 12.8 버전이 표시되어야 합니다.

### PyTorch CUDA 버전 설치 오류

1. **CPU 버전이 설치된 경우**:
   - `PyTorch 버전: 2.x.x+cpu`로 표시되면 CPU 버전이 설치된 것입니다
   - 다음 명령어로 CPU 버전을 제거하고 CUDA 버전을 설치합니다:
   ```powershell
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```

2. **CUDA 버전 확인**:
   - 시스템에 CUDA 12.8이 설치되어 있는지 확인합니다
   - `nvcc --version`으로 CUDA 12.8 설치를 확인하고, 위의 5단계에서 제공한 명령어를 사용합니다

3. **PyTorch 재설치**:
   ```powershell
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   ```
   (CUDA 12.8 직접 지원 버전 사용)

4. **설치 후 확인**:
   ```powershell
   python -c "import torch; print('PyTorch 버전:', torch.__version__); print('CUDA 사용 가능:', torch.cuda.is_available()); print('CUDA 버전:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"
   ```
   - `CUDA 사용 가능: True`가 표시되어야 합니다
   - `PyTorch 버전`에 `+cpu`가 아닌 `+cu128` 또는 `+cuda`가 포함되어야 합니다

3. **메모리 부족 오류**:
   - 가상 환경의 디스크 공간을 확인합니다
   - 필요시 다른 위치에 가상 환경을 생성합니다

---

## 다음 단계

설치가 완료되면 다음 문서를 참고하세요:

- **데이터 다운로드**: `utils/dataset/download_labeled_layers.py` 실행
- **데이터 분석**: `utils/dataset/analyze_defect_types.py` 실행
- **데이터 정리**: `utils/dataset/cleanup_dataset.py` 실행
- **연합학습 실행**: `python train_federated.py --data-dir data` 실행
- **프로젝트 개요**: `README.md` 참고

**중요 참고사항:**
- AprilGAN 모델은 DINOv2 Vision Transformer를 사용하므로 `timm` 패키지가 필수입니다
- 모델 경로를 지정하지 않으면 `timm`에서 사전 학습된 모델을 자동으로 다운로드합니다
- 실제 배포 시나리오를 반영하여 AprilGAN의 모든 검출 결과가 CNN 학습에 포함됩니다

---

## 추가 리소스

- [Python 가상 환경 공식 문서](https://docs.python.org/3/tutorial/venv.html)
- [pymongo 공식 문서](https://pymongo.readthedocs.io/)
- [python-dotenv 공식 문서](https://pypi.org/project/python-dotenv/)

---

**마지막 업데이트**: 2025년

