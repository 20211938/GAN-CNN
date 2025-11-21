"""
연합학습 실행 스크립트
Non-IID 데이터 분배를 포함한 전체 연합학습 파이프라인 실행
"""

import argparse
import time
import torch
from pathlib import Path
from threading import Thread

from models.aprilgan import AprilGAN
from models.cnn import create_cnn_model
from federated.server import FederatedServer
from federated.client import FederatedClient
from utils.client_data_loader import load_client_data


def main():
    parser = argparse.ArgumentParser(
        description='금속 3D 프린팅 결함 검출 연합학습 실행',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 설정으로 실행 (3개 클라이언트, 3 라운드)
  python train_federated.py --data-dir data

  # Non-IID 정도 조절 (매우 편향)
  python train_federated.py --data-dir data --non-iid-alpha 0.1

  # 더 많은 라운드와 에폭
  python train_federated.py --data-dir data --num-rounds 10 --epochs 3
        """
    )
    
    # 데이터 관련
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='데이터 디렉토리 경로 (기본값: data)'
    )
    
    # 클라이언트 및 서버 설정
    parser.add_argument(
        '--num-clients',
        type=int,
        default=3,
        help='클라이언트 수 (기본값: 3)'
    )
    parser.add_argument(
        '--min-clients',
        type=int,
        default=None,
        help='최소 클라이언트 수 (기본값: num-clients)'
    )
    parser.add_argument(
        '--server-port',
        type=int,
        default=5000,
        help='서버 포트 (기본값: 5000)'
    )
    
    # Non-IID 설정
    parser.add_argument(
        '--non-iid-alpha',
        type=float,
        default=0.5,
        help='Non-IID 정도 (0.1: 매우 편향, 0.5: 보통, 10.0: 균등) (기본값: 0.5)'
    )
    
    # 학습 설정
    parser.add_argument(
        '--num-rounds',
        type=int,
        default=3,
        help='연합학습 라운드 수 (기본값: 3)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='각 라운드당 로컬 학습 에폭 수 (기본값: 1)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='학습률 (기본값: 0.001)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='배치 크기 (기본값: 32)'
    )
    
    # 데이터 분할
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='학습 데이터 비율 (기본값: 0.8)'
    )
    
    # 모델 설정
    parser.add_argument(
        '--backbone',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet34', 'resnet50'],
        help='CNN 백본 모델 (기본값: resnet18)'
    )
    
    # 기타
    parser.add_argument(
        '--use-few-shot',
        action='store_true',
        help='퓨샷 학습 모드 사용'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='사용할 디바이스 (기본값: 자동 감지)'
    )
    
    args = parser.parse_args()
    
    # 최소 클라이언트 수 설정
    if args.min_clients is None:
        args.min_clients = args.num_clients
    
    # 디바이스 설정
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*70}")
    print(f"연합학습 시작")
    print(f"{'='*70}")
    print(f"  ├─ 데이터 디렉토리: {args.data_dir}")
    print(f"  ├─ 클라이언트 수: {args.num_clients}개")
    print(f"  ├─ 최소 클라이언트 수: {args.min_clients}개")
    print(f"  ├─ Non-IID 정도 (alpha): {args.non_iid_alpha}")
    print(f"  ├─ 연합학습 라운드: {args.num_rounds}개")
    print(f"  ├─ 로컬 학습 에폭: {args.epochs}개")
    print(f"  ├─ 학습률: {args.learning_rate}")
    print(f"  ├─ 배치 크기: {args.batch_size}")
    print(f"  ├─ 백본 모델: {args.backbone}")
    print(f"  ├─ 디바이스: {args.device}")
    print(f"  └─ 퓨샷 학습: {'사용' if args.use_few_shot else '미사용'}")
    print(f"{'='*70}\n")
    
    # 1. AprilGAN 모델 초기화
    print("[1단계] AprilGAN 모델 초기화 중...")
    aprilgan = AprilGAN()
    print("  └─ 완료!\n")
    
    # 2. 데이터 로드
    print("[2단계] Non-IID 데이터 로드 중...")
    try:
        train_loaders, val_loaders, defect_type_to_idx = load_client_data(
            data_dir=args.data_dir,
            aprilgan_model=aprilgan,
            num_clients=args.num_clients,
            train_ratio=args.train_ratio,
            batch_size=args.batch_size,
            patch_size=(224, 224),
            non_iid_alpha=args.non_iid_alpha,
            verbose=True
        )
        num_classes = len(defect_type_to_idx)
    except Exception as e:
        print(f"  ❌ 데이터 로드 실패: {e}")
        print("  💡 데이터 디렉토리를 확인하거나 --data-dir 옵션을 확인하세요.")
        return
    
    # 3. CNN 모델 생성
    print(f"\n[3단계] CNN 모델 생성 중...")
    cnn_model = create_cnn_model(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=True
    )
    print(f"  └─ 완료! (클래스 수: {num_classes})\n")
    
    # 4. 서버 시작
    print(f"[4단계] 연합학습 서버 시작 중...")
    server = FederatedServer(
        port=args.server_port,
        num_clients=args.num_clients,
        min_clients=args.min_clients
    )
    
    # 초기 가중치 설정
    initial_weights = cnn_model.state_dict()
    server.set_initial_weights(initial_weights)
    
    # 서버를 별도 스레드에서 실행
    server_thread = Thread(target=server.start, daemon=True)
    server_thread.start()
    
    # 서버 시작 대기
    time.sleep(2)
    print(f"  └─ 서버 시작 완료! (포트: {args.server_port})\n")
    
    # 5. 클라이언트 생성
    print(f"[5단계] 클라이언트 생성 중...")
    clients = []
    server_url = f'http://localhost:{args.server_port}'
    
    for client_id in range(args.num_clients):
        client = FederatedClient(
            client_id=client_id,
            server_url=server_url,
            model=cnn_model,
            device=args.device
        )
        clients.append(client)
        print(f"  ├─ 클라이언트 {client_id} 생성 완료")
    print(f"  └─ 총 {len(clients)}개 클라이언트 생성 완료\n")
    
    # 6. 연합학습 라운드 실행
    print(f"[6단계] 연합학습 라운드 실행")
    print(f"{'='*70}\n")
    
    for round_num in range(args.num_rounds):
        print(f"\n{'='*70}")
        print(f"라운드 {round_num + 1}/{args.num_rounds}")
        print(f"{'='*70}")
        
        # 6-1. 클라이언트가 서버에서 가중치 수신
        print(f"\n[라운드 {round_num + 1}] 1단계: 가중치 수신")
        for client in clients:
            client.fetch_aggregated_weights(round_num)
        
        # 6-2. 각 클라이언트가 로컬 데이터로 학습
        print(f"\n[라운드 {round_num + 1}] 2단계: 로컬 학습")
        for client in clients:
            client_train_loader = train_loaders[client.client_id]
            
            if args.use_few_shot:
                # 퓨샷 학습 모드 (추후 구현 필요)
                print(f"  클라이언트 {client.client_id}: 퓨샷 학습 모드 (미구현)")
                stats = client.train_local(
                    client_train_loader,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate
                )
            else:
                # 일반 학습 모드
                stats = client.train_local(
                    client_train_loader,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate
                )
        
        # 6-3. 각 클라이언트가 가중치를 서버로 전송
        print(f"\n[라운드 {round_num + 1}] 3단계: 가중치 업로드")
        for client in clients:
            data_size = len(train_loaders[client.client_id].dataset)
            client.upload_weights(round_num, data_size)
        
        # 6-4. 서버가 가중치 집계 (자동으로 수행됨)
        print(f"\n[라운드 {round_num + 1}] 4단계: 가중치 집계")
        time.sleep(1)  # 서버 처리 대기
        
        aggregated_weights = server.get_aggregated_weights()
        if aggregated_weights is not None:
            print(f"  ✅ 가중치 집계 완료 (라운드 {server.current_round})")
        else:
            print(f"  ⚠️  아직 집계되지 않음")
        
        print(f"\n라운드 {round_num + 1} 완료!")
        print(f"{'='*70}\n")
    
    # 7. 최종 평가
    print(f"\n{'='*70}")
    print(f"[7단계] 최종 모델 평가")
    print(f"{'='*70}\n")
    
    # 최종 가중치로 모델 업데이트
    final_weights = server.get_aggregated_weights()
    if final_weights is not None:
        cnn_model.load_state_dict(final_weights)
        print("✅ 최종 집계된 가중치로 모델 업데이트 완료\n")
        
        # 모든 클라이언트의 검증 데이터로 평가
        cnn_model.eval()
        total_correct = 0
        total_samples = 0
        
        for client_id, val_loader in enumerate(val_loaders):
            client_correct = 0
            client_samples = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(args.device)
                    labels = batch['label'].to(args.device)
                    
                    outputs = cnn_model(images)
                    _, predicted = torch.max(outputs, 1)
                    
                    batch_size = labels.size(0)
                    client_samples += batch_size
                    client_correct += (predicted == labels).sum().item()
            
            client_accuracy = client_correct / client_samples if client_samples > 0 else 0.0
            print(f"  클라이언트 {client_id}: {client_accuracy:.4f} ({client_correct}/{client_samples})")
            
            total_samples += client_samples
            total_correct += client_correct
        
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        print(f"\n  전체 모델 정확도: {overall_accuracy:.4f} ({total_correct}/{total_samples})")
    else:
        print("  ⚠️  최종 가중치를 가져올 수 없습니다")
    
    print(f"\n{'='*70}")
    print(f"연합학습 완료!")
    print(f"{'='*70}\n")
    
    # 서버 종료
    print("서버 종료 중...")
    # 서버는 데몬 스레드이므로 자동으로 종료됩니다


if __name__ == '__main__':
    main()

