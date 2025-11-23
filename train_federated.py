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
from utils.logger import create_logger
from utils.checkpoint import create_checkpoint_manager
from utils.metrics import evaluate_model, print_per_class_metrics
from pathlib import Path
import numpy as np


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
        default=5,
        help='클라이언트 수 (기본값: 5)'
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
    
    # 로깅 옵션
    parser.add_argument(
        '--log-dir',
        type=Path,
        default=Path('logs'),
        help='로그 저장 디렉토리 (기본값: logs)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='실험 이름 (기본값: 타임스탬프)'
    )
    parser.add_argument(
        '--no-log',
        action='store_true',
        help='로그 저장 비활성화'
    )
    
    # 체크포인트 옵션
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=Path('checkpoints'),
        help='체크포인트 저장 디렉토리 (기본값: checkpoints)'
    )
    parser.add_argument(
        '--save-checkpoints',
        action='store_true',
        default=True,
        help='체크포인트 저장 활성화 (기본값: True)'
    )
    parser.add_argument(
        '--no-save-checkpoints',
        action='store_false',
        dest='save_checkpoints',
        help='체크포인트 저장 비활성화'
    )
    parser.add_argument(
        '--resume-from',
        type=Path,
        default=None,
        help='체크포인트 파일 경로 (학습 재개용)'
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
    print(f"  ├─ 퓨샷 학습: {'사용' if args.use_few_shot else '미사용'}")
    print(f"  ├─ 로그 저장: {'비활성화' if args.no_log else f'{args.log_dir}'}")
    print(f"  └─ 체크포인트: {'비활성화' if not args.save_checkpoints else f'{args.checkpoint_dir}'}")
    print(f"{'='*70}\n")
    
    # 실험 설정 딕셔너리 생성 (로거와 체크포인트 매니저에서 공유)
    config = {
        'data_dir': str(args.data_dir),
        'num_clients': args.num_clients,
        'min_clients': args.min_clients,
        'non_iid_alpha': args.non_iid_alpha,
        'num_rounds': args.num_rounds,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'train_ratio': args.train_ratio,
        'backbone': args.backbone,
        'device': args.device,
        'use_few_shot': args.use_few_shot,
        'server_port': args.server_port
    }
    
    # 로거 초기화
    logger = None
    if not args.no_log:
        try:
            logger = create_logger(
                log_dir=args.log_dir,
                experiment_name=args.experiment_name
            )
            logger.log_config(config)
        except Exception as e:
            print(f"  ⚠️  로거 초기화 실패: {e}")
            print("  💡 로그 없이 계속 진행합니다.")
            import traceback
            traceback.print_exc()
            logger = None
    
    # 체크포인트 매니저 초기화
    checkpoint_manager = None
    if args.save_checkpoints:
        try:
            checkpoint_manager = create_checkpoint_manager(
                checkpoint_dir=args.checkpoint_dir,
                experiment_name=args.experiment_name,
                save_best=True,
                save_latest=True,
                save_rounds=True
            )
            print(f"[체크포인트] 체크포인트 매니저 초기화 완료")
        except Exception as e:
            print(f"  ⚠️  체크포인트 매니저 초기화 실패: {e}")
            print("  💡 체크포인트 없이 계속 진행합니다.")
            import traceback
            traceback.print_exc()
            checkpoint_manager = None
    
    # 1. AprilGAN 모델 초기화
    print("[1단계] AprilGAN 모델 초기화 중...")
    aprilgan = AprilGAN()
    print("  └─ 완료!\n")
    
    # 2. 데이터 로드
    print("[2단계] Non-IID 데이터 로드 중...")
    try:
        train_loaders, val_loaders, test_loader, defect_type_to_idx = load_client_data(
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
        
        # 클라이언트별 분포를 로거에 기록
        if logger is not None:
            try:
                from utils.bbox_utils import extract_bboxes_from_json, normalize_defect_type
                client_distributions = {}
                
                # load_client_data에서 반환된 client_data를 사용할 수 없으므로
                # 데이터셋에서 직접 샘플 수만 기록
                for client_id in range(args.num_clients):
                    train_dataset = train_loaders[client_id].dataset
                    val_dataset = val_loaders[client_id].dataset
                    
                    total_samples = len(train_dataset) + len(val_dataset)
                    train_samples = len(train_dataset)
                    val_samples = len(val_dataset)
                    
                    # 결함 유형 통계는 간단하게 샘플 수만 기록
                    # (정확한 분포는 이미 analyze_client_distribution에서 출력됨)
                    client_distributions[client_id] = {
                        'total_samples': total_samples,
                        'train_samples': train_samples,
                        'val_samples': val_samples,
                        'defect_distribution': {}  # 나중에 필요시 추가 가능
                    }
                
                logger.log_client_distribution(client_distributions)
            except Exception as e:
                print(f"  ⚠️  클라이언트 분포 로깅 실패: {e}")
                import traceback
                traceback.print_exc()
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
    
    # 체크포인트에서 재개 (있는 경우)
    start_round = 0
    if args.resume_from is not None and checkpoint_manager is not None:
        try:
            checkpoint = checkpoint_manager.load_checkpoint(
                args.resume_from,
                cnn_model,
                device=torch.device(args.device)
            )
            start_round = checkpoint.get('round', 0) + 1
            print(f"[체크포인트] ✅ 학습 재개: 라운드 {start_round}부터 시작")
        except Exception as e:
            print(f"  ⚠️  체크포인트 로드 실패: {e}")
            print("  💡 처음부터 학습을 시작합니다.")
            start_round = 0
    
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
    
    for round_num in range(start_round, args.num_rounds):
        print(f"\n{'='*70}")
        print(f"라운드 {round_num + 1}/{args.num_rounds}")
        print(f"{'='*70}")
        
        # 6-1. 클라이언트가 서버에서 가중치 수신
        print(f"\n[라운드 {round_num + 1}] 1단계: 가중치 수신")
        for client in clients:
            client.fetch_aggregated_weights(round_num)
        
        # 6-2. 각 클라이언트가 로컬 데이터로 학습
        print(f"\n[라운드 {round_num + 1}] 2단계: 로컬 학습")
        client_stats_list = []
        
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
            
            # 클라이언트 통계 저장
            client_stat = {
                'client_id': client.client_id,
                'loss': stats.get('loss', 0.0),
                'accuracy': stats.get('accuracy', 0.0),
                'samples': stats.get('samples', 0),
                'data_size': len(train_loaders[client.client_id].dataset)
            }
            client_stats_list.append(client_stat)
        
        # 6-3. 각 클라이언트가 가중치를 서버로 전송
        print(f"\n[라운드 {round_num + 1}] 3단계: 가중치 업로드")
        for client in clients:
            data_size = len(train_loaders[client.client_id].dataset)
            client.upload_weights(round_num, data_size)
        
        # 6-4. 서버가 가중치 집계 (자동으로 수행됨)
        print(f"\n[라운드 {round_num + 1}] 4단계: 가중치 집계")
        time.sleep(1)  # 서버 처리 대기
        
        aggregated_weights = server.get_aggregated_weights()
        server_stats = None
        if aggregated_weights is not None:
            print(f"  ✅ 가중치 집계 완료 (라운드 {server.current_round})")
            server_stats = {
                'round': server.current_round,
                'aggregated': True,
                'num_clients': len(client_stats_list)
            }
        else:
            print(f"  ⚠️  아직 집계되지 않음")
            server_stats = {
                'round': round_num,
                'aggregated': False,
                'num_clients': len(client_stats_list)
            }
        
        # 라운드 로그 기록
        if logger is not None:
            logger.log_round(round_num + 1, client_stats_list, server_stats)
        
        # 체크포인트 저장
        if checkpoint_manager is not None and aggregated_weights is not None:
            # 집계된 가중치로 모델 업데이트
            cnn_model.load_state_dict(aggregated_weights)
            
            # 평균 성능 계산
            avg_accuracy = sum(c.get('accuracy', 0) for c in client_stats_list) / len(client_stats_list) if client_stats_list else 0.0
            avg_loss = sum(c.get('loss', 0) for c in client_stats_list) / len(client_stats_list) if client_stats_list else 0.0
            
            # 최고 성능 확인
            is_best = checkpoint_manager.update_best(avg_accuracy, round_num + 1)
            
            # 체크포인트 저장
            checkpoint_manager.save_checkpoint(
                model=cnn_model,
                round_num=round_num + 1,
                metrics={
                    'accuracy': avg_accuracy,
                    'loss': avg_loss,
                    'num_clients': len(client_stats_list)
                },
                config=config,
                is_best=is_best
            )
        
        print(f"\n라운드 {round_num + 1} 완료!")
        print(f"{'='*70}\n")
    
    # 7. 최종 평가 (테스트 데이터셋 사용)
    print(f"\n{'='*70}")
    print(f"[7단계] 최종 모델 평가 (테스트 데이터셋)")
    print(f"{'='*70}\n")
    
    # 최종 가중치로 모델 업데이트
    final_weights = server.get_aggregated_weights()
    if final_weights is not None:
        cnn_model.load_state_dict(final_weights)
        print("✅ 최종 집계된 가중치로 모델 업데이트 완료\n")
        
        # 테스트 데이터셋으로 평가
        if test_loader is not None:
            # 클래스 이름 리스트 생성 (idx_to_defect_type)
            idx_to_defect_type = {idx: defect_type for defect_type, idx in defect_type_to_idx.items()}
            class_names = [idx_to_defect_type.get(i, f"Class_{i}") for i in range(num_classes)]
            
            # 클래스별 성능 평가
            test_metrics = evaluate_model(
                model=cnn_model,
                data_loader=test_loader,
                device=torch.device(args.device),
                num_classes=num_classes,
                class_names=class_names
            )
            
            # 클래스별 성능 출력
            print_per_class_metrics(test_metrics, "서버 모델 최종 성능 평가 (테스트 데이터셋)")
            
            # 최종 결과를 로거에 기록
            if logger is not None:
                final_results = {
                    'test_metrics': test_metrics,
                    'class_names': class_names,
                    'test_samples': test_metrics['total_samples']
                }
                
                logger.log_final_results(final_results)
                logger.save()
        else:
            print("  ⚠️  테스트 데이터셋이 없습니다. 검증 데이터로 평가합니다.")
            
            # 검증 데이터로 평가 (폴백)
            idx_to_defect_type = {idx: defect_type for defect_type, idx in defect_type_to_idx.items()}
            class_names = [idx_to_defect_type.get(i, f"Class_{i}") for i in range(num_classes)]
            
            # 모든 검증 데이터 합치기
            from torch.utils.data import ConcatDataset, DataLoader
            all_val_datasets = [loader.dataset for loader in val_loaders]
            combined_val_dataset = ConcatDataset(all_val_datasets)
            combined_val_loader = DataLoader(
                combined_val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0
            )
            
            val_metrics = evaluate_model(
                model=cnn_model,
                data_loader=combined_val_loader,
                device=torch.device(args.device),
                num_classes=num_classes,
                class_names=class_names
            )
            
            print_per_class_metrics(val_metrics, "서버 모델 최종 성능 평가 (검증 데이터셋)")
            
            if logger is not None:
                final_results = {
                    'val_metrics': val_metrics,
                    'class_names': class_names,
                    'val_samples': val_metrics['total_samples']
                }
                
                logger.log_final_results(final_results)
                logger.save()
    else:
        print("  ⚠️  최종 가중치를 가져올 수 없습니다")
        if logger is not None:
            logger.log_final_results({'error': '최종 가중치를 가져올 수 없음'})
            logger.save()
    
    print(f"\n{'='*70}")
    print(f"연합학습 완료!")
    if logger is not None:
        print(f"로그 저장 위치: {logger.get_log_path()}")
    if checkpoint_manager is not None:
        print(f"체크포인트 저장 위치: {checkpoint_manager.get_checkpoint_dir()}")
        print(f"  - 최고 성능: {checkpoint_manager.best_accuracy:.4f} (라운드 {checkpoint_manager.best_round})")
    print(f"{'='*70}\n")
    
    # 서버 종료
    print("서버 종료 중...")
    # 서버는 데몬 스레드이므로 자동으로 종료됩니다


if __name__ == '__main__':
    main()

