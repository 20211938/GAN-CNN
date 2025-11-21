"""
데이터셋 정리 및 삭제 스크립트
소수 클래스 및 의미 없는 이름을 가진 결함 유형 데이터를 삭제합니다.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def extract_defect_types_from_metadata(metadata: dict) -> List[str]:
    """JSON 메타데이터에서 결함 유형 추출"""
    defect_types = []
    
    # TagBoxes에서 결함 정보 추출
    if 'DepositionImageModel' in metadata:
        tag_boxes = metadata['DepositionImageModel'].get('TagBoxes', [])
        for tag in tag_boxes:
            name = tag.get('Name', '').strip()
            comment = tag.get('Comment', '').strip()
            if name:
                defect_type = comment if comment else name
                if defect_type and defect_type not in defect_types:
                    defect_types.append(defect_type)
    
    if 'ScanningImageModel' in metadata:
        tag_boxes = metadata['ScanningImageModel'].get('TagBoxes', [])
        for tag in tag_boxes:
            name = tag.get('Name', '').strip()
            comment = tag.get('Comment', '').strip()
            if name:
                defect_type = comment if comment else name
                if defect_type and defect_type not in defect_types:
                    defect_types.append(defect_type)
    
    return defect_types if defect_types else ["Normal"]


def is_meaningless_name(name: str) -> bool:
    """의미 없는 이름인지 확인 (숫자만 있거나 너무 짧은 경우)"""
    if not name or len(name.strip()) == 0:
        return True
    
    # 숫자만 있는 경우 (예: "3", "123")
    if re.match(r'^\d+$', name.strip()):
        return True
    
    # 너무 짧은 경우 (1-2자)
    if len(name.strip()) <= 2:
        return True
    
    # D1, D2 같은 패턴도 의미 없는 것으로 간주
    if re.match(r'^D\d+$', name.strip(), re.IGNORECASE):
        return True
    
    return False


def cleanup_dataset(
    data_dir: Path,
    min_ratio: float = 0.01,
    min_count: Optional[int] = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    다운로드된 데이터셋에서 소수 클래스 및 의미 없는 이름 제거
    
    Args:
        data_dir: 데이터 디렉토리 경로 (기본값: ./data)
        min_ratio: 최소 비율 (0.0-1.0). 이 비율 미만인 결함 유형은 삭제됨
        min_count: 최소 샘플 수. 이 값이 지정되면 min_ratio 대신 사용됨
        dry_run: True이면 실제 삭제하지 않고 미리보기만 수행
        verbose: True이면 상세한 진행 상황 출력
    
    Returns:
        정리 통계 딕셔너리 (total_files, deleted_files, removed_classes 등)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("데이터셋 정리 시작")
        print("=" * 60)
        
        print(f"\n[정리 기준]")
        if min_count is not None:
            print(f"  - 최소 샘플 수: {min_count}개")
        else:
            print(f"  - 최소 비율: {min_ratio*100:.2f}%")
        print(f"  - 의미 없는 이름 제거: 숫자만, D1/D2 패턴, 2자 이하")
        print(f"  - 모드: {'DRY RUN' if dry_run else '실제 삭제'}")
    
    # 1단계: 결함 유형 통계 수집
    if verbose:
        print(f"\n[1단계] 결함 유형 통계 수집 중...")
    defect_type_counts = Counter()
    files_with_defect_types = {}  # 파일 경로 -> 결함 유형 리스트
    
    data_path = Path(data_dir)
    if not data_path.exists():
        if verbose:
            print(f"[오류] 데이터 디렉토리가 없습니다: {data_dir}")
        return {}
    
    total_files = 0
    # data 폴더에 직접 저장된 이미지 파일들을 순회
    for img_file in data_path.glob("*.jpg"):
        # JSON 파일 경로: {filename}.jpg.json
        json_file = img_file.with_suffix(".jpg.json")
        if not json_file.exists():
            if verbose:
                print(f"  [경고] JSON 파일이 없습니다: {img_file.name}")
            continue
        
        total_files += 1
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            defect_types = extract_defect_types_from_metadata(metadata)
            files_with_defect_types[str(img_file)] = defect_types
            
            for defect_type in defect_types:
                defect_type_counts[defect_type] += 1
        except Exception as e:
            if verbose:
                print(f"  [경고] 파일 읽기 실패: {img_file.name} - {e}")
            continue
    
    if verbose:
        print(f"  - 검사한 파일 수: {total_files}")
        print(f"  - 발견된 결함 유형 수: {len(defect_type_counts)}")
    
    if total_files == 0:
        if verbose:
            print("\n[결과] 처리할 파일이 없습니다.")
        return {'total_files': 0, 'deleted_files': 0, 'removed_classes': 0}
    
    # 2단계: 삭제 대상 결정
    if verbose:
        print(f"\n[2단계] 삭제 대상 결정 중...")
    
    # 비율 또는 개수 기반 필터링
    if min_count is not None:
        threshold_count = min_count
    else:
        threshold_count = max(1, int(total_files * min_ratio))
    
    minor_classes = set()
    for defect_type, count in defect_type_counts.items():
        if count < threshold_count and defect_type != "Normal":
            minor_classes.add(defect_type)
            if verbose:
                print(f"  - 소수 클래스: {defect_type} ({count}개, {count/total_files*100:.2f}%)")
    
    # 의미 없는 이름 필터링
    meaningless_classes = set()
    for defect_type in defect_type_counts.keys():
        if is_meaningless_name(defect_type) and defect_type != "Normal":
            meaningless_classes.add(defect_type)
            count = defect_type_counts[defect_type]
            if verbose:
                print(f"  - 의미 없는 이름: {defect_type} ({count}개)")
    
    # 삭제 대상 통합
    classes_to_remove = minor_classes | meaningless_classes
    
    if not classes_to_remove:
        if verbose:
            print(f"\n[결과] 삭제할 클래스가 없습니다.")
        return {
            'total_files': total_files,
            'deleted_files': 0,
            'removed_classes': 0
        }
    
    if verbose:
        print(f"\n[삭제 대상 클래스] 총 {len(classes_to_remove)}개")
        for cls in sorted(classes_to_remove):
            count = defect_type_counts[cls]
            reason = []
            if cls in minor_classes:
                reason.append("소수")
            if cls in meaningless_classes:
                reason.append("의미없음")
            print(f"  - {cls}: {count}개 ({', '.join(reason)})")
    
    # 3단계: 삭제 대상 파일 찾기
    if verbose:
        print(f"\n[3단계] 삭제 대상 파일 찾는 중...")
    files_to_remove = []
    
    for img_path_str, defect_types in files_with_defect_types.items():
        img_path = Path(img_path_str)
        # JSON 파일 경로: {filename}.jpg.json
        json_path = img_path.with_suffix(".jpg.json")
        
        # 삭제 대상 클래스가 포함되어 있는지 확인
        has_removable_class = any(dt in classes_to_remove for dt in defect_types)
        
        if has_removable_class:
            matched_classes = [dt for dt in defect_types if dt in classes_to_remove]
            files_to_remove.append({
                'image': img_path,
                'json': json_path,
                'defect_types': defect_types,
                'matched_classes': matched_classes
            })
    
    if verbose:
        print(f"  - 삭제 대상 파일 수: {len(files_to_remove)}개")
    
    # 4단계: 실제 삭제
    if not dry_run:
        if verbose:
            print(f"\n[4단계] 파일 삭제 중...")
        deleted_count = 0
        error_count = 0
        
        for item in files_to_remove:
            try:
                if item['image'].exists():
                    item['image'].unlink()
                if item['json'].exists():
                    item['json'].unlink()
                deleted_count += 1
                
                if verbose and deleted_count % 100 == 0:
                    print(f"  - 삭제 진행: {deleted_count}/{len(files_to_remove)}")
            except Exception as e:
                error_count += 1
                if verbose:
                    print(f"  [오류] {item['image'].name} 삭제 실패: {e}")
        
        if verbose:
            print(f"\n[삭제 완료]")
            print(f"  - 삭제된 파일: {deleted_count}개")
            print(f"  - 삭제된 이미지+JSON: {deleted_count * 2}개 파일")
            print(f"  - 오류 발생: {error_count}개")
    else:
        if verbose:
            print(f"\n[DRY RUN 모드]")
            print(f"  - 삭제될 파일 수: {len(files_to_remove)}개 (이미지 + JSON = {len(files_to_remove) * 2}개 파일)")
    
    return {
        'total_files': total_files,
        'deleted_files': len(files_to_remove) if not dry_run else 0,
        'removed_classes': len(classes_to_remove),
        'classes_to_remove': list(classes_to_remove),
        'defect_type_counts': dict(defect_type_counts)
    }


def main():
    """메인 함수 - CLI 인터페이스"""
    parser = argparse.ArgumentParser(
        description="데이터셋에서 소수 클래스 및 의미 없는 이름을 가진 결함 유형 데이터 삭제"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='데이터 디렉토리 경로 (기본값: data)'
    )
    parser.add_argument(
        '--min-ratio',
        type=float,
        default=0.01,
        help='최소 비율 (0.0-1.0). 이 비율 미만인 결함 유형은 삭제됨 (기본값: 0.01 = 1%%)'
    )
    parser.add_argument(
        '--min-count',
        type=int,
        default=None,
        help='최소 샘플 수. 이 값이 지정되면 min-ratio 대신 사용됨'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='실제 삭제하지 않고 미리보기만 수행'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='상세 출력 비활성화'
    )
    
    args = parser.parse_args()
    
    stats = cleanup_dataset(
        data_dir=args.data_dir,
        min_ratio=args.min_ratio,
        min_count=args.min_count,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
    
    if not args.quiet:
        print("\n" + "=" * 60)
        print("정리 완료!")
        print("=" * 60)
        print(f"전체 파일: {stats.get('total_files', 0)}개")
        print(f"삭제된 파일: {stats.get('deleted_files', 0)}개")
        print(f"제거된 클래스: {stats.get('removed_classes', 0)}개")


if __name__ == "__main__":
    main()

