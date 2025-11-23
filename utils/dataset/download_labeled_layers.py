import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from pymongo import MongoClient
from pymongo.collection import Collection
import gridfs
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 환경 변수에서 값 읽기
DEFAULT_HOST = os.getenv("MONGODB_HOST")
DEFAULT_PORT = int(os.getenv("MONGODB_PORT", "50002"))
DEFAULT_USER = os.getenv("MONGODB_USER")
DEFAULT_PASSWORD = os.getenv("MONGODB_PASSWORD")
DEFAULT_AUTH_DB = os.getenv("MONGODB_AUTH_DB", "admin")

SYSTEM_DATABASES = {"admin", "local", "config"}


@dataclass
class DownloadResult:
    downloaded: int = 0
    skipped_existing: int = 0
    missing_layers: List[int] = field(default_factory=list)
    missing_layers_with_docs: List[str] = field(default_factory=list)

    def extend(self, other: "DownloadResult") -> None:
        self.downloaded += other.downloaded
        self.skipped_existing += other.skipped_existing
        self.missing_layers.extend(other.missing_layers)
        self.missing_layers_with_docs.extend(other.missing_layers_with_docs)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download labeled L-PBF layers from MongoDB GridFS."
    )

    conn = parser.add_argument_group("MongoDB connection")
    conn.add_argument("--host", default=DEFAULT_HOST, help="MongoDB host name")
    conn.add_argument("--port", type=int, default=DEFAULT_PORT, help="MongoDB port")
    conn.add_argument("--username", "--user", dest="username", default=DEFAULT_USER)
    conn.add_argument("--password", "--pw", dest="password", default=DEFAULT_PASSWORD)
    conn.add_argument("--auth-db", default=DEFAULT_AUTH_DB)
    conn.add_argument(
        "--uri",
        help="Full MongoDB URI. Overrides host/port/username/password/auth-db arguments.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data"),
        help="Directory to save downloaded layers.",
    )
    parser.add_argument(
        "--databases",
        nargs="+",
        help="Explicit list of databases to process. Defaults to all non-system DBs.",
    )
    parser.add_argument(
        "--match",
        help="Regex filter for database names when --databases is not provided.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist in the output folder.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of labeled layers to download per database.",
    )
    parser.add_argument(
        "--total-limit",
        type=int,
        default=50000,
        help="Maximum total number of labeled layers to download across all databases (default: 50000).",
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        default=True,
        help="Persist document metadata alongside each downloaded image (default: True).",
    )
    parser.add_argument(
        "--no-metadata",
        dest="metadata",
        action="store_false",
        help="Do not save metadata JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching documents without downloading any data.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information about GridFS structure and document fields.",
    )

    return parser.parse_args(argv)


def build_client(args: argparse.Namespace) -> MongoClient:
    if args.uri:
        return MongoClient(args.uri)

    return MongoClient(
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password,
        authSource=args.auth_db,
    )


def resolve_databases(client: MongoClient, args: argparse.Namespace) -> List[str]:
    db_names: Iterable[str]
    if args.databases:
        db_names = args.databases
    else:
        db_names = [
            name
            for name in client.list_database_names()
            if name not in SYSTEM_DATABASES
        ]

    if args.match:
        import re

        pattern = re.compile(args.match)
        db_names = [name for name in db_names if pattern.search(name)]

    # 날짜 형식의 데이터베이스 이름을 연도 기준으로 내림차순 정렬 (2025, 2024, ...)
    def extract_year(db_name: str) -> int:
        """데이터베이스 이름에서 연도를 추출. 연도가 없으면 0 반환"""
        import re
        # 문자열이 숫자로 시작하는 경우 처음 4자리를 연도로 간주 (예: 20210914 -> 2021)
        if db_name and db_name[0].isdigit():
            # 처음 4자리가 20xx 형식인지 확인
            if len(db_name) >= 4 and db_name[:2] == '20' and db_name[2:4].isdigit():
                return int(db_name[:4])
        
        # 다른 위치에서 연도 패턴 찾기 (예: 2025, 2024)
        match = re.search(r'(20\d{2})', db_name)
        if match:
            return int(match.group(1))
        return 0
    
    # 연도 기준으로 내림차순 정렬 (최신 연도 먼저: 2025, 2024, 2023, ...)
    return sorted(db_names, key=extract_year, reverse=True)


def ensure_collections(db, db_name: str) -> Optional[gridfs.GridFS]:
    if "LayersModelDB" not in db.list_collection_names():
        print(f"[{db_name}] Missing LayersModelDB collection. Skipping.")
        return None

    bucket_name = f"{db_name}_vision"
    files_name = f"{bucket_name}.files"
    chunks_name = f"{bucket_name}.chunks"

    if files_name not in db.client[db_name].list_collection_names():
        print(f"[{db_name}] Missing {files_name} collection. Skipping.")
        return None

    if chunks_name not in db.client[db_name].list_collection_names():
        print(f"[{db_name}] Missing {chunks_name} collection. Skipping.")
        return None

    return gridfs.GridFS(db, collection=bucket_name)


def truthy_filter() -> Dict[str, Dict[str, Sequence]]:
    return {
        "IsLabeled": {"$in": [True, "true", "True", 1]},
    }


def doc_to_filename(
    db_name: str, layer_num: Optional[int], doc_id, extension: str
) -> str:
    layer_part = f"layer{layer_num:04d}" if layer_num is not None else "layerXXXX"
    return f"{db_name}_{layer_part}_{doc_id}.{extension}"


def write_bytes(target_path: Path, content: bytes, overwrite: bool) -> bool:
    if target_path.exists() and not overwrite:
        return False

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_path, "wb") as f:
        f.write(content)
    return True


def write_metadata(target_path: Path, document: Dict) -> None:
    metadata_path = target_path.with_suffix(target_path.suffix + ".json")
    doc_copy = dict(document)
    doc_copy["_id"] = str(doc_copy.get("_id"))

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(doc_copy, f, indent=2, ensure_ascii=False)




def download_for_db(
    db_name: str,
    db,
    fs: gridfs.GridFS,
    output_dir: Path,
    overwrite: bool,
    limit: Optional[int],
    persist_metadata: bool,
    dry_run: bool,
    debug: bool = False,
    total_downloaded: int = 0,
    total_limit: Optional[int] = None,
) -> Tuple[DownloadResult, int]:
    result = DownloadResult()
    collection: Collection = db["LayersModelDB"]

    if debug:
        # Print sample document structure
        sample_doc = collection.find_one(truthy_filter())
        if sample_doc:
            print(f"\n[DEBUG] Sample document fields: {list(sample_doc.keys())}")
            print(f"[DEBUG] Sample doc LayerNum: {sample_doc.get('LayerNum')}")
            print(f"[DEBUG] Sample doc LayerIdx: {sample_doc.get('LayerIdx')}")
            print(f"[DEBUG] Sample doc Extension: {sample_doc.get('Extension')}")
        
        # Print sample GridFS file structure
        sample_file = fs.find_one()
        if sample_file:
            print(f"[DEBUG] Sample GridFS file filename: {sample_file.filename}")
            print(f"[DEBUG] Sample GridFS file metadata: {sample_file.metadata}")
        else:
            print(f"[DEBUG] No files found in GridFS")
        
        # Count total GridFS files
        try:
            total_files = len(list(fs.find().limit(10000)))  # Limit to avoid memory issues
            print(f"[DEBUG] Total GridFS files (sampled): {total_files}")
        except Exception as e:
            print(f"[DEBUG] Could not count GridFS files: {e}")

    # 전체 제한이 있는 경우, 남은 개수만큼만 다운로드
    remaining_limit = None
    if total_limit is not None:
        remaining_limit = max(0, total_limit - total_downloaded)
        if remaining_limit == 0:
            print(f"[{db_name}] Total limit reached. Skipping this database.")
            return result, total_downloaded
    
    # 데이터베이스별 제한과 전체 제한 중 작은 값 사용
    effective_limit = limit
    if remaining_limit is not None:
        if effective_limit is None:
            effective_limit = remaining_limit
        else:
            effective_limit = min(effective_limit, remaining_limit)
    
    cursor = (
        collection.find(truthy_filter(), sort=[("LayerNum", 1)])
        if effective_limit is None
        else collection.find(truthy_filter(), sort=[("LayerNum", 1)]).limit(effective_limit)
    )

    for doc in cursor:
        # 전체 제한 체크
        if total_limit is not None and total_downloaded >= total_limit:
            print(f"[{db_name}] Total limit ({total_limit}) reached. Stopping download.")
            break
        layer_num = doc.get("LayerNum")
        layer_idx = doc.get("LayerIdx")
        extension = doc.get("Extension", "jpg").lstrip(".")
        filename = doc_to_filename(db_name, layer_num, doc.get("_id"), extension)
        target_path = output_dir / filename

        if dry_run:
            print(f"[DRY-RUN] Would download {db_name}:{layer_num} -> {target_path}")
            continue

        file_id = None
        
        # Try multiple search methods
        # Method 1: Use LayerNum to search GridFS metadata.LayerIdx (LayerNum == LayerIdx in most cases)
        if layer_num is not None:
            file_query = {"metadata.LayerIdx": layer_num}
            grid_file = fs.find_one(file_query)
            if grid_file:
                file_id = grid_file._id
        
        # Method 2: If LayerIdx is explicitly provided, try that too
        if file_id is None and layer_idx is not None:
            file_query = {"metadata.LayerIdx": layer_idx}
            grid_file = fs.find_one(file_query)
            if grid_file:
                file_id = grid_file._id
        
        # Method 3: Search by filename pattern (e.g., "1 Layer_FirstShot.jpg")
        if file_id is None and layer_num is not None:
            # Try different filename patterns
            patterns = [
                f".*{layer_num}\\s+Layer.*",  # "1 Layer_FirstShot.jpg"
                f".*layer.*{layer_num:04d}.*",  # "layer0001"
                f".*{layer_num:04d}.*layer.*",  # "0001_layer"
                f".*{layer_num}\\s+layer.*",  # "1 layer"
            ]
            for pattern in patterns:
                filename_query = {"filename": {"$regex": pattern, "$options": "i"}}
                grid_file = fs.find_one(filename_query)
                if grid_file:
                    file_id = grid_file._id
                    break
        
        # Method 4: metadata.LayerNum (if exists)
        if file_id is None and layer_num is not None:
            layer_num_query = {"metadata.LayerNum": layer_num}
            grid_file = fs.find_one(layer_num_query)
            if grid_file:
                file_id = grid_file._id

        if file_id is None:
            # Only print first few missing layers to avoid spam
            if len(result.missing_layers) < 5:
                print(f"[{db_name}] Missing layer data for LayerNum={layer_num}, LayerIdx={layer_idx}")
            elif len(result.missing_layers) == 5:
                print(f"[{db_name}] ... (suppressing further missing layer messages)")
            result.missing_layers.append(layer_num)
            result.missing_layers_with_docs.append(str(doc.get("_id")))
            continue

        content = fs.get(file_id).read()
        saved = write_bytes(target_path, content, overwrite)
        if saved:
            result.downloaded += 1
            total_downloaded += 1
            print(f"[{db_name}] Saved {target_path} ({total_downloaded}/{total_limit if total_limit else 'unlimited'})")
            if persist_metadata:
                write_metadata(target_path, doc)
        else:
            result.skipped_existing += 1
            print(f"[{db_name}] Skipped existing {target_path}")

    return result, total_downloaded


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    client = build_client(args)
    db_names = resolve_databases(client, args)

    if not db_names:
        print("No databases matched criteria. Nothing to do.")
        return

    print(f"Processing {len(db_names)} database(s): {', '.join(db_names)}")
    print(f"Total download limit: {args.total_limit}")
    total = DownloadResult()
    total_downloaded = 0

    for db_name in db_names:
        print(f"\n==> Database: {db_name}")
        db = client[db_name]
        fs = ensure_collections(db, db_name)
        if not fs:
            continue

        result, total_downloaded = download_for_db(
            db_name=db_name,
            db=db,
            fs=fs,
            output_dir=output_dir,
            overwrite=args.overwrite,
            limit=args.limit,
            persist_metadata=args.metadata,
            dry_run=args.dry_run,
            debug=args.debug,
            total_downloaded=total_downloaded,
            total_limit=args.total_limit,
        )
        total.extend(result)
        
        # 전체 제한에 도달했으면 중단
        if args.total_limit is not None and total_downloaded >= args.total_limit:
            print(f"\nTotal limit ({args.total_limit}) reached. Stopping download.")
            break

    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Downloaded: {total.downloaded}")
    print(f"Skipped existing: {total.skipped_existing}")
    print(f"Total downloaded: {total_downloaded}")
    if total.missing_layers:
        print(f"Missing layers: {len(total.missing_layers)}")
        if len(total.missing_layers) <= 10:
            print("Layer numbers:", total.missing_layers)
    if total.missing_layers_with_docs:
        print(
            f"Docs without GridFS data: {len(total.missing_layers_with_docs)}"
        )


if __name__ == "__main__":
    main()

