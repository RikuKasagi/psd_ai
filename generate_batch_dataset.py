#!/usr/bin/env python3
"""
バッチデータセット生成スクリプト
複数のPSDファイルを一括処理してデータセットを生成
"""

import argparse
import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime
import random

# プロジェクトルートをPythonパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from utils.psd_tools.psd_split import extract_layers_from_psd
from tools.batch_pipeline import BatchPipeline
from utils.dataset.log import setup_logger, get_logger


def find_psd_files(folder_path: str) -> List[Path]:
    """指定フォルダからPSDファイルを検索"""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"フォルダが見つかりません: {folder_path}")
    
    psd_files = list(folder.glob("*.psd")) + list(folder.glob("*.psb"))
    if not psd_files:
        raise ValueError(f"PSDファイルが見つかりません: {folder_path}")
    
    return sorted(psd_files)


def main():
    parser = argparse.ArgumentParser(
        description="複数PSDファイルからバッチデータセット生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用方法
  python generate_batch_dataset.py ./psd_folder

  # 設定ファイルを指定
  python generate_batch_dataset.py ./psd_folder --config my_config.yaml

  # 出力先とtempフォルダを指定
  python generate_batch_dataset.py ./psd_folder --output ./my_dataset --temp ./my_temp

  # temp削除せずに保持
  python generate_batch_dataset.py ./psd_folder --keep-temp

  # 詳細ログ出力
  python generate_batch_dataset.py ./psd_folder --log-level DEBUG
        """
    )
    
    # 必須引数
    parser.add_argument(
        "psd_folder",
        help="PSDファイルが格納されたフォルダのパス"
    )
    
    # オプション引数
    parser.add_argument(
        "--config",
        default="./config/build_config.yaml",
        help="パイプライン設定ファイルのパス（デフォルト: ./config/build_config.yaml）"
    )
    
    parser.add_argument(
        "--classes",
        default="./config/classes.yaml",
        help="クラス定義ファイルのパス（デフォルト: ./config/classes.yaml）"
    )
    
    parser.add_argument(
        "--output",
        default="./dataset",
        help="出力データセットのパス（デフォルト: ./dataset）"
    )
    
    parser.add_argument(
        "--temp",
        help="一時ファイル保存フォルダ（指定しない場合は自動生成）"
    )
    
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="処理完了後もtempフォルダを削除しない"
    )
    
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="非表示レイヤーも含めて処理"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="ログレベル（デフォルト: INFO）"
    )
    
    parser.add_argument(
        "--base-name",
        default="batch",
        help="出力ベース名（デフォルト: batch）"
    )
    
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=42,
        help="タイル混合用のランダムシード（デフォルト: 42）"
    )
    
    args = parser.parse_args()
    
    # ログ設定
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = setup_logger(
        name="batch_dataset_generator",
        log_dir=str(log_dir),
        level=args.log_level
    )
    
    logger.info("=== バッチデータセット生成開始 ===")
    logger.info(f"PSDフォルダ: {args.psd_folder}")
    logger.info(f"出力先: {args.output}")
    
    try:
        # 1. PSDファイル検索
        logger.info("1. PSDファイル検索")
        psd_files = find_psd_files(args.psd_folder)
        logger.info(f"見つかったPSDファイル: {len(psd_files)}個")
        for psd_file in psd_files:
            logger.info(f"  - {psd_file.name}")
        
        # 2. tempフォルダ設定
        if args.temp:
            temp_dir = Path(args.temp)
            temp_dir.mkdir(parents=True, exist_ok=True)
            use_temp_dir = str(temp_dir)
            auto_temp = False
        else:
            temp_dir = tempfile.mkdtemp(prefix="psd_batch_")
            use_temp_dir = temp_dir
            auto_temp = True
        
        logger.info(f"一時フォルダ: {use_temp_dir}")
        
        # 3. バッチパイプライン初期化
        logger.info("2. バッチパイプライン初期化")
        pipeline = BatchPipeline(
            config_path=args.config,
            classes_config_path=args.classes
        )
        
        # 4. バッチ処理実行
        logger.info("3. バッチ処理実行")
        result = pipeline.process_batch(
            psd_folder=args.psd_folder,
            output_path=args.output,
            include_hidden=args.include_hidden,
            keep_temp=args.keep_temp
        )
        
        # 5. 結果表示
        logger.info("=== バッチ処理完了 ===")
        logger.info(f"処理ファイル数: {result['total_files_processed']}")
        logger.info(f"生成タイル数: {result['total_tiles_generated']}")
        logger.info(f"train: {result['splits']['train']}タイル")
        logger.info(f"val: {result['splits']['val']}タイル")
        logger.info(f"test: {result['splits']['test']}タイル")
        logger.info(f"出力先: {result['output_path']}")
        
        # 6. temp削除
        if not args.keep_temp:
            if auto_temp:
                shutil.rmtree(use_temp_dir)
                logger.info(f"一時フォルダを削除: {use_temp_dir}")
            else:
                logger.info(f"一時フォルダを保持: {use_temp_dir}")
        else:
            logger.info(f"一時フォルダを保持: {use_temp_dir}")
        
        print(f"\n✅ バッチ処理完了!")
        print(f"📁 出力先: {args.output}")
        print(f"📊 総タイル数: {result['total_tiles_generated']}")
        print(f"📂 処理ファイル数: {result['total_files_processed']}")
        
    except Exception as e:
        logger.error(f"バッチ処理エラー: {e}")
        print(f"\n❌ エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
