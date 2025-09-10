"""
ログ管理ユーティリティ
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "dataset_generator",
    log_dir: str = "./logs",
    level: str = "INFO",
    file_rotation: bool = True,
    max_size_mb: int = 10,
    backup_count: int = 5
) -> logging.Logger:
    """
    ログ設定を初期化
    
    Parameters
    ----------
    name : str
        ロガー名
    log_dir : str
        ログディレクトリ
    level : str
        ログレベル
    file_rotation : bool
        ファイルローテーション有効化
    max_size_mb : int
        ファイル最大サイズ（MB）
    backup_count : int
        バックアップファイル数
        
    Returns
    -------
    logging.Logger
        設定済みロガー
    """
    # ログディレクトリ作成
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # ロガー作成
    logger = logging.getLogger(name)
    
    # レベル設定（文字列または数値に対応）
    if isinstance(level, str):
        logger.setLevel(getattr(logging, level.upper()))
    else:
        logger.setLevel(level)
    
    # 既存のハンドラーをクリア
    logger.handlers.clear()
    
    # フォーマッター
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # コンソールハンドラー
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラー
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"run_{timestamp}.log"
    
    if file_rotation:
        # ローテーションハンドラー
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
    else:
        # 通常ファイルハンドラー
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
    
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Logger initialized: {name}")
    logger.info(f"Log file: {log_file}")
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    既存のロガーを取得
    
    Parameters
    ----------
    name : str, optional
        ロガー名
        
    Returns
    -------
    logging.Logger
        ロガー
    """
    if name is None:
        name = "dataset_generator"
    return logging.getLogger(name)
