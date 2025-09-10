"""
データ分割ツール
train/val/test 分割（seed対応）
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from ..dataset.log import get_logger
from ..dataset.io_utils import IOUtils


class Splitter:
    """データ分割クラス"""
    
    def __init__(self, config: Dict):
        """
        Parameters
        ----------
        config : Dict
            データ分割設定
        """
        self.config = config
        self.logger = get_logger()
        
        # 設定値の取得
        self.train_ratio = config.get('train_ratio', 0.8)
        self.val_ratio = config.get('val_ratio', 0.1)
        self.test_ratio = config.get('test_ratio', 0.1)
        self.seed = config.get('seed', 42)
        self.shuffle = config.get('shuffle', True)
        self.stratify = config.get('stratify', False)
        
        # 検証
        self._validate_config()
        
        # シード設定
        self._set_seed()
    
    def _validate_config(self):
        """設定値を検証"""
        ratios = [self.train_ratio, self.val_ratio, self.test_ratio]
        
        # 比率の範囲チェック
        if any(r < 0 or r > 1 for r in ratios):
            raise ValueError(f"分割比率は0-1の範囲である必要があります: {ratios}")
        
        # 合計チェック
        total = sum(ratios)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"分割比率の合計は1.0である必要があります: {total}")
        
        # 非負チェック
        if any(r == 0 for r in ratios):
            self.logger.warning("分割比率に0が含まれています。該当する分割セットは空になります。")
    
    def _set_seed(self):
        """ランダムシードを設定"""
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.logger.info(f"ランダムシード設定: {self.seed}")
    
    def split_tiles(
        self, 
        tiles: List[Dict], 
        output_base_path: str
    ) -> Dict[str, List[str]]:
        """
        タイルをtrain/val/testに分割
        
        Parameters
        ----------
        tiles : List[Dict]
            タイルリスト
        output_base_path : str
            出力ベースパス
            
        Returns
        -------
        Dict[str, List[str]]
            分割結果（split名 -> ファイル名リスト）
        """
        if not tiles:
            self.logger.warning("タイルリストが空です")
            return {'train': [], 'val': [], 'test': []}
        
        self.logger.info(f"データ分割開始: {len(tiles)}個のタイル")
        
        # 分割インデックスを計算
        split_indices = self._calculate_split_indices(len(tiles))
        
        # 結果辞書
        split_results = {}
        
        # 各分割セットを処理
        for split_name, indices in split_indices.items():
            if not indices:
                split_results[split_name] = []
                continue
            
            # 該当タイルを抽出
            split_tiles = [tiles[i] for i in indices]
            
            # 保存
            split_dir = Path(output_base_path) / split_name
            saved_files = self._save_split_tiles(split_tiles, str(split_dir), split_name)
            
            split_results[split_name] = saved_files
            
            self.logger.info(f"{split_name}: {len(saved_files)}ファイル保存")
        
        # 分割統計をログ出力
        self._log_split_statistics(split_results, len(tiles))
        
        return split_results
    
    def _calculate_split_indices(self, total_count: int) -> Dict[str, List[int]]:
        """
        分割インデックスを計算
        
        Parameters
        ----------
        total_count : int
            総データ数
            
        Returns
        -------
        Dict[str, List[int]]
            分割インデックス辞書
        """
        # インデックスリストを作成
        indices = list(range(total_count))
        
        # シャッフル
        if self.shuffle:
            random.shuffle(indices)
        
        # 分割サイズを計算
        train_size = int(total_count * self.train_ratio)
        val_size = int(total_count * self.val_ratio)
        test_size = total_count - train_size - val_size  # 余りはtestに
        
        # 分割実行
        split_indices = {
            'train': indices[:train_size],
            'val': indices[train_size:train_size + val_size],
            'test': indices[train_size + val_size:]
        }
        
        self.logger.info(f"分割サイズ: train={train_size}, val={val_size}, test={test_size}")
        
        return split_indices
    
    def _save_split_tiles(
        self, 
        tiles: List[Dict], 
        output_dir: str, 
        split_name: str
    ) -> List[str]:
        """
        分割されたタイルを保存
        
        Parameters
        ----------
        tiles : List[Dict]
            タイルリスト
        output_dir : str
            出力ディレクトリ
        split_name : str
            分割名
            
        Returns
        -------
        List[str]
            保存されたファイル名のリスト
        """
        if not tiles:
            return []
        
        # ディレクトリ作成
        images_dir = Path(output_dir) / "images"
        masks_dir = Path(output_dir) / "masks"
        
        IOUtils.ensure_dir(images_dir)
        IOUtils.ensure_dir(masks_dir)
        
        saved_files = []
        
        for i, tile_info in enumerate(tiles):
            # ファイル名生成（分割名を含む）
            filename = IOUtils.generate_filename(f"{split_name}_{i:06d}", 0, "png")
            
            # 画像保存
            image_path = images_dir / filename
            success_img = IOUtils.save_image(tile_info['image'], str(image_path))
            
            # マスク保存
            mask_path = masks_dir / filename
            success_mask = IOUtils.save_index_mask(tile_info['mask'], str(mask_path))
            
            if success_img and success_mask:
                saved_files.append(filename)
            else:
                self.logger.error(f"ファイル保存失敗: {filename}")
        
        return saved_files
    
    def _log_split_statistics(self, split_results: Dict[str, List[str]], total_count: int):
        """分割統計をログ出力"""
        self.logger.info("=== データ分割統計 ===")
        
        for split_name, files in split_results.items():
            count = len(files)
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            
            expected_ratio = {
                'train': self.train_ratio,
                'val': self.val_ratio,
                'test': self.test_ratio
            }.get(split_name, 0) * 100
            
            self.logger.info(
                f"{split_name}: {count}ファイル ({percentage:.1f}%) "
                f"[期待値: {expected_ratio:.1f}%]"
            )
    
    def validate_split_results(
        self, 
        split_results: Dict[str, List[str]]
    ) -> Tuple[bool, List[str]]:
        """
        分割結果を検証
        
        Parameters
        ----------
        split_results : Dict[str, List[str]]
            分割結果
            
        Returns
        -------
        Tuple[bool, List[str]]
            (検証結果, 問題点リスト)
        """
        issues = []
        
        # 基本チェック
        expected_splits = ['train', 'val', 'test']
        for split_name in expected_splits:
            if split_name not in split_results:
                issues.append(f"分割セットが不足: {split_name}")
        
        # ファイル数チェック
        total_files = sum(len(files) for files in split_results.values())
        if total_files == 0:
            issues.append("分割結果にファイルがありません")
        
        # 比率チェック
        if total_files > 0:
            for split_name in expected_splits:
                if split_name in split_results:
                    actual_count = len(split_results[split_name])
                    actual_ratio = actual_count / total_files
                    
                    expected_ratio = {
                        'train': self.train_ratio,
                        'val': self.val_ratio,
                        'test': self.test_ratio
                    }.get(split_name, 0)
                    
                    # 誤差許容範囲（5%）
                    if abs(actual_ratio - expected_ratio) > 0.05:
                        issues.append(
                            f"{split_name}の比率が期待値から大きく逸脱: "
                            f"{actual_ratio:.3f} vs 期待値 {expected_ratio:.3f}"
                        )
        
        # 重複チェック
        all_files = []
        for files in split_results.values():
            all_files.extend(files)
        
        if len(all_files) != len(set(all_files)):
            issues.append("分割セット間でファイル名の重複があります")
        
        return len(issues) == 0, issues
    
    def get_split_statistics(self, split_results: Dict[str, List[str]]) -> Dict:
        """
        分割統計を取得
        
        Parameters
        ----------
        split_results : Dict[str, List[str]]
            分割結果
            
        Returns
        -------
        Dict
            統計情報
        """
        total_files = sum(len(files) for files in split_results.values())
        
        stats = {
            'total_files': total_files,
            'splits': {},
            'ratios': {
                'expected': {
                    'train': self.train_ratio,
                    'val': self.val_ratio,
                    'test': self.test_ratio
                },
                'actual': {}
            },
            'seed': self.seed,
            'shuffle': self.shuffle
        }
        
        # 各分割の詳細
        for split_name, files in split_results.items():
            count = len(files)
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            
            stats['splits'][split_name] = {
                'count': count,
                'percentage': round(percentage, 2),
                'files_sample': files[:5] if len(files) > 5 else files
            }
            
            stats['ratios']['actual'][split_name] = round(count / total_files, 4) if total_files > 0 else 0
        
        return stats
    
    def reproduce_split(self, total_count: int) -> Dict[str, List[int]]:
        """
        分割を再現（インデックスのみ）
        
        Parameters
        ----------
        total_count : int
            総データ数
            
        Returns
        -------
        Dict[str, List[int]]
            分割インデックス辞書
        """
        # 同じシードで分割を再実行
        self._set_seed()
        return self._calculate_split_indices(total_count)
