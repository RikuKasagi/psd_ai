"""
バッチデータセット生成パイプライン（改良版）
複数PSDファイルから統合データセットを生成
"""

import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil
import numpy as np
from PIL import Image

from utils.dataset.log import get_logger
from utils.dataset.io_utils import IOUtils
from utils.dataset.validate import Validator
from utils.dataset.manifest import ManifestGenerator
from utils.psd_tools.psd_split import extract_layers_from_psd
import yaml


class BatchPipeline:
    """複数PSDファイルのバッチ処理パイプライン（改良版）"""
    
    def __init__(self, config_path: str, classes_config_path: str):
        """
        Parameters
        ----------
        config_path : str
            パイプライン設定ファイルパス
        classes_config_path : str
            クラス設定ファイルパス
        """
        self.config_path = config_path
        self.classes_config_path = classes_config_path
        self.logger = get_logger()
        
        # 設定読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        with open(classes_config_path, 'r', encoding='utf-8') as f:
            self.classes_config = yaml.safe_load(f)
    
    def process_batch(
        self,
        psd_folder: str,
        output_path: str,
        include_hidden: bool = False,
        keep_temp: bool = False
    ) -> Dict[str, Any]:
        """
        バッチ処理実行
        
        Parameters
        ----------
        psd_folder : str
            PSDファイルフォルダパス
        output_path : str
            最終出力先
        include_hidden : bool
            非表示レイヤーも含めるか
        keep_temp : bool
            一時フォルダを保持するか
            
        Returns
        -------
        Dict[str, Any]
            処理結果
        """
        # 1. PSDファイル検索
        psd_files = self._find_psd_files(psd_folder)
        self.logger.info(f"見つかったPSDファイル: {len(psd_files)}個")
        for psd_file in psd_files:
            self.logger.info(f"  - {psd_file.name}")
        
        if not psd_files:
            raise ValueError(f"PSDファイルが見つかりません: {psd_folder}")
        
        # 2. 全PSDファイルからタイル生成
        all_tiles = []
        
        for psd_file in psd_files:
            self.logger.info(f"処理中: {psd_file.name}")
            
            # レイヤー抽出
            layers = extract_layers_from_psd(str(psd_file), include_hidden=include_hidden)
            
            # タイル生成（分割はしない）
            tiles = self._generate_tiles_only(layers, psd_file.stem)
            all_tiles.extend(tiles)
            
            self.logger.info(f"  {psd_file.name}: {len(tiles)}タイル生成")
        
        self.logger.info(f"全タイル生成完了: {len(all_tiles)}タイル")
        
        # 3. 全タイルをシャッフル
        self.logger.info("タイルシャッフル中...")
        random.seed(self.config['data_split']['seed'])
        random.shuffle(all_tiles)
        
        # 4. train/val/testに分割
        self.logger.info("最終データセット生成中...")
        result = self._create_final_dataset(all_tiles, output_path)
        
        return result
    
    def _find_psd_files(self, folder_path: str) -> List[Path]:
        """PSDファイルを検索"""
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"フォルダが存在しません: {folder_path}")
        
        psd_files = list(folder.glob("*.psd")) + list(folder.glob("*.psb"))
        return sorted(psd_files)
    
    def _generate_tiles_only(self, layers: Dict, source_name: str) -> List[Dict]:
        """
        タイル生成のみ（分割は行わない）
        
        Parameters
        ----------
        layers : Dict
            PSDレイヤー辞書
        source_name : str
            元ファイル名
            
        Returns
        -------
        List[Dict]
            タイル情報リスト（画像・マスクデータ含む）
        """
        from tools.mask_builder import MaskBuilder
        from tools.tiler import Tiler
        from utils.dataset.color_map import ColorMapper
        
        # 1. マスク統合
        color_mapper = ColorMapper(self.classes_config_path)
        mask_builder = MaskBuilder(color_mapper)
        index_mask = mask_builder.build_index_mask(layers)
        
        # 2. タイル分割
        tiler = Tiler(self.config['tiling'])
        tiles = tiler.generate_tiles(layers['original'], index_mask)
        
        # 3. タイル情報にソース名を追加
        for i, tile in enumerate(tiles):
            tile['source_psd'] = source_name
            tile['global_index'] = f"{source_name}_{i:06d}"
        
        return tiles
    
    def _create_final_dataset(self, all_tiles: List[Dict], output_path: str) -> Dict[str, Any]:
        """
        最終データセットを作成
        
        Parameters
        ----------
        all_tiles : List[Dict]
            全タイル情報
        output_path : str
            出力先パス
            
        Returns
        -------
        Dict[str, Any]
            作成結果
        """
        from tools.splitter import Splitter
        
        # 出力ディレクトリ作成
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 分割比率
        split_config = self.config['data_split']
        train_ratio = split_config['train_ratio']
        val_ratio = split_config['val_ratio']
        test_ratio = split_config['test_ratio']
        
        # 分割サイズ計算
        total_tiles = len(all_tiles)
        train_size = int(total_tiles * train_ratio)
        val_size = int(total_tiles * val_ratio)
        test_size = total_tiles - train_size - val_size
        
        self.logger.info(f"分割サイズ: train={train_size}, val={val_size}, test={test_size}")
        
        # 分割
        splits = {
            'train': all_tiles[:train_size],
            'val': all_tiles[train_size:train_size + val_size],
            'test': all_tiles[train_size + val_size:]
        }
        
        # 各splitのファイル保存
        result = {'splits': {}}
        
        for split_name, tiles in splits.items():
            if not tiles:
                continue
            
            # ディレクトリ作成
            split_dir = output_dir / split_name
            images_dir = split_dir / "images"
            masks_dir = split_dir / "masks"
            
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)
            
            # ファイル保存
            saved_count = 0
            for i, tile in enumerate(tiles):
                try:
                    # 通し番号でファイル名生成
                    filename = f"{i+1:05d}.png"
                    
                    # 画像保存
                    image_path = images_dir / filename
                    tile['image'].save(str(image_path))
                    
                    # マスク保存
                    mask_path = masks_dir / filename
                    mask_image = Image.fromarray(tile['mask'].astype(np.uint8))
                    mask_image.save(str(mask_path))
                    
                    saved_count += 1
                    
                except Exception as e:
                    self.logger.error(f"ファイル保存エラー: {filename}, {e}")
            
            result['splits'][split_name] = saved_count
            self.logger.info(f"{split_name}: {saved_count}ファイル保存完了")
        
        # 統計情報
        result.update({
            'total_files_processed': len(set(tile['source_psd'] for tile in all_tiles)),
            'total_tiles_generated': len(all_tiles),
            'output_path': str(output_dir)
        })
        
        return result
