"""
タイル分割ツール
画像・マスクのスケール/タイル分割（パディング対応）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageOps
import logging
from pathlib import Path

from ..dataset.log import get_logger
from ..dataset.io_utils import IOUtils


class Tiler:
    """タイル分割クラス"""
    
    def __init__(self, config: Dict):
        """
        Parameters
        ----------
        config : Dict
            タイル分割設定
        """
        self.config = config
        self.logger = get_logger()
        
        # 設定値の取得
        self.tile_size = config.get('tile_size', 512)
        self.overlap = config.get('overlap', 128)
        self.stride = config.get('stride') or (self.tile_size - self.overlap)
        self.pad_mode = config.get('padding', {}).get('mode', 'constant')
        self.pad_value = config.get('padding', {}).get('value', 0)
        self.min_foreground_ratio = config.get('min_foreground_ratio', 0.0)
        
        # 検証
        self._validate_config()
    
    def _validate_config(self):
        """設定値を検証"""
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive: {self.tile_size}")
        
        if self.overlap < 0:
            raise ValueError(f"overlap must be non-negative: {self.overlap}")
        
        if self.overlap >= self.tile_size:
            raise ValueError(f"overlap must be less than tile_size: {self.overlap} >= {self.tile_size}")
        
        if self.stride <= 0:
            raise ValueError(f"stride must be positive: {self.stride}")
    
    def generate_tiles(
        self, 
        original: Image.Image, 
        index_mask: np.ndarray
    ) -> List[Dict]:
        """
        画像とマスクからタイルを生成
        
        Parameters
        ----------
        original : Image.Image
            元画像（RGB）
        index_mask : np.ndarray
            インデックスマスク（H, W）
            
        Returns
        -------
        List[Dict]
            タイル情報のリスト
            各要素: {'image': Image, 'mask': np.ndarray, 'position': (x, y), 'index': int}
        """
        # サイズ検証
        if original.size != (index_mask.shape[1], index_mask.shape[0]):
            raise ValueError(f"画像とマスクのサイズが不一致: {original.size} != {index_mask.shape[:2][::-1]}")
        
        self.logger.info(f"タイル生成開始: 画像サイズ={original.size}, タイルサイズ={self.tile_size}, オーバーラップ={self.overlap}")
        
        # パディングを適用
        padded_original, padded_mask, padding_info = self._apply_padding(original, index_mask)
        
        # タイル座標を計算
        tile_positions = self._calculate_tile_positions(padded_original.size)
        
        self.logger.info(f"タイル位置計算完了: {len(tile_positions)}個のタイル")
        
        # タイルを生成
        tiles = []
        for i, (x, y) in enumerate(tile_positions):
            tile_image, tile_mask = self._extract_tile(padded_original, padded_mask, x, y)
            
            # 前景ピクセル比率をチェック
            if self._should_keep_tile(tile_mask):
                tiles.append({
                    'image': tile_image,
                    'mask': tile_mask,
                    'position': (x, y),
                    'index': i,
                    'padded_position': True if self._is_padded_area(x, y, padding_info) else False
                })
        
        self.logger.info(f"タイル生成完了: {len(tiles)}個のタイルを生成（フィルタ後）")
        
        return tiles
    
    def _apply_padding(
        self, 
        original: Image.Image, 
        index_mask: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray, Dict]:
        """
        画像とマスクにパディングを適用
        
        Parameters
        ----------
        original : Image.Image
            元画像
        index_mask : np.ndarray
            インデックスマスク
            
        Returns
        -------
        Tuple[Image.Image, np.ndarray, Dict]
            (パディング後画像, パディング後マスク, パディング情報)
        """
        width, height = original.size
        
        # 必要なパディングサイズを計算
        # 右端・下端がタイルサイズで割り切れるように調整
        pad_right = 0
        pad_bottom = 0
        
        # 右端のパディング
        if (width - self.tile_size) % self.stride != 0:
            tiles_x = (width - self.tile_size) // self.stride + 1
            required_width = self.tile_size + tiles_x * self.stride
            pad_right = required_width - width
        
        # 下端のパディング
        if (height - self.tile_size) % self.stride != 0:
            tiles_y = (height - self.tile_size) // self.stride + 1
            required_height = self.tile_size + tiles_y * self.stride
            pad_bottom = required_height - height
        
        padding_info = {
            'original_size': (width, height),
            'pad_right': pad_right,
            'pad_bottom': pad_bottom,
            'padded_size': (width + pad_right, height + pad_bottom)
        }
        
        # 画像にパディング適用
        if pad_right > 0 or pad_bottom > 0:
            # PIL の ImageOps.expand でパディング
            if self.pad_mode == 'constant':
                fill_color = self.pad_value if original.mode == 'L' else (self.pad_value,) * len(original.getbands())
                padded_original = ImageOps.expand(original, border=(0, 0, pad_right, pad_bottom), fill=fill_color)
            else:
                # 他のパディングモードは後で実装（今は constant のみ）
                padded_original = ImageOps.expand(original, border=(0, 0, pad_right, pad_bottom), fill=0)
            
            # マスクにパディング適用
            padded_mask = np.pad(
                index_mask, 
                ((0, pad_bottom), (0, pad_right)), 
                mode='constant', 
                constant_values=0  # 背景クラスID
            )
            
            self.logger.info(f"パディング適用: {original.size} -> {padded_original.size}")
        else:
            padded_original = original
            padded_mask = index_mask
            self.logger.info("パディング不要")
        
        return padded_original, padded_mask, padding_info
    
    def _calculate_tile_positions(self, image_size: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        タイル位置を計算
        
        Parameters
        ----------
        image_size : Tuple[int, int]
            画像サイズ（width, height）
            
        Returns
        -------
        List[Tuple[int, int]]
            タイル左上座標のリスト
        """
        width, height = image_size
        positions = []
        
        # Y座標
        y = 0
        while y + self.tile_size <= height:
            # X座標
            x = 0
            while x + self.tile_size <= width:
                positions.append((x, y))
                x += self.stride
            y += self.stride
        
        return positions
    
    def _extract_tile(
        self, 
        image: Image.Image, 
        mask: np.ndarray, 
        x: int, 
        y: int
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        指定位置からタイルを抽出
        
        Parameters
        ----------
        image : Image.Image
            パディング済み画像
        mask : np.ndarray
            パディング済みマスク
        x, y : int
            タイル左上座標
            
        Returns
        -------
        Tuple[Image.Image, np.ndarray]
            (タイル画像, タイルマスク)
        """
        # 画像タイル
        tile_image = image.crop((x, y, x + self.tile_size, y + self.tile_size))
        
        # マスクタイル
        tile_mask = mask[y:y + self.tile_size, x:x + self.tile_size]
        
        return tile_image, tile_mask
    
    def _should_keep_tile(self, tile_mask: np.ndarray) -> bool:
        """
        タイルを保持すべきかチェック
        
        Parameters
        ----------
        tile_mask : np.ndarray
            タイルマスク
            
        Returns
        -------
        bool
            保持フラグ
        """
        if self.min_foreground_ratio <= 0.0:
            return True
        
        # 前景ピクセル比率を計算
        foreground_pixels = np.sum(tile_mask > 0)  # 背景以外
        total_pixels = tile_mask.size
        foreground_ratio = foreground_pixels / total_pixels
        
        return foreground_ratio >= self.min_foreground_ratio
    
    def _is_padded_area(self, x: int, y: int, padding_info: Dict) -> bool:
        """
        タイルがパディング領域を含むかチェック
        
        Parameters
        ----------
        x, y : int
            タイル左上座標
        padding_info : Dict
            パディング情報
            
        Returns
        -------
        bool
            パディング領域を含むフラグ
        """
        original_width, original_height = padding_info['original_size']
        
        # タイルの右端・下端がオリジナル領域を超えているかチェック
        tile_right = x + self.tile_size
        tile_bottom = y + self.tile_size
        
        return tile_right > original_width or tile_bottom > original_height
    
    def save_tiles(
        self, 
        tiles: List[Dict], 
        output_dir: str, 
        base_name: str = "tile"
    ) -> List[str]:
        """
        タイルをファイルに保存
        
        Parameters
        ----------
        tiles : List[Dict]
            タイルリスト
        output_dir : str
            出力ディレクトリ
        base_name : str
            ベース名
            
        Returns
        -------
        List[str]
            保存されたファイル名のリスト
        """
        saved_files = []
        
        # ディレクトリ作成
        images_dir = Path(output_dir) / "images"
        masks_dir = Path(output_dir) / "masks"
        
        IOUtils.ensure_dir(images_dir)
        IOUtils.ensure_dir(masks_dir)
        
        for tile_info in tiles:
            # ファイル名生成
            filename = IOUtils.generate_filename(base_name, tile_info['index'])
            
            # 画像保存
            image_path = images_dir / filename
            if IOUtils.save_image(tile_info['image'], str(image_path)):
                saved_files.append(filename)
            
            # マスク保存
            mask_path = masks_dir / filename
            IOUtils.save_index_mask(tile_info['mask'], str(mask_path))
        
        self.logger.info(f"タイル保存完了: {len(saved_files)}ファイル -> {output_dir}")
        
        return saved_files
    
    def get_statistics(self, tiles: List[Dict]) -> Dict:
        """
        タイル統計を取得
        
        Parameters
        ----------
        tiles : List[Dict]
            タイルリスト
            
        Returns
        -------
        Dict
            統計情報
        """
        stats = {
            'total_tiles': len(tiles),
            'padded_tiles': sum(1 for t in tiles if t.get('padded_position', False)),
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'stride': self.stride,
            'class_distribution': {}
        }
        
        # クラス分布統計
        all_pixels = {}
        for tile_info in tiles:
            mask = tile_info['mask']
            unique_classes, counts = np.unique(mask, return_counts=True)
            
            for class_id, count in zip(unique_classes, counts):
                if class_id not in all_pixels:
                    all_pixels[class_id] = 0
                all_pixels[class_id] += count
        
        total_pixels = sum(all_pixels.values())
        for class_id, pixel_count in all_pixels.items():
            percentage = (pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
            stats['class_distribution'][int(class_id)] = {
                'pixel_count': int(pixel_count),
                'percentage': round(percentage, 2)
            }
        
        return stats
