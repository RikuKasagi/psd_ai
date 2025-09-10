"""
タイル分割ツール
画像・マスクのスケール/タイル分割（横長画像の縦向き変換、グリッド分割対応）
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image, ImageOps
import logging
from pathlib import Path

from utils.dataset.log import get_logger
from utils.dataset.io_utils import IOUtils


class Tiler:
    """タイル分割クラス（新仕様：横長→縦向き変換、グリッド分割）"""
    
    def __init__(self, config: Dict):
        """
        Parameters
        ----------
        config : Dict
            タイル分割設定
            - tile_size: 分割後のタイルサイズ（正方形）
            - grid_size: [縦の分割数, 横の分割数] 
            - overlap: オーバーラップのピクセル数（片側）
            - auto_orient: 横長画像を縦向きに変換するか（デフォルト: True）
        """
        self.config = config
        self.logger = get_logger()
        
        # 設定値の取得
        self.tile_size = config.get('tile_size', 512)
        self.grid_size = config.get('grid_size', [4, 4])  # [rows, cols]
        self.overlap = config.get('overlap', 128)  # 片側のピクセル数
        self.auto_orient = config.get('auto_orient', True)  # 横長→縦向き変換
        self.pad_mode = config.get('padding', {}).get('mode', 'constant')
        self.pad_value = config.get('padding', {}).get('value', 0)
        self.min_foreground_ratio = config.get('min_foreground_ratio', 0.0)
        
        # stride計算（オーバーラップを考慮）
        self.stride = self.tile_size - self.overlap
        
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
        
        if len(self.grid_size) != 2 or any(x <= 0 for x in self.grid_size):
            raise ValueError(f"grid_size must be [rows, cols] with positive values: {self.grid_size}")
        
        if self.stride <= 0:
            raise ValueError(f"stride must be positive: {self.stride}")
        
        self.logger.info(f"タイル分割設定: tile_size={self.tile_size}, grid={self.grid_size}, overlap={self.overlap}(片側)")
    
    def _auto_orient_image(self, image: Image.Image) -> Image.Image:
        """横長画像を縦向きに変換"""
        width, height = image.size
        
        if self.auto_orient and width > height:
            self.logger.info(f"横長画像を縦向きに変換: {width}x{height} -> {height}x{width}")
            return image.rotate(90, expand=True)
        
        return image
    
    def _calculate_target_size(self, rows: int, cols: int) -> Tuple[int, int]:
        """グリッド分割に必要な画像サイズを計算"""
        # オーバーラップを考慮したサイズ計算
        # 最後のタイル以外は overlap分重複するため
        target_width = cols * self.tile_size - (cols - 1) * self.overlap
        target_height = rows * self.tile_size - (rows - 1) * self.overlap
        
        return target_width, target_height
    
    def _resize_for_grid(self, image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """グリッド分割用にリサイズ"""
        original_size = image.size
        self.logger.info(f"グリッド分割用リサイズ: {original_size} -> {target_size}")
        
        # アスペクト比を保持してリサイズし、必要に応じてクロップ
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        
        return image_resized
    
    def generate_tiles(
        self, 
        original: Image.Image, 
        index_mask: np.ndarray
    ) -> List[Dict]:
        """
        画像とマスクからタイルを生成（新仕様：グリッド分割）
        
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
            各要素: {'image': Image, 'mask': np.ndarray, 'position': (row, col), 'index': int}
        """
        # 1. 横長画像の縦向き変換
        oriented_image = self._auto_orient_image(original)
        
        # マスクも同様に変換
        if self.auto_orient and original.size[0] > original.size[1]:
            oriented_mask = np.rot90(index_mask, k=1)  # 90度回転
        else:
            oriented_mask = index_mask
        
        # サイズ検証
        if oriented_image.size != (oriented_mask.shape[1], oriented_mask.shape[0]):
            raise ValueError(f"画像とマスクのサイズが不一致: {oriented_image.size} != {oriented_mask.shape[::-1]}")
        
        # 2. グリッド分割用のターゲットサイズ計算
        rows, cols = self.grid_size
        target_width, target_height = self._calculate_target_size(rows, cols)
        
        self.logger.info(f"タイル生成開始: 画像サイズ={oriented_image.size}, グリッド={rows}x{cols}, タイルサイズ={self.tile_size}, オーバーラップ={self.overlap}")
        self.logger.info(f"ターゲットサイズ: {target_width}x{target_height}")
        
        # 3. リサイズ
        resized_image = self._resize_for_grid(oriented_image, (target_width, target_height))
        resized_mask = self._resize_mask_for_grid(oriented_mask, (target_width, target_height))
        
        # 4. グリッド分割実行
        tiles = self._split_into_grid(resized_image, resized_mask, rows, cols)
        
        # 5. フィルタリング
        filtered_tiles = self._filter_tiles(tiles)
        
        self.logger.info(f"タイル生成完了: {len(filtered_tiles)}個のタイルを生成（元:{len(tiles)}個）")
        
        return filtered_tiles
    
    def _resize_mask_for_grid(self, mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """マスクをグリッド分割用にリサイズ"""
        # PIL Imageに変換してリサイズ（nearest neighbor補間）
        mask_image = Image.fromarray(mask.astype(np.uint8))
        mask_resized = mask_image.resize(target_size, Image.Resampling.NEAREST)
        
        return np.array(mask_resized)
    
    def _split_into_grid(self, image: Image.Image, mask: np.ndarray, rows: int, cols: int) -> List[Dict]:
        """画像とマスクをグリッド分割"""
        tiles = []
        tile_index = 0
        
        for row in range(rows):
            for col in range(cols):
                # タイル位置計算（オーバーラップ考慮）
                start_x = col * self.stride
                start_y = row * self.stride
                end_x = start_x + self.tile_size
                end_y = start_y + self.tile_size
                
                # 画像クロップ
                tile_image = image.crop((start_x, start_y, end_x, end_y))
                
                # マスククロップ
                tile_mask = mask[start_y:end_y, start_x:end_x]
                
                # タイル情報作成
                tile_info = {
                    'image': tile_image,
                    'mask': tile_mask,
                    'position': (row, col),
                    'grid_position': f"{row:03d}_{col:03d}",
                    'index': tile_index,
                    'bbox': (start_x, start_y, end_x, end_y)
                }
                
                tiles.append(tile_info)
                tile_index += 1
        
        return tiles
    
    def _filter_tiles(self, tiles: List[Dict]) -> List[Dict]:
        """前景ピクセル比率でタイルをフィルタリング"""
        if self.min_foreground_ratio <= 0:
            return tiles
        
        filtered_tiles = []
        for tile in tiles:
            if self._should_keep_tile(tile['mask']):
                filtered_tiles.append(tile)
        
        return filtered_tiles
    
    def _should_keep_tile(self, mask: np.ndarray) -> bool:
        """タイルを保持するかの判定"""
        total_pixels = mask.size
        foreground_pixels = np.sum(mask > 0)
        ratio = foreground_pixels / total_pixels
        
        return ratio >= self.min_foreground_ratio
    
    def get_statistics(self, tiles: List[Dict]) -> Dict:
        """タイル統計を取得"""
        if not tiles:
            return {'total_tiles': 0}
        
        # 前景ピクセル統計
        foreground_ratios = []
        for tile in tiles:
            mask = tile['mask']
            total_pixels = mask.size
            foreground_pixels = np.sum(mask > 0)
            ratio = foreground_pixels / total_pixels
            foreground_ratios.append(ratio)
        
        return {
            'total_tiles': len(tiles),
            'grid_size': self.grid_size,
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'stride': self.stride,
            'avg_foreground_ratio': np.mean(foreground_ratios),
            'min_foreground_ratio': np.min(foreground_ratios),
            'max_foreground_ratio': np.max(foreground_ratios)
        }
        
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
