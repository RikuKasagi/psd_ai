"""
色→クラスIDマッピングユーティリティ
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image
import yaml
from pathlib import Path


class ColorMapper:
    """色からクラスIDへのマッピングを管理"""
    
    def __init__(self, config_path: str):
        """
        Parameters
        ----------
        config_path : str
            クラス定義設定ファイルパス
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_mapping()
    
    def _load_config(self) -> Dict:
        """設定ファイルを読み込み"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_mapping(self):
        """色→クラスIDマッピングを設定"""
        self.color_to_id = {}
        self.id_to_info = {}
        
        # 背景クラス
        bg = self.config['background']
        bg_color = tuple(bg['color'])
        self.color_to_id[bg_color] = bg['id']
        self.id_to_info[bg['id']] = {
            'name': bg['name'],
            'color': bg_color,
            'alpha_threshold': bg.get('alpha_threshold', 128)
        }
        
        # 前景クラス
        for cls in self.config['classes']:
            color = tuple(cls['color'])
            self.color_to_id[color] = cls['id']
            self.id_to_info[cls['id']] = {
                'name': cls['name'],
                'color': color,
                'description': cls.get('description', '')
            }
        
        self.background_id = bg['id']
        self.alpha_threshold = bg.get('alpha_threshold', 128)
        self.matching_mode = self.config.get('matching', {}).get('mode', 'exact')
    
    def map_color_to_id(self, color: Tuple[int, int, int], alpha: Optional[int] = None) -> int:
        """
        色をクラスIDにマッピング
        
        Parameters
        ----------
        color : Tuple[int, int, int]
            RGB色値
        alpha : int, optional
            アルファ値
            
        Returns
        -------
        int
            クラスID
        """
        # アルファ値による背景判定
        if alpha is not None and alpha <= self.alpha_threshold:
            return self.background_id
        
        # 厳密一致モード
        if self.matching_mode == "exact":
            return self.color_to_id.get(color, self.background_id)
        
        # 将来の拡張ポイント：しきい値・クラスタリング
        # elif self.matching_mode == "threshold":
        #     return self._find_nearest_color(color)
        # elif self.matching_mode == "clustering":
        #     return self._cluster_color(color)
        
        return self.background_id
    
    def convert_image_to_index_mask(self, image: Image.Image) -> np.ndarray:
        """
        画像をインデックスマスクに変換
        
        Parameters
        ----------
        image : Image.Image
            入力画像（RGB/RGBA）
            
        Returns
        -------
        np.ndarray
            インデックスマスク（H, W）
        """
        # RGBA変換
        if image.mode != 'RGBA':
            if image.mode == 'RGB':
                # RGB画像の場合、アルファチャンネルを追加
                image = image.convert('RGBA')
            else:
                raise ValueError(f"サポートされていない画像モード: {image.mode}")
        
        # NumPy配列に変換
        img_array = np.array(image)  # (H, W, 4)
        height, width = img_array.shape[:2]
        
        # インデックスマスク初期化
        index_mask = np.full((height, width), self.background_id, dtype=np.uint8)
        
        # ピクセルごとにマッピング
        for y in range(height):
            for x in range(width):
                r, g, b, a = img_array[y, x]
                color = (int(r), int(g), int(b))
                class_id = self.map_color_to_id(color, int(a))
                index_mask[y, x] = class_id
        
        return index_mask
    
    def get_class_info(self, class_id: int) -> Dict:
        """
        クラス情報を取得
        
        Parameters
        ----------
        class_id : int
            クラスID
            
        Returns
        -------
        Dict
            クラス情報
        """
        return self.id_to_info.get(class_id, {})
    
    def get_all_classes(self) -> Dict[int, Dict]:
        """
        全クラス情報を取得
        
        Returns
        -------
        Dict[int, Dict]
            クラスID→情報の辞書
        """
        return self.id_to_info.copy()
    
    def validate_image(self, image: Image.Image) -> Tuple[bool, List[str]]:
        """
        画像の色が定義済みクラスに適合するかチェック
        
        Parameters
        ----------
        image : Image.Image
            チェック対象画像
            
        Returns
        -------
        Tuple[bool, List[str]]
            (適合フラグ, 警告メッセージリスト)
        """
        warnings = []
        
        if image.mode not in ['RGB', 'RGBA']:
            warnings.append(f"非対応の画像モード: {image.mode}")
            return False, warnings
        
        # サンプリングによる色チェック（全ピクセルは重いため）
        img_array = np.array(image)
        if len(img_array.shape) < 3:
            warnings.append("色情報が不足しています")
            return False, warnings
        
        # ユニーク色を取得（サンプリング）
        h, w = img_array.shape[:2]
        step = max(1, min(h, w) // 100)  # 適度なサンプリング
        sample_pixels = img_array[::step, ::step]
        
        unique_colors = set()
        for row in sample_pixels:
            for pixel in row:
                if len(pixel) >= 3:
                    color = tuple(pixel[:3])
                    unique_colors.add(color)
        
        # 未定義色をチェック
        undefined_colors = []
        for color in unique_colors:
            if color not in self.color_to_id:
                undefined_colors.append(color)
        
        if undefined_colors:
            warnings.append(f"未定義の色が見つかりました: {len(undefined_colors)}色")
            if len(undefined_colors) <= 5:
                warnings.append(f"例: {undefined_colors}")
        
        return len(undefined_colors) == 0, warnings
