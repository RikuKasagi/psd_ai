"""
入出力ユーティリティ
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from PIL import Image
import numpy as np
from datetime import datetime


class IOUtils:
    """ファイル入出力を管理するユーティリティクラス"""
    
    @staticmethod
    def ensure_dir(path: str) -> Path:
        """
        ディレクトリが存在することを保証
        
        Parameters
        ----------
        path : str
            ディレクトリパス
            
        Returns
        -------
        Path
            作成されたパスオブジェクト
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def save_image(
        image: Image.Image, 
        filepath: str, 
        format: str = "PNG",
        compression: int = 6
    ) -> bool:
        """
        画像を保存
        
        Parameters
        ----------
        image : Image.Image
            保存する画像
        filepath : str
            保存先パス
        format : str
            画像フォーマット
        compression : int
            圧縮レベル（PNG用）
            
        Returns
        -------
        bool
            保存成功フラグ
        """
        try:
            # ディレクトリ作成
            IOUtils.ensure_dir(os.path.dirname(filepath))
            
            # 保存オプション
            save_kwargs = {}
            if format.upper() == "PNG":
                save_kwargs['optimize'] = True
                save_kwargs['compress_level'] = compression
            
            image.save(filepath, format=format.upper(), **save_kwargs)
            return True
            
        except Exception as e:
            print(f"画像保存エラー: {filepath} -> {e}")
            return False
    
    @staticmethod
    def save_index_mask(
        mask: np.ndarray, 
        filepath: str,
        compression: int = 6
    ) -> bool:
        """
        インデックスマスクを保存
        
        Parameters
        ----------
        mask : np.ndarray
            インデックスマスク（H, W）
        filepath : str
            保存先パス
        compression : int
            圧縮レベル
            
        Returns
        -------
        bool
            保存成功フラグ
        """
        try:
            # ディレクトリ作成
            IOUtils.ensure_dir(os.path.dirname(filepath))
            
            # PIL Imageに変換
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            
            image = Image.fromarray(mask, mode='L')  # グレースケール
            image.save(
                filepath, 
                format="PNG", 
                optimize=True, 
                compress_level=compression
            )
            return True
            
        except Exception as e:
            print(f"マスク保存エラー: {filepath} -> {e}")
            return False
    
    @staticmethod
    def load_image(filepath: str) -> Optional[Image.Image]:
        """
        画像を読み込み
        
        Parameters
        ----------
        filepath : str
            画像ファイルパス
            
        Returns
        -------
        Image.Image or None
            読み込まれた画像
        """
        try:
            if not os.path.exists(filepath):
                print(f"ファイルが見つかりません: {filepath}")
                return None
            
            return Image.open(filepath)
            
        except Exception as e:
            print(f"画像読み込みエラー: {filepath} -> {e}")
            return None
    
    @staticmethod
    def validate_layers(layers: Dict[str, Image.Image], required: List[str]) -> Tuple[bool, List[str]]:
        """
        レイヤー辞書を検証
        
        Parameters
        ----------
        layers : Dict[str, Image.Image]
            レイヤー辞書
        required : List[str]
            必須レイヤー名リスト
            
        Returns
        -------
        Tuple[bool, List[str]]
            (検証結果, エラーメッセージリスト)
        """
        errors = []
        
        # 必須レイヤーチェック
        for layer_name in required:
            if layer_name not in layers:
                errors.append(f"必須レイヤーが不足: {layer_name}")
                continue
            
            if not isinstance(layers[layer_name], Image.Image):
                errors.append(f"レイヤーが画像オブジェクトではありません: {layer_name}")
                continue
        
        # サイズ一致チェック
        if len(layers) >= 2:
            sizes = [(name, img.size) for name, img in layers.items()]
            first_size = sizes[0][1]
            
            for name, size in sizes[1:]:
                if size != first_size:
                    errors.append(f"画像サイズが一致しません: {sizes[0][0]}{first_size} != {name}{size}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def create_dataset_structure(base_path: str) -> Dict[str, Path]:
        """
        データセット出力構造を作成
        
        Parameters
        ----------
        base_path : str
            ベースディレクトリ
            
        Returns
        -------
        Dict[str, Path]
            作成されたディレクトリパス辞書
        """
        base = Path(base_path)
        
        # ディレクトリ構造定義
        dirs = {
            'base': base,
            'train_images': base / 'train' / 'images',
            'train_masks': base / 'train' / 'masks',
            'val_images': base / 'val' / 'images',
            'val_masks': base / 'val' / 'masks',
            'test_images': base / 'test' / 'images',
            'test_masks': base / 'test' / 'masks',
        }
        
        # ディレクトリ作成
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        
        return dirs
    
    @staticmethod
    def convert_transparent_to_background(
        image: Image.Image, 
        background_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> Image.Image:
        """
        透明ピクセルを背景色に変換
        
        Parameters
        ----------
        image : Image.Image
            入力画像
        background_color : Tuple[int, int, int]
            背景色
            
        Returns
        -------
        Image.Image
            変換後画像
        """
        if image.mode != 'RGBA':
            return image
        
        # 背景画像作成
        background = Image.new('RGB', image.size, background_color)
        
        # アルファブレンド
        background.paste(image, mask=image.split()[-1])  # アルファチャンネルをマスクに使用
        
        return background
    
    @staticmethod
    def generate_filename(base_name: str, index: int, extension: str = "png") -> str:
        """
        ファイル名を生成
        
        Parameters
        ----------
        base_name : str
            ベース名
        index : int
            インデックス番号
        extension : str
            拡張子
            
        Returns
        -------
        str
            生成されたファイル名
        """
        return f"{base_name}_{index:06d}.{extension}"
    
    @staticmethod
    def save_json(data: Dict, filepath: str, indent: int = 2) -> bool:
        """
        JSON ファイルを保存
        
        Parameters
        ----------
        data : Dict
            保存するデータ
        filepath : str
            保存先パス
        indent : int
            インデント数
            
        Returns
        -------
        bool
            保存成功フラグ
        """
        try:
            IOUtils.ensure_dir(os.path.dirname(filepath))
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
            
            return True
            
        except Exception as e:
            print(f"JSON保存エラー: {filepath} -> {e}")
            return False
