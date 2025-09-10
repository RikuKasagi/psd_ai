"""
入力・出力検証ユーティリティ
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image
from pathlib import Path
from datetime import datetime


class Validator:
    """データセット生成の入力・出力を検証"""
    
    @staticmethod
    def validate_input_layers(
        layers: Dict[str, Image.Image], 
        required_layers: List[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        入力レイヤーを検証
        
        Parameters
        ----------
        layers : Dict[str, Image.Image]
            入力レイヤー辞書
        required_layers : List[str]
            必須レイヤー名リスト
            
        Returns
        -------
        Tuple[bool, List[str]]
            (検証結果, 問題点リスト)
        """
        if required_layers is None:
            required_layers = ["original", "mask", "refined"]
        
        issues = []
        
        # 基本チェック
        if not isinstance(layers, dict):
            issues.append("レイヤーが辞書型ではありません")
            return False, issues
        
        if len(layers) == 0:
            issues.append("レイヤーが空です")
            return False, issues
        
        # 必須レイヤーチェック
        for layer_name in required_layers:
            if layer_name not in layers:
                issues.append(f"必須レイヤーが不足: {layer_name}")
                continue
            
            layer_img = layers[layer_name]
            if not isinstance(layer_img, Image.Image):
                issues.append(f"レイヤーが画像オブジェクトではありません: {layer_name}")
                continue
            
            # 画像基本チェック
            if layer_img.size[0] <= 0 or layer_img.size[1] <= 0:
                issues.append(f"レイヤーサイズが不正: {layer_name} {layer_img.size}")
        
        # サイズ統一チェック
        if len([l for l in layers.values() if isinstance(l, Image.Image)]) >= 2:
            sizes = {name: img.size for name, img in layers.items() if isinstance(img, Image.Image)}
            unique_sizes = set(sizes.values())
            
            if len(unique_sizes) > 1:
                issues.append(f"レイヤーサイズが統一されていません: {sizes}")
        
        # フォーマットチェック
        for name, img in layers.items():
            if isinstance(img, Image.Image):
                if img.mode not in ['RGB', 'RGBA', 'L']:
                    issues.append(f"非対応の画像モード: {name} -> {img.mode}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_config(config: Dict) -> Tuple[bool, List[str]]:
        """
        設定ファイルを検証
        
        Parameters
        ----------
        config : Dict
            設定辞書
            
        Returns
        -------
        Tuple[bool, List[str]]
            (検証結果, 問題点リスト)
        """
        issues = []
        
        # 必須セクションチェック
        required_sections = ['tiling', 'data_split', 'output']
        for section in required_sections:
            if section not in config:
                issues.append(f"必須設定セクションが不足: {section}")
        
        # タイル設定チェック
        if 'tiling' in config:
            tiling = config['tiling']
            
            if 'tile_size' not in tiling or tiling['tile_size'] <= 0:
                issues.append("tile_sizeが不正です")
            
            if 'overlap' not in tiling or tiling['overlap'] < 0:
                issues.append("overlapが不正です")
            
            if tiling.get('overlap', 0) >= tiling.get('tile_size', 1):
                issues.append("overlapがtile_size以上です")
        
        # データ分割設定チェック
        if 'data_split' in config:
            split = config['data_split']
            
            ratios = [
                split.get('train_ratio', 0),
                split.get('val_ratio', 0),
                split.get('test_ratio', 0)
            ]
            
            if any(r < 0 or r > 1 for r in ratios):
                issues.append("分割比率が範囲外です（0-1）")
            
            if abs(sum(ratios) - 1.0) > 0.001:
                issues.append(f"分割比率の合計が1.0ではありません: {sum(ratios)}")
            
            if 'seed' not in split:
                issues.append("シード値が設定されていません")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_tile_generation(
        original: Image.Image, 
        mask: np.ndarray,
        tile_size: int,
        overlap: int
    ) -> Tuple[bool, List[str]]:
        """
        タイル生成パラメータを検証
        
        Parameters
        ----------
        original : Image.Image
            元画像
        mask : np.ndarray
            マスク配列
        tile_size : int
            タイルサイズ
        overlap : int
            オーバーラップ
            
        Returns
        -------
        Tuple[bool, List[str]]
            (検証結果, 問題点リスト)
        """
        issues = []
        
        # 画像とマスクのサイズ一致
        if original.size != (mask.shape[1], mask.shape[0]):
            issues.append(f"画像とマスクのサイズが不一致: {original.size} != {mask.shape[:2][::-1]}")
        
        # タイルパラメータ
        if tile_size <= 0:
            issues.append(f"tile_sizeが不正: {tile_size}")
        
        if overlap < 0:
            issues.append(f"overlapが不正: {overlap}")
        
        if overlap >= tile_size:
            issues.append(f"overlapがtile_size以上: {overlap} >= {tile_size}")
        
        # 画像サイズとタイルサイズの関係
        stride = tile_size - overlap
        if stride <= 0:
            issues.append(f"ストライドが不正: {stride}")
        
        min_size = min(original.size)
        if tile_size > min_size * 2:
            issues.append(f"tile_sizeが画像サイズに対して大きすぎます: {tile_size} > {min_size * 2}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_output_dataset(dataset_path: str) -> Tuple[bool, List[str], Dict]:
        """
        生成されたデータセットを検証
        
        Parameters
        ----------
        dataset_path : str
            データセットルートパス
            
        Returns
        -------
        Tuple[bool, List[str], Dict]
            (検証結果, 問題点リスト, 統計情報)
        """
        issues = []
        stats = {
            'train_count': 0,
            'val_count': 0,
            'test_count': 0,
            'missing_masks': [],
            'size_mismatches': [],
            'format_issues': []
        }
        
        dataset_root = Path(dataset_path)
        
        # ディレクトリ構造チェック
        required_dirs = [
            'train/images', 'train/masks',
            'val/images', 'val/masks',
            'test/images', 'test/masks'
        ]
        
        for dir_path in required_dirs:
            full_path = dataset_root / dir_path
            if not full_path.exists():
                issues.append(f"必須ディレクトリが不足: {dir_path}")
        
        # 各分割セットをチェック
        for split in ['train', 'val', 'test']:
            images_dir = dataset_root / split / 'images'
            masks_dir = dataset_root / split / 'masks'
            
            if not images_dir.exists() or not masks_dir.exists():
                continue
            
            # 画像ファイル取得
            image_files = list(images_dir.glob('*.png'))
            mask_files = list(masks_dir.glob('*.png'))
            
            stats[f'{split}_count'] = len(image_files)
            
            # 画像とマスクの対応チェック
            image_names = {f.stem for f in image_files}
            mask_names = {f.stem for f in mask_files}
            
            missing_masks = image_names - mask_names
            if missing_masks:
                stats['missing_masks'].extend([f"{split}:{name}" for name in missing_masks])
            
            # サンプル画像のサイズ・フォーマットチェック
            for img_file in image_files[:5]:  # サンプル数を制限
                try:
                    img = Image.open(img_file)
                    
                    # 対応するマスクファイル
                    mask_file = masks_dir / f"{img_file.stem}.png"
                    if mask_file.exists():
                        mask = Image.open(mask_file)
                        
                        # サイズ一致チェック
                        if img.size != mask.size:
                            stats['size_mismatches'].append(f"{split}:{img_file.stem}")
                        
                        # フォーマットチェック
                        if img.mode not in ['RGB', 'RGBA']:
                            stats['format_issues'].append(f"{split}:{img_file.stem} -> {img.mode}")
                        
                        if mask.mode != 'L':
                            stats['format_issues'].append(f"{split}:{img_file.stem}_mask -> {mask.mode}")
                
                except Exception as e:
                    issues.append(f"ファイル読み込みエラー: {img_file} -> {e}")
        
        # 統計レポート
        total_count = stats['train_count'] + stats['val_count'] + stats['test_count']
        if total_count == 0:
            issues.append("データセットが空です")
        
        if len(stats['missing_masks']) > 0:
            issues.append(f"マスクが不足: {len(stats['missing_masks'])}件")
        
        if len(stats['size_mismatches']) > 0:
            issues.append(f"サイズ不一致: {len(stats['size_mismatches'])}件")
        
        if len(stats['format_issues']) > 0:
            issues.append(f"フォーマット問題: {len(stats['format_issues'])}件")
        
        return len(issues) == 0, issues, stats
    
    @staticmethod
    def generate_validation_report(
        input_validation: Tuple[bool, List[str]],
        config_validation: Tuple[bool, List[str]],
        output_validation: Tuple[bool, List[str], Dict]
    ) -> Dict:
        """
        検証レポートを生成
        
        Parameters
        ----------
        input_validation : Tuple[bool, List[str]]
            入力検証結果
        config_validation : Tuple[bool, List[str]]
            設定検証結果
        output_validation : Tuple[bool, List[str], Dict]
            出力検証結果
            
        Returns
        -------
        Dict
            検証レポート
        """
        return {
            'validation_time': str(datetime.now()),
            'overall_success': all([
                input_validation[0],
                config_validation[0],
                output_validation[0]
            ]),
            'input': {
                'success': input_validation[0],
                'issues': input_validation[1]
            },
            'config': {
                'success': config_validation[0],
                'issues': config_validation[1]
            },
            'output': {
                'success': output_validation[0],
                'issues': output_validation[1],
                'stats': output_validation[2]
            }
        }
