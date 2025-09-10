"""
マスク統合ツール
mask優先でrefinedをマージしたインデックスマスクを生成
"""

import numpy as np
from typing import Dict, Optional
from PIL import Image
import logging

from utils.dataset.color_map import ColorMapper
from utils.dataset.log import get_logger


class MaskBuilder:
    """マスク統合クラス"""
    
    def __init__(self, color_mapper: ColorMapper):
        """
        Parameters
        ----------
        color_mapper : ColorMapper
            色→クラスIDマッピング
        """
        self.color_mapper = color_mapper
        self.logger = get_logger()
    
    def build_index_mask(
        self, 
        layers: Dict[str, Image.Image],
        priority_order: Optional[list] = None
    ) -> np.ndarray:
        """
        レイヤーからインデックスマスクを生成
        
        Parameters
        ----------
        layers : Dict[str, Image.Image]
            入力レイヤー辞書
        priority_order : list, optional
            優先順位（デフォルト: ["mask", "refined"]）
            
        Returns
        -------
        np.ndarray
            インデックスマスク（H, W）
        """
        if priority_order is None:
            priority_order = ["mask", "refined"]
        
        # 画像サイズを取得（originalから）
        if "original" not in layers:
            raise ValueError("originalレイヤーが必要です")
        
        original = layers["original"]
        width, height = original.size
        
        # インデックスマスクを初期化（背景クラス）
        index_mask = np.full((height, width), self.color_mapper.background_id, dtype=np.uint8)
        
        self.logger.info(f"インデックスマスク初期化: {width}x{height}, 背景ID={self.color_mapper.background_id}")
        
        # 優先順位に従ってレイヤーを統合
        processed_layers = []
        for layer_name in reversed(priority_order):  # 低優先度から処理
            if layer_name in layers:
                layer_img = layers[layer_name]
                self._merge_layer_to_mask(index_mask, layer_img, layer_name)
                processed_layers.append(layer_name)
        
        # 統計情報をログ出力
        self._log_mask_stats(index_mask, processed_layers)
        
        return index_mask
    
    def _merge_layer_to_mask(
        self, 
        index_mask: np.ndarray, 
        layer_img: Image.Image, 
        layer_name: str
    ):
        """
        レイヤーをインデックスマスクにマージ
        
        Parameters
        ----------
        index_mask : np.ndarray
            マージ先インデックスマスク
        layer_img : Image.Image
            マージするレイヤー画像
        layer_name : str
            レイヤー名（ログ用）
        """
        # サイズチェック
        if layer_img.size != (index_mask.shape[1], index_mask.shape[0]):
            self.logger.warning(
                f"レイヤー '{layer_name}' のサイズが不一致: "
                f"{layer_img.size} != {(index_mask.shape[1], index_mask.shape[0])}"
            )
            # リサイズして続行
            layer_img = layer_img.resize((index_mask.shape[1], index_mask.shape[0]), Image.NEAREST)
        
        # レイヤーをインデックス配列に変換
        layer_index_mask = self.color_mapper.convert_image_to_index_mask(layer_img)
        
        # 非背景ピクセルのみマージ（上書き）
        non_background_mask = layer_index_mask != self.color_mapper.background_id
        index_mask[non_background_mask] = layer_index_mask[non_background_mask]
        
        # 統計
        merged_pixels = np.sum(non_background_mask)
        total_pixels = index_mask.size
        percentage = (merged_pixels / total_pixels) * 100
        
        self.logger.info(
            f"レイヤー '{layer_name}' をマージ: "
            f"{merged_pixels:,}ピクセル ({percentage:.1f}%) を更新"
        )
    
    def _log_mask_stats(self, index_mask: np.ndarray, processed_layers: list):
        """マスク統計をログ出力"""
        unique_ids, counts = np.unique(index_mask, return_counts=True)
        total_pixels = index_mask.size
        
        self.logger.info("=== マスク統合完了 ===")
        self.logger.info(f"処理レイヤー: {', '.join(processed_layers)}")
        self.logger.info(f"総ピクセル数: {total_pixels:,}")
        
        for class_id, count in zip(unique_ids, counts):
            class_info = self.color_mapper.get_class_info(class_id)
            class_name = class_info.get('name', f'Unknown_{class_id}')
            percentage = (count / total_pixels) * 100
            
            self.logger.info(f"  クラス {class_id} ({class_name}): {count:,}ピクセル ({percentage:.1f}%)")
    
    def validate_input_layers(self, layers: Dict[str, Image.Image]) -> tuple[bool, list]:
        """
        入力レイヤーを検証
        
        Parameters
        ----------
        layers : Dict[str, Image.Image]
            入力レイヤー辞書
            
        Returns
        -------
        tuple[bool, list]
            (検証結果, 問題点リスト)
        """
        issues = []
        
        # 必須レイヤーチェック
        required = ["original"]
        for layer_name in required:
            if layer_name not in layers:
                issues.append(f"必須レイヤーが不足: {layer_name}")
        
        # アノテーションレイヤーチェック
        annotation_layers = ["mask", "refined"]
        available_annotations = [name for name in annotation_layers if name in layers]
        
        if not available_annotations:
            issues.append("アノテーションレイヤー（mask または refined）が必要です")
        
        # サイズ統一チェック
        if "original" in layers:
            original_size = layers["original"].size
            
            for name, img in layers.items():
                if img.size != original_size:
                    issues.append(f"レイヤー '{name}' のサイズが不一致: {img.size} != {original_size}")
        
        # 色検証（サンプリング）
        for name in annotation_layers:
            if name in layers:
                is_valid, warnings = self.color_mapper.validate_image(layers[name])
                if not is_valid:
                    issues.extend([f"レイヤー '{name}': {w}" for w in warnings])
        
        return len(issues) == 0, issues
    
    def preview_mask_integration(
        self, 
        layers: Dict[str, Image.Image],
        priority_order: Optional[list] = None
    ) -> Dict[str, any]:
        """
        マスク統合の事前プレビュー（統計のみ）
        
        Parameters
        ----------
        layers : Dict[str, Image.Image]
            入力レイヤー辞書
        priority_order : list, optional
            優先順位
            
        Returns
        -------
        Dict[str, any]
            プレビュー情報
        """
        if priority_order is None:
            priority_order = ["mask", "refined"]
        
        preview = {
            'available_layers': list(layers.keys()),
            'processing_order': [name for name in priority_order if name in layers],
            'layer_stats': {}
        }
        
        # 各レイヤーの統計
        for name, img in layers.items():
            if name in priority_order:
                # 色分布を分析
                layer_mask = self.color_mapper.convert_image_to_index_mask(img)
                unique_ids, counts = np.unique(layer_mask, return_counts=True)
                
                stats = {}
                total_pixels = layer_mask.size
                
                for class_id, count in zip(unique_ids, counts):
                    class_info = self.color_mapper.get_class_info(class_id)
                    class_name = class_info.get('name', f'Unknown_{class_id}')
                    percentage = (count / total_pixels) * 100
                    
                    stats[class_name] = {
                        'pixel_count': int(count),
                        'percentage': round(percentage, 2)
                    }
                
                preview['layer_stats'][name] = stats
        
        return preview
