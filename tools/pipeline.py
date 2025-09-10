"""
データセット生成パイプライン
一括実行（上記コンポーネントを順次呼び出す）
"""

import yaml
from datetime import datetime
from typing import Dict, Optional, List
from pathlib import Path
import logging

from .mask_builder import MaskBuilder
from .tiler import Tiler
from .splitter import Splitter
from ..dataset import (
    setup_logger, get_logger, ColorMapper, IOUtils, 
    Validator, ManifestGenerator
)


class DatasetPipeline:
    """データセット生成パイプライン"""
    
    def __init__(
        self, 
        config_path: str, 
        classes_config_path: str,
        log_level: str = "INFO"
    ):
        """
        Parameters
        ----------
        config_path : str
            パイプライン設定ファイルパス
        classes_config_path : str
            クラス定義設定ファイルパス
        log_level : str
            ログレベル
        """
        # 設定読み込み
        self.config = self._load_config(config_path)
        self.classes_config_path = classes_config_path
        
        # ログ設定
        log_config = self.config.get('logging', {})
        self.logger = setup_logger(
            level=log_level,
            log_dir=log_config.get('log_dir', './logs'),
            file_rotation=log_config.get('file_rotation', True),
            max_size_mb=log_config.get('max_size_mb', 10),
            backup_count=log_config.get('backup_count', 5)
        )
        
        # コンポーネント初期化
        self.color_mapper = ColorMapper(classes_config_path)
        self.mask_builder = MaskBuilder(self.color_mapper)
        self.tiler = Tiler(self.config.get('tiling', {}))
        self.splitter = Splitter(self.config.get('data_split', {}))
        self.validator = Validator()
        self.manifest = ManifestGenerator()
        
        # 実行時情報
        self.start_time = None
        self.end_time = None
        self.errors = []
        
        self.logger.info("=== データセット生成パイプライン初期化完了 ===")
    
    def _load_config(self, config_path: str) -> Dict:
        """設定ファイルを読み込み"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"設定ファイル読み込みエラー: {config_path} -> {e}")
    
    def generate_dataset(
        self, 
        layers: Dict, 
        output_path: Optional[str] = None,
        base_name: str = "dataset"
    ) -> Dict:
        """
        データセットを生成
        
        Parameters
        ----------
        layers : Dict[str, Image.Image]
            入力レイヤー辞書
        output_path : str, optional
            出力パス（設定ファイルから取得する場合はNone）
        base_name : str
            ベース名
            
        Returns
        -------
        Dict
            生成結果サマリー
        """
        self.start_time = datetime.now()
        self.logger.info("=== データセット生成開始 ===")
        
        try:
            # 1. 入力検証
            self.logger.info("1. 入力検証")
            validation_result = self._validate_inputs(layers)
            
            # 2. マスク統合
            self.logger.info("2. マスク統合")
            index_mask = self._build_index_mask(layers)
            
            # 3. タイル分割
            self.logger.info("3. タイル分割")
            tiles = self._generate_tiles(layers, index_mask)
            
            # 4. データ分割・保存
            self.logger.info("4. データ分割・保存")
            output_path = output_path or self.config.get('output', {}).get('base_path', './dataset')
            split_results = self._split_and_save(tiles, output_path)
            
            # 5. 出力検証
            self.logger.info("5. 出力検証")
            output_validation = self._validate_outputs(output_path)
            
            # 6. マニフェスト生成
            self.logger.info("6. マニフェスト生成")
            manifest_result = self._generate_manifest(
                layers, tiles, split_results, 
                validation_result, output_validation, output_path
            )
            
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            
            # 7. 結果サマリー
            summary = self._generate_summary(
                len(tiles), split_results, duration, manifest_result
            )
            
            self.logger.info("=== データセット生成完了 ===")
            self.logger.info(f"実行時間: {duration:.1f}秒")
            
            return summary
            
        except Exception as e:
            self.end_time = datetime.now()
            self.errors.append(str(e))
            self.logger.error(f"パイプライン実行エラー: {e}")
            
            # エラー時もマニフェストを生成
            try:
                self._generate_error_manifest(layers, output_path or './dataset')
            except:
                pass
            
            raise
    
    def _validate_inputs(self, layers: Dict) -> Dict:
        """入力を検証"""
        # レイヤー検証
        required_layers = self.config.get('input', {}).get('required_layers', ['original', 'mask', 'refined'])
        layer_valid, layer_issues = IOUtils.validate_layers(layers, required_layers)
        
        # 設定検証
        config_valid, config_issues = self.validator.validate_config(self.config)
        
        # マスクビルダー検証
        mask_valid, mask_issues = self.mask_builder.validate_input_layers(layers)
        
        validation_result = {
            'layer_validation': {'valid': layer_valid, 'issues': layer_issues},
            'config_validation': {'valid': config_valid, 'issues': config_issues},
            'mask_validation': {'valid': mask_valid, 'issues': mask_issues},
            'overall_valid': layer_valid and config_valid and mask_valid
        }
        
        # 問題があれば警告/エラー
        all_issues = layer_issues + config_issues + mask_issues
        for issue in all_issues:
            if validation_result['overall_valid']:
                self.logger.warning(f"検証警告: {issue}")
            else:
                self.logger.error(f"検証エラー: {issue}")
        
        if not validation_result['overall_valid']:
            raise ValueError(f"入力検証失敗: {len(all_issues)}件の問題")
        
        return validation_result
    
    def _build_index_mask(self, layers: Dict):
        """インデックスマスクを構築"""
        priority_order = self.config.get('mask_integration', {}).get('priority_order', ['mask', 'refined'])
        
        # プレビュー情報をログ出力
        preview = self.mask_builder.preview_mask_integration(layers, priority_order)
        self.logger.info(f"マスク統合プレビュー: {preview['processing_order']}")
        
        # マスク統合実行
        index_mask = self.mask_builder.build_index_mask(layers, priority_order)
        
        return index_mask
    
    def _generate_tiles(self, layers: Dict, index_mask):
        """タイルを生成"""
        original = layers['original']
        
        # タイル生成
        tiles = self.tiler.generate_tiles(original, index_mask)
        
        # 統計をログ出力
        stats = self.tiler.get_statistics(tiles)
        self.logger.info(f"タイル統計: {stats['total_tiles']}個生成, パディング含む: {stats['padded_tiles']}個")
        
        return tiles
    
    def _split_and_save(self, tiles: List[Dict], output_path: str):
        """データを分割・保存"""
        # 出力ディレクトリ構造を作成
        IOUtils.create_dataset_structure(output_path)
        
        # データ分割・保存
        split_results = self.splitter.split_tiles(tiles, output_path)
        
        # 分割結果検証
        split_valid, split_issues = self.splitter.validate_split_results(split_results)
        if not split_valid:
            for issue in split_issues:
                self.logger.warning(f"分割警告: {issue}")
        
        return split_results
    
    def _validate_outputs(self, output_path: str):
        """出力を検証"""
        output_valid, output_issues, output_stats = self.validator.validate_output_dataset(output_path)
        
        if not output_valid:
            for issue in output_issues:
                self.logger.warning(f"出力検証警告: {issue}")
        
        self.logger.info(f"出力統計: {output_stats}")
        
        return {
            'valid': output_valid,
            'issues': output_issues,
            'stats': output_stats
        }
    
    def _generate_manifest(
        self, 
        layers: Dict, 
        tiles: List[Dict], 
        split_results: Dict,
        validation_result: Dict,
        output_validation: Dict,
        output_path: str
    ):
        """マニフェストを生成"""
        # 基本情報設定
        self.manifest.set_generation_time(self.start_time)
        self.manifest.set_config(self.config)
        self.manifest.set_input_info(layers)
        
        # 処理情報設定
        if 'original' in layers:
            original_size = layers['original'].size
            self.manifest.set_processing_info(
                tile_count=len(tiles),
                original_size=original_size,
                tile_size=self.tiler.tile_size,
                overlap=self.tiler.overlap,
                stride=self.tiler.stride
            )
        
        # 出力情報設定
        self.manifest.set_output_info(output_path, split_results)
        
        # 統計・検証結果設定
        tile_stats = self.tiler.get_statistics(tiles)
        split_stats = self.splitter.get_split_statistics(split_results)
        
        self.manifest.set_stats({
            'tiles': tile_stats,
            'splits': split_stats,
            'output': output_validation.get('stats', {})
        })
        
        # 検証結果
        self.manifest.set_validation_results({
            'input': validation_result,
            'output': output_validation
        })
        
        # クラスマッピング
        self.manifest.add_class_mapping(self.color_mapper.get_all_classes())
        
        # 処理時間
        if self.start_time and self.end_time:
            self.manifest.add_processing_time(self.start_time, self.end_time)
        
        # エラーログ
        if self.errors:
            self.manifest.add_error_log(self.errors)
        
        # 保存
        manifest_dir = Path(output_path).parent / 'manifests'
        success = self.manifest.save(str(manifest_dir))
        
        if success:
            self.logger.info(f"マニフェスト保存完了: {manifest_dir}")
        else:
            self.logger.error("マニフェスト保存失敗")
        
        return {'saved': success, 'path': str(manifest_dir)}
    
    def _generate_error_manifest(self, layers: Dict, output_path: str):
        """エラー時のマニフェスト生成"""
        self.manifest.set_generation_time(self.start_time)
        self.manifest.set_config(self.config)
        
        if layers:
            self.manifest.set_input_info(layers)
        
        if self.errors:
            self.manifest.add_error_log(self.errors)
        
        if self.start_time and self.end_time:
            self.manifest.add_processing_time(self.start_time, self.end_time)
        
        manifest_dir = Path(output_path).parent / 'manifests'
        self.manifest.save(str(manifest_dir))
    
    def _generate_summary(
        self, 
        tile_count: int, 
        split_results: Dict, 
        duration: float,
        manifest_result: Dict
    ) -> Dict:
        """結果サマリーを生成"""
        summary = {
            'success': True,
            'execution_time_seconds': round(duration, 2),
            'tiles_generated': tile_count,
            'splits': {name: len(files) for name, files in split_results.items()},
            'manifest_saved': manifest_result.get('saved', False),
            'config': {
                'tile_size': self.tiler.tile_size,
                'overlap': self.tiler.overlap,
                'split_ratios': {
                    'train': self.splitter.train_ratio,
                    'val': self.splitter.val_ratio,
                    'test': self.splitter.test_ratio
                },
                'seed': self.splitter.seed
            }
        }
        
        return summary
    
    def get_pipeline_info(self) -> Dict:
        """パイプライン情報を取得"""
        return {
            'config_loaded': bool(self.config),
            'components': {
                'color_mapper': bool(self.color_mapper),
                'mask_builder': bool(self.mask_builder),
                'tiler': bool(self.tiler),
                'splitter': bool(self.splitter),
                'validator': bool(self.validator)
            },
            'settings': {
                'tile_size': getattr(self.tiler, 'tile_size', None),
                'overlap': getattr(self.tiler, 'overlap', None),
                'split_ratios': {
                    'train': getattr(self.splitter, 'train_ratio', None),
                    'val': getattr(self.splitter, 'val_ratio', None),
                    'test': getattr(self.splitter, 'test_ratio', None)
                },
                'seed': getattr(self.splitter, 'seed', None)
            }
        }
