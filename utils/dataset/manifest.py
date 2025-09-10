"""
マニフェスト生成ユーティリティ
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import hashlib


class ManifestGenerator:
    """データセット生成のマニフェストを管理"""
    
    def __init__(self):
        self.manifest = {
            'version': '1.0',
            'generated_at': None,
            'config': {},
            'input': {},
            'processing': {},
            'output': {},
            'stats': {},
            'validation': {}
        }
    
    def set_generation_time(self, timestamp: Optional[datetime] = None):
        """生成時刻を設定"""
        if timestamp is None:
            timestamp = datetime.now()
        self.manifest['generated_at'] = timestamp.isoformat()
    
    def set_config(self, config: Dict):
        """設定情報を記録"""
        self.manifest['config'] = {
            'tiling': config.get('tiling', {}),
            'data_split': config.get('data_split', {}),
            'output': config.get('output', {}),
            'quality': config.get('quality', {})
        }
    
    def set_input_info(self, layers: Dict, input_path: Optional[str] = None):
        """入力情報を記録"""
        self.manifest['input'] = {
            'source_path': input_path,
            'layers': list(layers.keys()),
            'layer_info': {}
        }
        
        # 各レイヤーの情報
        for name, image in layers.items():
            if hasattr(image, 'size') and hasattr(image, 'mode'):
                self.manifest['input']['layer_info'][name] = {
                    'size': image.size,
                    'mode': image.mode,
                    'channels': len(image.getbands()) if hasattr(image, 'getbands') else None
                }
    
    def set_processing_info(
        self,
        tile_count: int,
        original_size: tuple,
        tile_size: int,
        overlap: int,
        stride: int
    ):
        """処理情報を記録"""
        self.manifest['processing'] = {
            'original_size': original_size,
            'tile_size': tile_size,
            'overlap': overlap,
            'stride': stride,
            'total_tiles_generated': tile_count,
            'grid_size': self._calculate_grid_size(original_size, tile_size, stride)
        }
    
    def set_output_info(self, output_path: str, split_results: Dict[str, List[str]]):
        """出力情報を記録"""
        self.manifest['output'] = {
            'dataset_path': output_path,
            'splits': {}
        }
        
        # 各分割の情報
        for split_name, file_list in split_results.items():
            self.manifest['output']['splits'][split_name] = {
                'count': len(file_list),
                'files': file_list[:10] if len(file_list) > 10 else file_list,  # サンプル
                'total_files': len(file_list)
            }
    
    def set_stats(self, stats: Dict):
        """統計情報を記録"""
        self.manifest['stats'] = stats
    
    def set_validation_results(self, validation_results: Dict):
        """検証結果を記録"""
        self.manifest['validation'] = validation_results
    
    def add_class_mapping(self, class_mapping: Dict):
        """クラスマッピング情報を追加"""
        self.manifest['class_mapping'] = class_mapping
    
    def add_processing_time(self, start_time: datetime, end_time: datetime):
        """処理時間を追加"""
        duration = (end_time - start_time).total_seconds()
        self.manifest['processing']['duration_seconds'] = duration
        self.manifest['processing']['start_time'] = start_time.isoformat()
        self.manifest['processing']['end_time'] = end_time.isoformat()
    
    def add_error_log(self, errors: List[str]):
        """エラーログを追加"""
        if 'errors' not in self.manifest:
            self.manifest['errors'] = []
        self.manifest['errors'].extend(errors)
    
    def _calculate_grid_size(self, image_size: tuple, tile_size: int, stride: int) -> tuple:
        """グリッドサイズを計算"""
        width, height = image_size
        
        # タイル数を計算
        tiles_x = (width - tile_size) // stride + 1
        tiles_y = (height - tile_size) // stride + 1
        
        # パディングが必要な場合を考慮
        if (width - tile_size) % stride != 0:
            tiles_x += 1
        if (height - tile_size) % stride != 0:
            tiles_y += 1
        
        return (tiles_x, tiles_y)
    
    def generate_file_hash(self, filepath: str) -> Optional[str]:
        """ファイルのハッシュ値を生成"""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return None
    
    def save(self, filepath: str) -> bool:
        """マニフェストを保存"""
        try:
            # パスの正規化
            filepath = os.path.abspath(filepath)
            
            # ディレクトリパスの確定
            if os.path.isdir(filepath) or not filepath.endswith('.json'):
                # ディレクトリパスまたは拡張子なしの場合
                dir_path = filepath if os.path.isdir(filepath) else os.path.dirname(filepath)
                
                # ファイル名に生成時刻を含める
                if self.manifest['generated_at']:
                    timestamp = self.manifest['generated_at'].replace(':', '').replace('-', '').split('T')[0]
                    base_name = f"manifest_{timestamp}"
                else:
                    base_name = "manifest"
                
                filepath = os.path.join(dir_path, f"{base_name}.json")
            
            # ディレクトリ作成
            dir_path = os.path.dirname(filepath)
            os.makedirs(dir_path, exist_ok=True)
            
            # 保存前の検証
            if not self.manifest:
                raise ValueError("マニフェストデータが空です")
            
            # 保存
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.manifest, f, indent=2, ensure_ascii=False, default=str)
            
            # 保存確認
            if not os.path.exists(filepath):
                raise RuntimeError(f"ファイル保存後に存在確認失敗: {filepath}")
            
            print(f"マニフェスト保存成功: {filepath}")
            return True
            
        except Exception as e:
            import traceback
            print(f"マニフェスト保存エラー: {e}")
            print(f"詳細: {traceback.format_exc()}")
            return False
    
    def load(self, filepath: str) -> bool:
        """マニフェストを読み込み"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.manifest = json.load(f)
            return True
        except Exception as e:
            print(f"マニフェスト読み込みエラー: {e}")
            return False
    
    def get_summary(self) -> Dict:
        """サマリー情報を取得"""
        summary = {
            'version': self.manifest.get('version'),
            'generated_at': self.manifest.get('generated_at'),
            'total_tiles': self.manifest.get('processing', {}).get('total_tiles_generated', 0),
            'splits': {}
        }
        
        # 分割情報サマリー
        splits = self.manifest.get('output', {}).get('splits', {})
        for split_name, split_info in splits.items():
            summary['splits'][split_name] = split_info.get('count', 0)
        
        return summary
    
    def validate_manifest(self) -> tuple[bool, List[str]]:
        """マニフェストの整合性を検証"""
        issues = []
        
        # 必須フィールドチェック
        required_fields = ['version', 'generated_at', 'config', 'processing', 'output']
        for field in required_fields:
            if field not in self.manifest:
                issues.append(f"必須フィールドが不足: {field}")
        
        # 処理情報の整合性チェック
        processing = self.manifest.get('processing', {})
        if 'total_tiles_generated' in processing:
            total_generated = processing['total_tiles_generated']
            
            # 分割数との整合性
            splits = self.manifest.get('output', {}).get('splits', {})
            total_split = sum(split.get('count', 0) for split in splits.values())
            
            if total_generated != total_split:
                issues.append(f"生成タイル数と分割数が不一致: {total_generated} != {total_split}")
        
        return len(issues) == 0, issues
