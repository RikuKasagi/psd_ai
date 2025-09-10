"""
データセット自動生成システム - メイン実行スクリプト
PSDレイヤーからデータセット生成パイプラインを実行
"""

import argparse
import sys
from pathlib import Path

# プロジェクトパスを追加
sys.path.append(str(Path(__file__).parent))

from utils.psd_tools.psd_split import extract_layers_from_psd
from tools.pipeline import DatasetPipeline


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="PSDファイルからデータセットを自動生成"
    )
    
    # 必須引数
    parser.add_argument(
        'psd_path',
        help="入力PSDファイルパス"
    )
    
    # オプション引数
    parser.add_argument(
        '--config',
        default='./config/build_config.yaml',
        help="パイプライン設定ファイル（デフォルト: ./config/build_config.yaml）"
    )
    
    parser.add_argument(
        '--classes',
        default='./config/classes.yaml',
        help="クラス定義設定ファイル（デフォルト: ./config/classes.yaml）"
    )
    
    parser.add_argument(
        '--output',
        help="出力ディレクトリ（設定ファイルの値を上書き）"
    )
    
    parser.add_argument(
        '--include-hidden',
        action='store_true',
        help="非表示レイヤーも含める"
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="ログレベル（デフォルト: INFO）"
    )
    
    parser.add_argument(
        '--base-name',
        default='dataset',
        help="ベース名（デフォルト: dataset）"
    )
    
    args = parser.parse_args()
    
    try:
        # 1. PSDファイルからレイヤー抽出
        print(f"📂 PSDファイル読み込み: {args.psd_path}")
        layers = extract_layers_from_psd(args.psd_path, args.include_hidden)
        
        if not layers:
            print("❌ レイヤーが抽出できませんでした")
            return 1
        
        print(f"✅ {len(layers)}個のレイヤーを抽出")
        for layer_name in layers.keys():
            print(f"  - {layer_name}")
        
        # 2. パイプライン初期化
        print(f"🔧 パイプライン初期化")
        pipeline = DatasetPipeline(
            config_path=args.config,
            classes_config_path=args.classes,
            log_level=args.log_level
        )
        
        # パイプライン情報表示
        info = pipeline.get_pipeline_info()
        print(f"   タイルサイズ: {info['settings']['tile_size']}")
        print(f"   オーバーラップ: {info['settings']['overlap']}")
        print(f"   分割比率: train={info['settings']['split_ratios']['train']}, "
              f"val={info['settings']['split_ratios']['val']}, "
              f"test={info['settings']['split_ratios']['test']}")
        print(f"   シード: {info['settings']['seed']}")
        
        # 3. データセット生成実行
        print(f"🚀 データセット生成開始")
        result = pipeline.generate_dataset(
            layers=layers,
            output_path=args.output,
            base_name=args.base_name
        )
        
        # 4. 結果表示
        print("=" * 50)
        print("🎉 データセット生成完了！")
        print("=" * 50)
        print(f"実行時間: {result['execution_time_seconds']}秒")
        print(f"生成タイル数: {result['tiles_generated']}")
        print("分割結果:")
        for split_name, count in result['splits'].items():
            print(f"  {split_name}: {count}ファイル")
        
        if result['manifest_saved']:
            print("✅ マニフェスト保存完了")
        else:
            print("⚠️ マニフェスト保存失敗")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ ファイルエラー: {e}")
        return 1
    
    except ValueError as e:
        print(f"❌ 設定エラー: {e}")
        return 1
    
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


def show_example():
    """使用例を表示"""
    examples = """
使用例:

# 基本的な使用
python generate_dataset.py input.psd

# 設定ファイルを指定
python generate_dataset.py input.psd --config my_config.yaml --classes my_classes.yaml

# 出力先を指定
python generate_dataset.py input.psd --output ./my_dataset

# 非表示レイヤーも含める
python generate_dataset.py input.psd --include-hidden

# ログレベルを変更
python generate_dataset.py input.psd --log-level DEBUG

# 全オプション指定
python generate_dataset.py input.psd \\
    --config ./config/build_config.yaml \\
    --classes ./config/classes.yaml \\
    --output ./dataset \\
    --include-hidden \\
    --log-level INFO \\
    --base-name my_dataset
"""
    print(examples)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['--help-examples', '-he']:
        show_example()
    else:
        sys.exit(main())
