# PSD AI - 高速データセット自動生成システム

PSDファイルから機械学習用データセットを自動生成する高速処理システムです。

## 🚀 主要機能

- **🔥 劇的高速化**: 色マッピング処理を20-200倍高速化（NumPyベクトル化処理）
- **⚡ バッチ処理**: 複数PSDファイルの一括処理
- **🎯 新オーバーラップ仕様**: `stride = tile_size - (overlap × 2)` による正確な重複制御
- **🔄 自動画像変換**: 横長画像を縦向きに自動変換
- **📊 詳細ログ**: 各処理段階の時間計測とボトルネック特定
- **🧹 統一命名**: train/val/test内で00001.png形式の通し番号

## 📁 ディレクトリ構成

```
psd_ai/
├── README.md
├── LICENSE                     # MIT License
├── requirements.txt            # 依存関係
├── setup.py                   # パッケージ設定
├── generate_dataset.py         # 単一PSDファイル用データセット生成
├── generate_batch_dataset.py   # 🆕 複数PSDファイル一括処理
├── __init__.py                # パッケージエントリーポイント
├── config/                    # 設定ファイル
│   ├── classes.yaml           # クラス定義設定
│   └── build_config.yaml      # パイプライン設定
├── tools/                     # データセット生成ツール
│   ├── __init__.py
│   ├── mask_builder.py        # マスク統合
│   ├── tiler.py              # タイル分割（新オーバーラップ仕様）
│   ├── splitter.py           # データ分割
│   ├── pipeline.py           # 単一ファイル用パイプライン
│   └── batch_pipeline.py     # 🆕 バッチ処理パイプライン
├── utils/
│   ├── dataset/              # データセット生成ユーティリティ
│   │   ├── __init__.py
│   │   ├── color_map.py      # 色→クラスIDマッピング
│   │   ├── io_utils.py       # 入出力ユーティリティ
│   │   ├── log.py            # ログ管理
│   │   ├── manifest.py       # マニフェスト生成
│   │   └── validate.py       # 検証
│   └── psd_tools/
│       ├── __init__.py
│       ├── psd_split.py       # PSDファイルからレイヤーを抽出
│       └── psd_maker.py       # PNG画像からPSDファイルを作成
├── dataset/                  # 生成されたデータセット（出力先）
├── manifests/                # 実行メタ情報
└── logs/                     # 実行ログ
```

## 🚀 インストール

### 方法1: pip install（推奨）

```bash
pip install git+https://github.com/RikuKasagi/psd_ai.git
```

### 方法2: 手動インストール

```bash
git clone https://github.com/RikuKasagi/psd_ai.git
cd psd_ai
pip install -r requirements.txt
pip install -e .
```

### 方法3: 依存ライブラリのみインストール

```bash
pip install psd-tools pytoshop pillow numpy
```

## 📖 機能説明

### 🆕 データセット自動生成システム

PSDレイヤーから機械学習用データセットを自動生成する包括的なシステムです。

**🔥 主要機能:**

- **バッチ処理対応**: 複数PSDファイルの一括処理が可能
- **劇的高速化**: 色マッピング処理を20-200倍高速化（NumPyベクトル化処理）
- **新オーバーラップ仕様**: `stride = tile_size - (overlap × 2)` で正確な重複制御
- **横長画像自動変換**: 横長画像を縦向きに自動変換してからタイル分割
- **グリッド分割**: 指定グリッド数で画像をリサイズして均等分割
- **マスク統合**: mask優先でrefinedをマージしたインデックスマスクを生成
- **通し番号命名**: train/val/test内で00001.png形式の通し番号
- **ランダム分割**: 全PSDファイルからタイルを収集後、ランダムにtrain/val/testに分割
- **一時フォルダ管理**: 処理中の一時ファイル管理（削除可/不可選択）
- **メタデータ生成**: manifestとログを自動生成
- **検証機能**: 入力・出力の整合性チェック
- **詳細ログ**: 各処理段階の時間計測とボトルネック特定

### PSDユーティリティ機能

基本的なPSD処理機能も提供しています：

- **レイヤー抽出**: PSDファイルからレイヤーをPNG画像として抽出
- **PSD作成**: 複数PNG画像からPSDファイルを作成

**🔄 処理フロー（バッチ処理）:**

1. 複数PSDファイル検索・読み込み
2. 各PSDからレイヤー抽出（original, mask, refined）
3. マスク統合（優先度: mask > refined）
4. 横長→縦向き変換（auto_orient: true）
5. グリッド分割用リサイズ
6. タイル分割（新オーバーラップ仕様）
7. 全タイル収集・ランダムシャッフル
8. train/val/testランダム分割（通し番号命名）
9. マニフェスト・ログ生成

**💡 オーバーラップ仕様:**
```
tile_size = 512, overlap = 128の場合
├─ 出力タイル: 512×512ピクセル（フルサイズ）
├─ 新規部分: 256×256ピクセル（重複なし部分）
├─ stride: 256ピクセル（移動距離）
└─ 重複: 上下左右に128ピクセルずつ
```

---

## 🔧 使用方法

### 🆕 バッチデータセット生成（メイン機能）

#### ✅ 基本的な使用方法

```bash
# 基本実行（フォルダ内の全PSDファイルを処理）
python generate_batch_dataset.py psd_folder --output batch_dataset

# 設定ファイルを指定
python generate_batch_dataset.py psd_folder --output my_dataset --config my_config.yaml --classes my_classes.yaml

# 非表示レイヤーも含める
python generate_batch_dataset.py psd_folder --output my_dataset --include-hidden

# 詳細ログ出力（処理時間分析）
python generate_batch_dataset.py psd_folder --output my_dataset --log-level DEBUG

# 一時フォルダを残す（デバッグ用）
python generate_batch_dataset.py psd_folder --output my_dataset --keep-temp
```

#### ✅ 出力フォルダ構造

```
my_dataset/
├── train/
│   ├── images/
│   │   ├── 00001.png
│   │   ├── 00002.png
│   │   └── ...
│   └── masks/
│       ├── 00001.png
│       ├── 00002.png
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### データセット自動生成システム（単一ファイル）

#### ✅ 基本的な使用方法

```bash
# 基本実行（デフォルト設定）
python generate_dataset.py input.psd

# 設定ファイルを指定
python generate_dataset.py input.psd --config my_config.yaml --classes my_classes.yaml

# 出力先を指定
python generate_dataset.py input.psd --output ./my_dataset

# 非表示レイヤーも含める
python generate_dataset.py input.psd --include-hidden

# 詳細ログ出力
python generate_dataset.py input.psd --log-level DEBUG
```

#### ✅ 設定ファイルのカスタマイズ

**クラス定義（`config/classes.yaml`）:**
```yaml
# 背景クラス
background:
  id: 0
  name: "background"
  color: [0, 0, 0]
  alpha_threshold: 128

# 前景クラス定義
classes:
  - id: 1
    name: "object1"
    color: [255, 0, 0]  # 赤
  - id: 2
    name: "object2" 
    color: [0, 255, 0]  # 緑
```

**パイプライン設定（`config/build_config.yaml`）:**

```yaml
# タイル分割設定（新仕様）
tiling:
  tile_size: 512          # 正方形タイルサイズ
  grid_size: [18, 5]       # [縦の分割数, 横の分割数]
  overlap: 128            # オーバーラップピクセル数（片側）
  auto_orient: true       # 横長画像を縦向きに変換
  padding:
    mode: "constant"      # constant, reflect, symmetric
    value: 0             # パディング値
  min_foreground_ratio: 0.0  # 前景ピクセル最小比率

# データ分割設定
data_split:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  seed: 42              # ランダムシャッフル用シード値

# 出力設定
output:
  base_path: "./dataset"
  image_format: "PNG"
  mask_format: "PNG"
  compression: 6          # PNG圧縮レベル（0-9）
```

#### ✅ プログラムからの実行

```python
from tools.pipeline import DatasetPipeline
from utils.psd_tools.psd_split import extract_layers_from_psd

# 1. PSDからレイヤー抽出
layers = extract_layers_from_psd("input.psd", include_hidden=True)

# 2. パイプライン初期化
pipeline = DatasetPipeline(
    config_path="./config/build_config.yaml",
    classes_config_path="./config/classes.yaml"
)

# 3. データセット生成
result = pipeline.generate_dataset(
    layers=layers,
    output_path="./dataset"
)

print(f"生成完了: {result['tiles_generated']}タイル")
```

---

### パッケージとしてインポート

```python
# データセット生成の基本的な使用
from tools.batch_pipeline import BatchPipeline

# バッチパイプライン初期化
pipeline = BatchPipeline(
    config_path="./config/build_config.yaml",
    classes_config_path="./config/classes.yaml"
)

# バッチ処理実行
result = pipeline.process_batch(
    psd_folder="./psd_folder",
    output_path="./dataset",
    include_hidden=False,
    keep_temp=False
)

print(f"生成完了: {result['total_tiles_generated']}タイル")
```

### PSDユーティリティ（基本機能）

```python
from psd_ai import extract_layers_from_psd, save_images_as_psd

# PSDファイルからレイヤーを抽出
layers = extract_layers_from_psd("input.psd")

# PNG画像からPSDファイルを作成
save_images_as_psd(
    image_paths=["bg.png", "fg.png"],
    layer_names=["背景", "前景"],
    save_path="output.psd"
)
```

---

## 🎯 **API リファレンス**

### `extract_layers_from_psd(psd_path, include_hidden=False)`

PSDファイルからレイヤーを抽出します。

**パラメータ:**
- `psd_path` (str): PSDファイルのパス
- `include_hidden` (bool): 非表示レイヤーも含めるか（デフォルト: False）

**戻り値:**
- `Dict[str, Image.Image]`: レイヤー名をキーとした辞書

### `save_images_as_psd(image_paths, layer_names, save_path)`

PNG画像からPSDファイルを作成します。

**パラメータ:**
- `image_paths` (List[str]): PNG画像ファイルのパスリスト
- `layer_names` (List[str]): レイヤー名のリスト
- `save_path` (str): 保存先PSDファイルのパス

---

## ⚠️ 注意事項

### PSD分割 (`psd_split.py`)

- ✅ **対応形式**: PSDファイルのみ
- ✅ **出力形式**: PNG（RGBAモード）
- ✅ **出力サイズ**: 元のPSDキャンバスサイズで統一
- ✅ **非表示レイヤー**: `include_hidden=True` で抽出可能

### PSD作成 (`psd_maker.py`)

- ✅ **入力形式**: PNGファイルのみ
- ⚠️ **サイズ制限**: 全ての画像が同じサイズである必要があります
- ⚠️ **レイヤー順**: `image_paths[0]`が最下層になります

---

## 🐛 トラブルシューティング

### よくあるエラーと解決方法

#### 1. `ModuleNotFoundError`
```
解決方法: pip install psd-tools pytoshop pillow numpy
```

#### 2. `FileNotFoundError`
```
原因: 指定したファイルが存在しない
解決方法: ファイルパスを確認してください
```

#### 3. `ValueError: 画像サイズが一致していません`
```
原因: PNG画像のサイズがバラバラ
解決方法: 全ての画像を同じサイズにリサイズしてください
```

#### 4. `ValueError: PNG以外が含まれています`
```
原因: PNG以外の形式の画像が含まれている
解決方法: 全ての画像をPNG形式に変換してください
```

---

## 💡 使用例

### 完全なワークフロー例

```python
import os
from psd_ai import extract_layers_from_psd, save_images_as_psd

# 1. PSDファイルからレイヤーを抽出（非表示レイヤーも含む）
print("📂 PSDファイルからレイヤーを抽出中...")
layers = extract_layers_from_psd("original.psd", include_hidden=True)

# 2. 抽出したレイヤーをPNGとして保存
print("💾 レイヤーをPNGとして保存中...")
temp_dir = "temp_layers"
os.makedirs(temp_dir, exist_ok=True)

png_paths = []
layer_names = []

for layer_name, image in layers.items():
    safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in layer_name)
    png_path = os.path.join(temp_dir, f"{safe_name}.png")
    image.save(png_path)
    png_paths.append(png_path)
    layer_names.append(layer_name)

# 3. PNGファイルから新しいPSDを作成
print("🔄 新しいPSDファイルを作成中...")
save_images_as_psd(
    image_paths=png_paths,
    layer_names=layer_names,
    save_path="recreated.psd"
)

print("✅ 処理完了!")
```

---

## 📝 開発情報

- **読み込み**: `psd-tools`ライブラリを使用（安定性重視）
- **作成**: `pytoshop`ライブラリを使用（機能充実）
- **画像処理**: `PIL (Pillow)`を使用
- **数値処理**: `NumPy`を使用
- **ライセンス**: MIT License
- **Python**: 3.8+ 対応

## 🤝 コントリビューション

1. このリポジトリをフォーク
2. フィーチャーブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. Pull Request を作成

## 📄 ライセンス

このプロジェクトは MIT License の下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。

---

**🚀 新機能**:

- **バッチ処理システム**: 複数PSDファイルの一括処理が可能になりました！
- **新オーバーラップ仕様**: より正確な重複制御
- **横長画像自動変換**: 横長画像を縦向きに自動変換
- **通し番号命名**: train/val/test内で00001.png形式の統一命名
- **ランダム分割**: 全PSDファイルからタイル収集後、完全ランダム分割
- **一時フォルダ管理**: デバッグ用の一時ファイル保持オプション