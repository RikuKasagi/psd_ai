# PSD AI

このリポジトリには、PSD処理に関連するファイルが格納されています。

## 📁 ディレクトリ構成

```
psd_ai/
├── README.md
└── utils/
    └── psd_tools/
        ├── psd_split.py    # PSDファイルからレイヤーを抽出
        └── psd_maker.py    # PNG画像からPSDファイルを作成
```

## 🚀 セットアップ

### 必要なライブラリをインストール

```bash
pip install psd-tools pytoshop pillow numpy
```

## 📖 機能説明

### 1. PSDファイルからレイヤーを抽出 (`psd_split.py`)

PSDファイルを読み込み、各レイヤーをPNG画像として抽出します。

**特徴:**
- 元のPSDキャンバスサイズで統一出力
- 透明チャンネル(α)を保持
- グループレイヤーにも対応
- レイヤーの位置情報を保持
- 非表示レイヤーの抽出にも対応

### 2. PNG画像からPSDファイルを作成 (`psd_maker.py`)

複数のPNG画像を重ね合わせて、一つのPSDファイルを作成します。

**特徴:**
- PNG形式のみ対応
- レイヤー名の自動補完
- サイズ一致チェック
- 透明チャンネル対応

---

## 🔧 使用方法

### PSDファイルからレイヤーを抽出する

#### ✅ 基本的な使い方

```python
from utils.psd_tools.psd_split import extract_layers_from_psd

# PSDファイルからレイヤーを抽出（表示レイヤーのみ）
psd_path = "input.psd"
layers = extract_layers_from_psd(psd_path)

# 結果を確認
for layer_name, image in layers.items():
    print(f"レイヤー名: {layer_name}")
    print(f"画像サイズ: {image.size}")
    
    # 個別にPNG保存
    image.save(f"{layer_name}.png")
```

#### ✅ 非表示レイヤーも含めて抽出

```python
from utils.psd_tools.psd_split import extract_layers_from_psd

# 非表示レイヤーも含めて抽出
psd_path = "input.psd"
all_layers = extract_layers_from_psd(psd_path, include_hidden=True)

print(f"抽出されたレイヤー数: {len(all_layers)}")
for layer_name, image in all_layers.items():
    print(f"レイヤー名: {layer_name}")
    image.save(f"all_{layer_name}.png")
```

#### ✅ フォルダに一括保存

```python
import os
from utils.psd_tools.psd_split import extract_layers_from_psd

def save_all_layers(psd_path, output_dir, include_hidden=False):
    """PSDの全レイヤーを指定フォルダに保存"""
    
    # レイヤーを抽出
    layers = extract_layers_from_psd(psd_path, include_hidden=include_hidden)
    
    # 出力フォルダを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 各レイヤーを保存
    for layer_name, image in layers.items():
        # ファイル名として使用できない文字を置換
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in layer_name)
        output_path = os.path.join(output_dir, f"{safe_name}.png")
        image.save(output_path)
        print(f"保存完了: {layer_name} -> {output_path}")

# 使用例
save_all_layers("input.psd", "extracted_layers")  # 表示レイヤーのみ
save_all_layers("input.psd", "all_layers", include_hidden=True)  # 全レイヤー
```

---

### PNG画像からPSDファイルを作成する

#### ✅ 基本的な使い方

```python
from utils.psd_tools.psd_maker import save_images_as_psd

# 重ね合わせる画像のパス（下から順番）
image_paths = [
    "background.png",    # 最下層
    "character.png",     # 中間層
    "effects.png"        # 最上層
]

# レイヤー名
layer_names = [
    "背景",
    "キャラクター", 
    "エフェクト"
]

# PSDとして保存
save_images_as_psd(
    image_paths=image_paths,
    layer_names=layer_names,
    save_path="output.psd"
)
```

#### ✅ レイヤー名の自動補完

```python
# レイヤー名が足りない場合は自動で番号が付きます
image_paths = [
    "layer1.png",
    "layer2.png", 
    "layer3.png"
]

layer_names = ["背景"]  # 1つしか指定していない

# 結果: ["背景", "2", "3"] として処理されます
save_images_as_psd(image_paths, layer_names, "output.psd")
```

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
from utils.psd_tools.psd_split import extract_layers_from_psd
from utils.psd_tools.psd_maker import save_images_as_psd

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

このツールは将来的に大きなシステムに組み込まれる予定ですが、現在は個別の関数として利用できます。