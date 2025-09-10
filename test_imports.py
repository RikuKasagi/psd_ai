#!/usr/bin/env python3
"""
インポートテスト用スクリプト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print(f"Python version: {sys.version}")
print(f"Project root: {project_root}")
print(f"Python path:")
for p in sys.path[:3]:
    print(f"  {p}")

try:
    print("\n1. PSD tools import test...")
    from utils.psd_tools.psd_split import extract_layers_from_psd
    print("✅ PSD tools imported successfully")
except Exception as e:
    print(f"❌ PSD tools import failed: {e}")

try:
    print("\n2. Dataset utils import test...")
    from utils.dataset.log import setup_logger
    from utils.dataset.color_map import ColorMapper
    from utils.dataset.io_utils import IOUtils
    print("✅ Dataset utils imported successfully")
except Exception as e:
    print(f"❌ Dataset utils import failed: {e}")

try:
    print("\n3. Tools import test...")
    from tools.mask_builder import MaskBuilder
    from tools.tiler import Tiler
    from tools.splitter import Splitter
    from tools.pipeline import DatasetPipeline
    print("✅ Tools imported successfully")
except Exception as e:
    print(f"❌ Tools import failed: {e}")

print("\n4. Config files check...")
config_path = project_root / "config" / "build_config.yaml"
classes_path = project_root / "config" / "classes.yaml"
print(f"Config exists: {config_path.exists()}")
print(f"Classes exists: {classes_path.exists()}")

print("\n✅ Import test completed!")
