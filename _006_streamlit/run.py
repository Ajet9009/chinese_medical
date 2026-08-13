#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Streamlit 启动器。

用法:
    python _006_streamlit/run.py          # 默认 8501 端口
    python _006_streamlit/run.py 8502     # 指定端口
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "app.py")

if not os.path.exists(APP):
    print(f"[错误] 找不到 {APP}")
    sys.exit(1)

port = sys.argv[1] if len(sys.argv) > 1 else "8501"

print(f" Streamlit 启动中...")
print(f"   页面: {APP}")
print(f"   端口: {port}")
print(f"   访问: http://localhost:{port}")
print()

subprocess.run([
    sys.executable, "-m", "streamlit", "run", APP,
    "--server.port", port,
    "--server.address", "0.0.0.0",
])
