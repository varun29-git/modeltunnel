#!/bin/bash
# Bootstrap an AL2023 prep VM for ModelTunnelv2 corpus preparation (in-region).
set -euxo pipefail
sudo dnf install -y git python3.11 python3.11-devel gcc
python3.11 -m venv ~/venv
~/venv/bin/pip install -q --upgrade pip
~/venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
~/venv/bin/pip install -q "numpy>=1.26,<3" pyarrow boto3 "regex>=2024.0" "sentencepiece==0.2.1" "protobuf>=4.25,<7"
mkdir -p ~/mt2 && cd ~/mt2
git init -q . && git fetch -q ~/mt2.bundle main && git checkout -q FETCH_HEAD
~/venv/bin/python -c "import terminus; print('repo ok')"
