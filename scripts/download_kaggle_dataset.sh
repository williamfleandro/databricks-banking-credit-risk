#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
kaggle datasets download -d laotse/credit-risk-dataset -p data/raw --unzip
ls -lh data/raw
