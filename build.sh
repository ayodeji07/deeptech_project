#!/usr/bin/env bash
echo "=== Installing system dependencies for SciPy (gfortran + OpenBLAS) ==="
apt-get update && apt-get install -y gfortran libopenblas-dev liblapack-dev

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt --no-cache-dir

echo "=== Build completed successfully ==="