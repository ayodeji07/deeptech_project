#!/usr/bin/env bash
echo "=== Installing system dependencies for SciPy (gfortran + BLAS/LAPACK) ==="
apt-get update && apt-get install -y gfortran libblas-dev liblapack-dev libopenblas-dev libgfortran5

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt --no-cache-dir

echo "=== Build completed successfully ==="