#!/usr/bin/env bash
echo "=== Installing Fortran compiler for SciPy ==="
apt-get update && apt-get install -y gfortran libgfortran5

echo "=== Installing Python dependencies ==="
pip install -r requirements.txt --no-cache-dir

echo "=== Build completed successfully ==="