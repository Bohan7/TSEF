#!/bin/bash
# Run TSEF Attack on all datasets

echo "=========================================="
echo "Running TSEF Attack on all datasets"
echo "=========================================="

# Make all scripts executable
chmod +x run_*.sh

echo ""
echo "[1/6] Running LowVarDetect..."
echo "=========================================="
bash run_lowvar.sh

echo ""
echo "[2/6] Running SeqCombMV..."
echo "=========================================="
bash run_seqcombmv.sh

echo ""
echo "[3/6] Running SeqCombSingle..."
echo "=========================================="
bash run_seqcombsingle.sh

echo ""
echo "[4/6] Running MITECG..."
echo "=========================================="
bash run_mitecg.sh

echo ""
echo "[5/6] Running PAM..."
echo "=========================================="
bash run_pam.sh

echo ""
echo "[6/6] Running Epilepsy..."
echo "=========================================="
bash run_epilepsy.sh

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
