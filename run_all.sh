#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
for e in exp1_mine_corpus exp2_build_benchmark exp3_detectors exp3b_negative_type_analysis exp4_cross_project exp5_feature_analysis exp6_wild_scan exp7_leave_one_mutation_out; do
  echo "=============== $e ==============="
  python experiments/$e.py
done
