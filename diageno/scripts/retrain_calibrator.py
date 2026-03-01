#!/usr/bin/env python3
"""Retrain the calibrator with the new percentile-based algorithm."""

import logging
from diageno.training.disease_scorer import fit_calibrator
from diageno.training.phenotype_selector import load_matrix_artifacts
from diageno.config import settings

logging.basicConfig(level=logging.INFO)

matrix, disease_index, hpo_dict, inv = load_matrix_artifacts(settings.artifacts_dir)

cal = fit_calibrator(settings.silver_dir, settings.artifacts_dir, matrix, disease_index, hpo_dict)
if cal:
    print(f"Calibrator type: {cal['type']}")
    print(f"Cases used: {cal['n_cases']}")
    print(f"Top-1 mean: {cal['top1_mean']:.4f} +/- {cal['top1_std']:.4f}")
    print(f"Gap mean: {cal['gap_mean']:.4f} +/- {cal['gap_std']:.4f}")
    print(f"Top-1 percentiles (10/25/50/75/90): {[round(p,4) for p in cal['top1_percentiles']]}")
    print(f"Gap percentiles (10/25/50/75/90): {[round(p,4) for p in cal['gap_percentiles']]}")
else:
    print("Calibrator failed")
