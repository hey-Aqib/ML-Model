# =============================================================================
# run_demo.py  —  ONE-CLICK DEMO LAUNCHER
# Run this file in Spyder: press F5 or click "Run"
# =============================================================================
#
# STEP 1: Trains all 5 ML models (if not already trained)
# STEP 2: Generates all evaluation plots
# STEP 3: Launches the Flask web interface at http://127.0.0.1:5000
#
# Requirements: conda activate ml_metrics_b6
#               pip install -r requirements.txt
# =============================================================================

import os
import sys

print("=" * 60)
print("  PROJECT B6 — Comparative Study of Evaluation Metrics")
print("  UCI Adult Income Dataset")
print("=" * 60)

# Make sure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"\nWorking directory: {script_dir}")

# ── Step 1: Train models if needed ──────────────────────────────────────────
models_exist = all([
    os.path.exists('models/random_forest.pkl'),
    os.path.exists('models/scaler.pkl'),
    os.path.exists('data/adult.data'),
])

if not models_exist:
    print("\n[INFO] Models not found. Running training pipeline...")
    import train_models  # This executes training
else:
    print("\n[INFO] Pre-trained models found. Skipping training.")
    print("       (Delete the models/ folder to retrain from scratch)")

# ── Step 2: Launch Flask ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Launching Flask web interface...")
print("  Open your browser at:  http://127.0.0.1:5000")
print("  Press Ctrl+C in the console to stop the server.")
print("=" * 60 + "\n")

# Import and run the Flask app
from app import app
app.run(debug=False, use_reloader=False, host='127.0.0.1', port=5000)
