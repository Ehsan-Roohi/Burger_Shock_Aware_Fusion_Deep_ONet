# Shock-aware Fusion-DeepONet for viscous Burgers' equation

Research prototype for a physics-guided branch/trunk network applied to the
one-dimensional viscous Burgers equation.

## Files

- `burgers_fusion_deeponet.py` generates Crank–Nicolson reference fields,
  calibrates shock time from viscosity, trains a residual Fusion-DeepONet with
  shock-aware features and gradient weighting, and evaluates one held viscosity.
- `evaluate_burgers_model.py` rebuilds the data/scaler context and evaluates an
  existing `burgers_best_model.keras` checkpoint on interpolation and
  extrapolation cases.
- `Codes.ipynb` is the Colab notebook record.

## Run

Python dependencies are NumPy, SciPy, Matplotlib, scikit-learn, and TensorFlow.
Training creates the checkpoint and comparison plot:

```bash
python burgers_fusion_deeponet.py
python evaluate_burgers_model.py
```

The evaluation script requires `burgers_best_model.keras` in the repository
root.  Generated checkpoints and figures are not currently versioned.

## Scientific status

This is a compact prototype, not a benchmark-qualified operator-learning
package.  The training and evaluation scripts duplicate some model/data logic;
keep them synchronized before trusting a checkpoint.  A publication-quality
comparison should freeze seeds and software versions, separate viscosities by
parameter rather than individual space-time samples, state boundary-condition
handling, check the implicit reference discretization, and report shock-region
as well as global errors against independent baselines.

No repository-wide license or citation metadata is declared yet.
