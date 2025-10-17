#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_burgers_model.py
- This script loads the pre-trained weights from 'burgers_best_model.keras'
  and runs ONLY the evaluation for the Burgers' equation benchmark.
- It does NOT perform any training.
- It regenerates the necessary data and scalers to ensure a correct evaluation context.
- UPDATE: Now evaluates on TWO test cases: one for interpolation and one for extrapolation,
  to provide a more robust assessment of the model's generalization capabilities.

Prerequisites:
  - The 'burgers_best_model.keras' file must be in the same directory.

Run:
  python evaluate_burgers_model.py
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------- HYPERPARAMETERS (Must match the training script) ----------------
FUSION_DIM, DECODER_DIM = 96, 128
L2W, DROPOUT, INPUT_NOISE = 1e-4, 0.2, 0.02
VAL_SIZE = 0.2
BATCH = 512
# ----------------------------------------------------------------------------------

# ---------------- DATA GENERATION & FEATURE ENGINEERING (Copied from training script) ----------------

def solve_burgers_implicit(nu, nx=256, nt=200, L=2.0, T=1.0):
    dx = L / (nx - 1)
    dt = T / nt
    x = np.linspace(-L / 2, L / 2, nx)
    t = np.linspace(0, T, nt)
    u = np.sin(np.pi * x)
    u_solution = np.zeros((nt, nx))
    u_solution[0, :] = u.copy()
    alpha = nu * dt / (2 * dx**2)
    A = diags([-alpha, 1 + 2 * alpha, -alpha], [-1, 0, 1], shape=(nx, nx)).tocsc()
    max_grad_t, max_grad = 0, 0
    for n in range(nt - 1):
        un = u.copy()
        rhs = un.copy()
        rhs[1:-1] -= (dt / (2*dx)) * 0.5 * (un[2:]**2 - un[:-2]**2)
        rhs[1:-1] += nu * dt / (2*dx**2) * (un[2:] - 2*un[1:-1] + un[:-2])
        u = spsolve(A, rhs)
        u_solution[n + 1, :] = u
        current_max_grad = np.max(np.abs(np.gradient(u, dx)))
        if current_max_grad > max_grad:
            max_grad, max_grad_t = current_max_grad, t[n+1]
    return x, t, u_solution, max_grad_t

def calibrate_shock_map(viscosities, all_solutions, x_domain, t_domain):
    shock_times = []
    for i, u_sol in enumerate(all_solutions):
        max_grad = 0
        max_grad_t = 0
        for n in range(u_sol.shape[0]):
            current_max_grad = np.max(np.abs(np.gradient(u_sol[n, :], np.median(np.diff(x_domain)))))
            if current_max_grad > max_grad:
                max_grad = current_max_grad
                max_grad_t = t_domain[n]
        shock_times.append(max_grad_t)
    
    viscosities_arr = np.array(viscosities).reshape(-1, 1)
    shock_times_arr = np.array(shock_times)
    shock_time_regressor = HuberRegressor().fit(viscosities_arr, shock_times_arr)
    return shock_time_regressor

def get_features(pr, x, y, x_shock):
    branch_features = np.stack([np.full(x.size, pr), np.full(x.size, x_shock)], axis=-1)
    d_t = y - x_shock
    s_t = 1.0 / (1.0 + np.exp(-50.0 * d_t))
    dt_unique = 1.0 / 199 # T / (nt-1)
    r1 = np.exp(-(d_t**2)/(2.0*(3.0*dt_unique)**2))
    r2 = np.exp(-(d_t**2)/(2.0*(7.0*dt_unique)**2))
    trunk_features = np.stack([x.ravel(), y.ravel(), d_t.ravel(), s_t.ravel(), np.abs(d_t.ravel()), d_t.ravel()**2, r1.ravel(), r2.ravel()], axis=-1)
    return branch_features, trunk_features

def build_fusion_model():
    b_in, t_in = layers.Input(shape=(2,), name="branch"), layers.Input(shape=(8,), name="trunk")
    def res_block(x, dim):
        x_res = layers.Dense(dim)(x)
        x_out = layers.BatchNormalization()(x)
        x_out = layers.Activation('swish')(x_out)
        x_out = layers.Dense(dim, kernel_regularizer=regularizers.l2(L2W))(x_out)
        x_out = layers.Dropout(DROPOUT)(x_out)
        return layers.Add()([x_res, x_out])
    b, t = layers.GaussianNoise(INPUT_NOISE)(b_in), layers.GaussianNoise(INPUT_NOISE)(t_in)
    for _ in range(3): b = res_block(b, FUSION_DIM)
    for _ in range(4): t = res_block(t, FUSION_DIM)
    b, t = layers.Activation('tanh')(b), layers.Activation('tanh')(t)
    f = layers.Multiply()([b, t])
    d = f
    for _ in range(3): d = res_block(d, DECODER_DIM)
    out = layers.Dense(1, activation='linear')(d)
    return keras.Model([b_in, t_in], out)

# ---------------- EVALUATION SCRIPT ----------------

def main():
    print("[INFO] Re-generating data context to fit scalers...")
    viscosities = np.logspace(-2.5, -1.5, 30)
    all_solutions, valid_viscosities = [], []
    x_domain, t_domain = None, None
    for nu in viscosities:
        x_d, t_d, u_sol, _ = solve_burgers_implicit(nu)
        if x_domain is None: x_domain, t_domain = x_d, t_d
        if not (np.isnan(u_sol).any() or np.isinf(u_sol).any()):
            all_solutions.append(u_sol)
            valid_viscosities.append(nu)
    viscosities = np.array(valid_viscosities)

    shock_time_regressor = calibrate_shock_map(viscosities, all_solutions, x_domain, t_domain)
    def shock_time_from_nu(nu):
        return float(shock_time_regressor.predict(np.array([[nu]]))[0])
        
    all_B, all_T, all_Y = [], [], []
    xx, tt = np.meshgrid(x_domain, t_domain)
    for i, nu in enumerate(viscosities):
        t_shock_pred = shock_time_from_nu(nu)
        branch, trunk = get_features(nu, xx, tt, t_shock_pred)
        all_B.append(branch)
        all_T.append(trunk)
        all_Y.append(all_solutions[i].ravel().reshape(-1, 1))

    B, T, Y = np.vstack(all_B).astype(np.float32), np.vstack(all_T).astype(np.float32), np.vstack(all_Y).astype(np.float32)
    groups = np.repeat(viscosities, xx.size)
    
    # --- Define test cases ---
    test_cases = [
        {"name": "Interpolation", "nu": viscosities[len(viscosities) // 2]},
        {"name": "Extrapolation", "nu": 0.04} # A value outside the training range
    ]

    # --- Fit Scalers ONCE on the full training data partition ---
    # We define a combined training mask to exclude all test cases
    train_mask = np.full(len(groups), True)
    for case in test_cases:
        # Note: only exclude if the nu is actually in the training set
        if case['nu'] in viscosities:
            train_mask &= (groups != case['nu'])

    bsc = StandardScaler().fit(B[train_mask])
    tsc = StandardScaler().fit(T[train_mask])
    ysc = StandardScaler().fit(Y[train_mask])
    
    # --- Build and Load Model ---
    keras.backend.clear_session()
    model = build_fusion_model()
    checkpoint_filepath = './burgers_best_model.keras'

    if not os.path.exists(checkpoint_filepath):
        print(f"[ERROR] Checkpoint file not found: {checkpoint_filepath}")
        return

    print(f"\n[INFO] Loading weights from {checkpoint_filepath}...")
    model.load_weights(checkpoint_filepath)
    print("[INFO] Weights loaded successfully.")

    # --- Loop through and evaluate each test case ---
    for case in test_cases:
        test_nu = case['nu']
        case_name = case['name']

        print("\n" + "="*80)
        print(f"[*] Evaluating Test Case: {case_name} (nu = {test_nu:.4f})")
        print("="*80)

        # --- Prepare Test Data ---
        x_test_domain, t_test_domain, u_exact, _ = solve_burgers_implicit(test_nu)
        
        xx_test, tt_test = np.meshgrid(x_test_domain, t_test_domain)
        t_shock_test = shock_time_from_nu(test_nu)
        
        Xb_test, Xt_test = get_features(test_nu, xx_test, tt_test, t_shock_test)

        # --- Predict and Evaluate ---
        pred_scaled = model.predict([bsc.transform(Xb_test), tsc.transform(Xt_test)], batch_size=BATCH*4)
        pred_unscaled = ysc.inverse_transform(pred_scaled).reshape(u_exact.shape)
        
        l2_error = np.linalg.norm(u_exact - pred_unscaled) / np.linalg.norm(u_exact)
        print(f"[RESULT] L2 Relative Error for {case_name}: {l2_error:.4%}")

        # --- Visualize ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        fig.suptitle(f'Burgers Eq. - {case_name} Test ($\\nu={test_nu:.4f}$)', fontsize=16)
        time_steps_to_plot = [int(len(t_test_domain) * p) for p in [0.25, 0.5, 0.75, 0.99]]
        for i, ax in enumerate(axes.flatten()):
            t_idx = time_steps_to_plot[i]
            ax.plot(x_test_domain, u_exact[t_idx, :], 'b-', label='Numerical Solution')
            ax.plot(x_test_domain, pred_unscaled[t_idx, :], 'r--', label='Fusion-DeepONet')
            ax.set_title(f'Time t = {t_test_domain[t_idx]:.2f} s'); ax.grid(True, linestyle=':'); ax.legend()
        fig.supxlabel('Spatial Coordinate (x)'); fig.supylabel('Velocity (u)')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        output_filename = f'burgers_evaluation_{case_name.lower()}.png'
        plt.savefig(output_filename)
        print(f"[OK] Evaluation plot saved to '{output_filename}'")

if __name__ == "__main__":
    main()

