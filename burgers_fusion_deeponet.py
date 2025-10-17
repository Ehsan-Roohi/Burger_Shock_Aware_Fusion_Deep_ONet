#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
burgers_fusion_deeponet.py
- Implements the user's specific "Shock-Aware Physics-Guided Fusion-DeepONet" for the 1D viscous Burgers' equation.
- This script directly mimics the methodology from the user's paper (the 'External_Calib' case).
- It demonstrates the generality of the proposed physics-guided feature engineering approach.
- OPTIMIZATIONS:
  1. Replaced the explicit solver with a more stable and accurate Implicit (Crank-Nicolson) solver.
  2. Enhanced the network architecture with Residual Connections to improve gradient flow and training speed.
  3. Implemented a Cosine Annealing learning rate scheduler for faster and more robust convergence.

Key Features:
1.  **External Calibration:** A simple regressor learns the mapping from viscosity (nu) to the shock formation time (t_shock).
2.  **Physics-Guided Features:**
    - Branch Network Input: (nu, t_shock)
    - Trunk Network Input: A rich set of features including (x, t), distance to shock time, sigmoid, and Gaussian basis functions.
3.  **Gradient-Based Sample Weighting:** Focuses training on the shock region using |du/dx|.
4.  **Two-Phase Training:** Uses a warm-up and focus phase for curriculum learning.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------- HYPERPARAMETERS (Optimized) ----------------
FUSION_DIM, DECODER_DIM = 96, 128
L2W, DROPOUT, INPUT_NOISE = 1e-4, 0.2, 0.02
INITIAL_LR, BATCH = 1e-3, 512
EPOCHS_WARM, EPOCHS_FOCUS = 150, 200 # Can use fewer epochs with faster convergence
VAL_SIZE = 0.2
HUBER_WARM, HUBER_FOCUS = 0.5, 0.2
GW_SCALE = 0.8
CLIP1, CLIP2 = (1.0, 4.5), (1.0, 5.0)
RLROP1, RLROP2 = 8, 8
EARLY1, EARLY2 = 20, 25 # FIX: Re-added missing parameters
# -------------------------------------------------

# ---------------- 1. OPTIMIZED DATA GENERATION (Implicit Solver) ----------------
def solve_burgers_implicit(nu, nx=256, nt=200, L=2.0, T=1.0):
    """
    Solves the 1D viscous Burgers' equation using a stable Crank-Nicolson implicit scheme.
    """
    dx = L / (nx - 1)
    dt = T / nt
    x = np.linspace(-L / 2, L / 2, nx)
    t = np.linspace(0, T, nt)
    
    u = np.sin(np.pi * x)
    u_solution = np.zeros((nt, nx))
    u_solution[0, :] = u.copy()
    
    # Create the matrices for the implicit system
    alpha = nu * dt / (2 * dx**2)
    A_diag = 1 + 2 * alpha
    A_upper = -alpha
    A_lower = -alpha
    A = diags([A_lower, A_diag, A_upper], [-1, 0, 1], shape=(nx, nx)).tocsc()

    max_grad_t, max_grad = 0, 0
    
    for n in range(nt - 1):
        un = u.copy()
        
        # Right-hand side (RHS) of the linear system
        rhs = un.copy()
        # Advection term (explicit part)
        rhs[1:-1] -= (dt / (2*dx)) * 0.5 * (un[2:]**2 - un[:-2]**2)
        # Diffusion term (explicit part)
        rhs[1:-1] += nu * dt / (2*dx**2) * (un[2:] - 2*un[1:-1] + un[:-2])
        
        # Solve the linear system A*u_new = rhs
        u = spsolve(A, rhs)
        u_solution[n + 1, :] = u
        
        current_max_grad = np.max(np.abs(np.gradient(u, dx)))
        if current_max_grad > max_grad:
            max_grad, max_grad_t = current_max_grad, t[n+1]
            
    return x, t, u_solution, max_grad_t

# --- Generate a dataset for a range of viscosities and find shock times ---
print("[INFO] Generating training data for Burgers' equation using implicit solver...")
viscosities = np.logspace(-2.5, -1.5, 30)
shock_times, all_solutions = [], []
valid_viscosities = []

for nu in viscosities:
    print(f"  Solving for nu = {nu:.4f}")
    x_domain, t_domain, u_sol, t_shock = solve_burgers_implicit(nu)
    if np.isnan(u_sol).any() or np.isinf(u_sol).any():
        print(f"  [ERROR] Solver became unstable for nu = {nu:.4f}. Skipping.")
        continue
    shock_times.append(t_shock)
    all_solutions.append(u_sol)
    valid_viscosities.append(nu)

viscosities = np.array(valid_viscosities) # Use only valid viscosities

# ---------------- 2. EXTERNAL CALIBRATION ----------------
print("\n[INFO] Calibrating shock time predictor (nu -> t_shock)...")
viscosities_arr = np.array(viscosities).reshape(-1, 1)
shock_times_arr = np.array(shock_times)
shock_time_regressor = HuberRegressor().fit(viscosities_arr, shock_times_arr)

def shock_time_from_nu(nu):
    return float(shock_time_regressor.predict(np.array([[nu]]))[0])

# ---------------- 3. FEATURE ENGINEERING & DATA ASSEMBLY ----------------
print("[INFO] Assembling physics-guided features...")
all_B, all_T, all_Y, all_Wg = [], [], [], []
xx, tt = np.meshgrid(x_domain, t_domain)
for i, nu in enumerate(viscosities):
    u_sol = all_solutions[i]
    t_shock_pred = shock_time_from_nu(nu)
    d_t = tt - t_shock_pred
    s_t = 1.0 / (1.0 + np.exp(-50.0 * d_t))
    dt_unique = np.median(np.diff(t_domain))
    r1 = np.exp(-(d_t**2)/(2.0*(3.0*dt_unique)**2))
    r2 = np.exp(-(d_t**2)/(2.0*(7.0*dt_unique)**2))
    trunk_features = np.stack([xx.ravel(), tt.ravel(), d_t.ravel(), s_t.ravel(), np.abs(d_t.ravel()), d_t.ravel()**2, r1.ravel(), r2.ravel()], axis=-1)
    branch_features = np.stack([np.full(xx.size, nu), np.full(xx.size, t_shock_pred)], axis=-1)
    grad_u = np.abs(np.gradient(u_sol, x_domain, axis=1)).ravel()
    q95 = np.percentile(grad_u, 95)
    wg = 1.0 + GW_SCALE * np.clip(grad_u / (q95 + 1e-8), 0, 1)
    all_B.append(branch_features); all_T.append(trunk_features); all_Y.append(u_sol.ravel().reshape(-1, 1)); all_Wg.append(wg)

B, T, Y, Wg = np.vstack(all_B).astype(np.float32), np.vstack(all_T).astype(np.float32), np.vstack(all_Y).astype(np.float32), np.hstack(all_Wg).astype(np.float32)
groups = np.repeat(viscosities, xx.size)

# ---------------- 4. DEEPONET MODEL DEFINITION (with Residual Connections) ----------------
def build_fusion_model():
    b_in, t_in = layers.Input(shape=(2,), name="branch"), layers.Input(shape=(8,), name="trunk")

    def res_block(x, dim):
        x_res = layers.Dense(dim)(x) # Project residual if dimensions differ
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

# ---------------- 5. TRAINING PIPELINE (with Cosine Annealing) ----------------
class CosineAnnealingScheduler(keras.callbacks.Callback):
    def __init__(self, n_epochs, n_cycles, lr_max):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lr_max
        self.lrates = []

    def cosine_annealing(self, epoch):
        epochs_per_cycle = np.floor(self.epochs/self.cycles)
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / epochs_per_cycle
        return self.lr_max/2 * (np.cos(cos_inner) + 1)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.cosine_annealing(epoch)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.lrates.append(lr)

# --- Data Split & Scaling ---
test_nu = viscosities[len(viscosities) // 2]
train_mask = (groups != test_nu); test_mask = (groups == test_nu)
Xb_train, Xb_test = B[train_mask], B[test_mask]
Xt_train, Xt_test = T[train_mask], T[test_mask]
Y_train, Y_test = Y[train_mask], Y[test_mask]
Wg_train = Wg[train_mask]
Xb_train, Xb_val, Xt_train, Xt_val, Y_train, Y_val, Wg_train, _ = train_test_split(
    Xb_train, Xt_train, Y_train, Wg_train, test_size=VAL_SIZE, random_state=42)
bsc, tsc, ysc = StandardScaler().fit(Xb_train), StandardScaler().fit(Xt_train), StandardScaler().fit(Y_train)
Xb_train_s, Xt_train_s, Y_train_s = bsc.transform(Xb_train), tsc.transform(Xt_train), ysc.transform(Y_train)
Xb_val_s, Xt_val_s, Y_val_s = bsc.transform(Xb_val), tsc.transform(Xt_val), ysc.transform(Y_val)
W1, W2 = np.clip(Wg_train, *CLIP1), np.clip(Wg_train, *CLIP2)

# --- Training ---
keras.backend.clear_session()
model = build_fusion_model()
opt = keras.optimizers.Adam(learning_rate=INITIAL_LR, clipnorm=1.0)
checkpoint_filepath = './burgers_best_model.keras'
model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)
lr_scheduler_warm = CosineAnnealingScheduler(EPOCHS_WARM, 5, INITIAL_LR)
lr_scheduler_focus = CosineAnnealingScheduler(EPOCHS_FOCUS, 5, INITIAL_LR/5) # Lower LR for focus phase
cbs1 = [keras.callbacks.EarlyStopping(patience=EARLY1, min_delta=1e-5), model_checkpoint, lr_scheduler_warm, keras.callbacks.TerminateOnNaN()]
cbs2 = [keras.callbacks.EarlyStopping(patience=EARLY2, min_delta=1e-5), model_checkpoint, lr_scheduler_focus, keras.callbacks.TerminateOnNaN()]

print("\n[INFO] Starting Phase 1 Training (Warm-up)...")
model.compile(optimizer=opt, loss=tf.keras.losses.Huber(delta=HUBER_WARM), metrics=['mae'])
model.fit([Xb_train_s, Xt_train_s], Y_train_s, batch_size=BATCH, epochs=EPOCHS_WARM, validation_data=([Xb_val_s, Xt_val_s], Y_val_s), sample_weight=W1, callbacks=cbs1, verbose=2)

print("\n[INFO] Starting Phase 2 Training (Focus)...")
model.compile(optimizer=opt, loss=tf.keras.losses.Huber(delta=HUBER_FOCUS), metrics=['mae'])
model.fit([Xb_train_s, Xt_train_s], Y_train_s, batch_size=BATCH, epochs=EPOCHS_FOCUS, validation_data=([Xb_val_s, Xt_val_s], Y_val_s), sample_weight=W2, callbacks=cbs2, verbose=2)

# ---------------- 6. EVALUATION & VISUALIZATION ----------------
print("[INFO] Loading best model weights for final evaluation...")
model.load_weights(checkpoint_filepath)
print(f"\n[INFO] Evaluating model on unseen viscosity: nu = {test_nu:.4f}")
_, _, u_exact = solve_burgers_implicit(test_nu)
pred_scaled = model.predict([bsc.transform(Xb_test), tsc.transform(Xt_test)])
pred_unscaled = ysc.inverse_transform(pred_scaled).reshape(u_exact.shape)
l2_error = np.linalg.norm(u_exact - pred_unscaled) / np.linalg.norm(u_exact)
print(f"[RESULT] L2 Relative Error: {l2_error:.4%}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
fig.suptitle(f'Burgers Eq. with Physics-Guided Fusion-DeepONet ($\\nu={test_nu:.4f}$)', fontsize=16)
time_steps_to_plot = [int(len(t_domain) * p) for p in [0.25, 0.5, 0.75, 0.99]]
for i, ax in enumerate(axes.flatten()):
    t_idx = time_steps_to_plot[i]
    ax.plot(x_domain, u_exact[t_idx, :], 'b-', label='Numerical Solution')
    ax.plot(x_domain, pred_unscaled[t_idx, :], 'r--', label='Fusion-DeepONet')
    ax.set_title(f'Time t = {t_domain[t_idx]:.2f} s'); ax.grid(True, linestyle=':'); ax.legend()
fig.supxlabel('Spatial Coordinate (x)'); fig.supylabel('Velocity (u)')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('burgers_fusion_deeponet_comparison.png')
print("\n[OK] Comparison plot saved to 'burgers_fusion_deeponet_comparison.png'")

