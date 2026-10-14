import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
N_MORPH = 16        
N_TIME_PTS = 60     
TIME_POINTS = np.arange(N_TIME_PTS)
DELAYS_MS = np.array([3, 6, 9]) 
N_SUBJECTS = 15     
LAG = 6            # True simulated transmission delay

def get_triu(mat): 
    return mat[np.triu_indices(N_MORPH, k=1)]

# Establish Ground-Truth Model Templates (4 positions x 4 categories = 16 conditions)
pos_labels = np.repeat(np.arange(4), 4).reshape(-1, 1) 
MODEL_POS = get_triu(squareform(pdist(pos_labels, metric='hamming')))

cat_labels = np.tile(np.arange(4), 4).reshape(-1, 1)  
MODEL_CAT = get_triu(squareform(pdist(cat_labels, metric='hamming')))

MODEL_CONTROL = get_triu(squareform(pdist(np.random.permutation(cat_labels), metric='hamming')))

# --- 2. CORE ANALYTICAL HELPER ---
def get_partial_corr(target_vec, nuisance_vecs, neural_vec):
    try:
        X = np.column_stack([c.flatten() for c in nuisance_vecs])
        reg_t = LinearRegression().fit(X, target_vec.flatten())
        res_t = target_vec.flatten() - reg_t.predict(X)
        reg_n = LinearRegression().fit(X, neural_vec.flatten())
        res_n = neural_vec.flatten() - reg_n.predict(X)
        rho, _ = spearmanr(res_t, res_n)
        return rho if not np.isnan(rho) else 0
    except: 
        return 0

# --- 3. MULTIVARIATE DATA SIMULATION (No CV Splits) ---
n_elements = MODEL_POS.shape[0]
sim_neural_A = np.zeros((N_SUBJECTS, N_TIME_PTS, n_elements))
sim_neural_B = np.zeros((N_SUBJECTS, N_TIME_PTS, n_elements))

activation_window = range(15, 35)

for sub in range(N_SUBJECTS):
    for t in TIME_POINTS:
        # Area A encodes Position during activation window
        if t in activation_window:
            sim_neural_A[sub, t, :] = MODEL_POS + np.random.normal(0, 2.5, n_elements)
        else:
            sim_neural_A[sub, t, :] = np.random.normal(0, 2.5, n_elements)
        
        # Area B receives transformed Category information after the LAG
        if (t - LAG) in activation_window:
            sim_neural_B[sub, t, :] = MODEL_CAT + np.random.normal(0, 2.5, n_elements)
        else:
            sim_neural_B[sub, t, :] = np.random.normal(0, 2.5, n_elements)

# --- 4. STEP-BY-STEP EMULATION PIPELINE ---
group_ff = np.zeros((N_SUBJECTS, len(DELAYS_MS), len(TIME_POINTS)))
group_ff_ident_pos = np.zeros((N_SUBJECTS, len(DELAYS_MS), len(TIME_POINTS)))
group_ff_ident_cat = np.zeros((N_SUBJECTS, len(DELAYS_MS), len(TIME_POINTS)))

group_c_ant_pos = np.zeros((N_SUBJECTS, len(TIME_POINTS)))
group_c_ant_cat = np.zeros((N_SUBJECTS, len(TIME_POINTS)))
group_c_post_pos = np.zeros((N_SUBJECTS, len(TIME_POINTS)))
group_c_post_cat = np.zeros((N_SUBJECTS, len(TIME_POINTS)))

for sub in range(N_SUBJECTS):
    scalar_A_pos = np.zeros(len(TIME_POINTS))
    scalar_A_cat = np.zeros(len(TIME_POINTS))
    scalar_B_pos = np.zeros(len(TIME_POINTS))
    scalar_B_cat = np.zeros(len(TIME_POINTS))
    
    # Pass 1: Extract standard time-resolved tracking scalars
    for i_t, t in enumerate(TIME_POINTS):
        eval_neural_ant = sim_neural_A[sub, t]
        eval_neural_post = sim_neural_B[sub, t]
        
        scalar_A_pos[i_t] = get_partial_corr(MODEL_POS, [MODEL_CAT, MODEL_CONTROL], eval_neural_ant)
        scalar_A_cat[i_t] = get_partial_corr(MODEL_CAT, [MODEL_POS, MODEL_CONTROL], eval_neural_ant)
        scalar_B_pos[i_t] = get_partial_corr(MODEL_POS, [MODEL_CAT, MODEL_CONTROL], eval_neural_post)
        scalar_B_cat[i_t] = get_partial_corr(MODEL_CAT, [MODEL_POS, MODEL_CONTROL], eval_neural_post)
    
    group_c_ant_pos[sub] = scalar_A_pos
    group_c_ant_cat[sub] = scalar_A_cat
    group_c_post_pos[sub] = scalar_B_pos
    group_c_post_cat[sub] = scalar_B_cat

    # Pass 2: Directed Flow via Scalar Products with Destination RDM-level History Regression
    for i_t, t in enumerate(TIME_POINTS):
        for d_idx, dly in enumerate(DELAYS_MS):
            t_p = t - dly
            if t_p in TIME_POINTS:
                i_t_past = np.where(TIME_POINTS == t_p)[0][0]
                
                past_dest_rdm = sim_neural_B[sub, t_p].flatten().reshape(-1, 1)
                eval_neural_post = sim_neural_B[sub, t].flatten()
                
                # Regress destination history directly out of the Destination Neural RDM
                res_neural_post = eval_neural_post - LinearRegression().fit(past_dest_rdm, eval_neural_post).predict(past_dest_rdm)
                
                # Regress destination history directly out of the Model Templates
                res_model_pos = MODEL_POS.flatten() - LinearRegression().fit(past_dest_rdm, MODEL_POS.flatten()).predict(past_dest_rdm)
                res_model_cat = MODEL_CAT.flatten() - LinearRegression().fit(past_dest_rdm, MODEL_CAT.flatten()).predict(past_dest_rdm)
                res_model_ctrl = MODEL_CONTROL.flatten() - LinearRegression().fit(past_dest_rdm, MODEL_CONTROL.flatten()).predict(past_dest_rdm)
                
                # Compute history-stripped partial correlations at destination
                now_post_cat = get_partial_corr(res_model_cat, [res_model_pos, res_model_ctrl], res_neural_post)
                now_post_pos = get_partial_corr(res_model_pos, [res_model_cat, res_model_ctrl], res_neural_post)
                
                # Fetch past coding scalars from Source Area A
                past_ant_pos = scalar_A_pos[i_t_past]
                past_ant_cat = scalar_A_cat[i_t_past]
                
                # Calculate Flow combinations via scalar multiplication
                group_ff[sub, d_idx, i_t] = past_ant_pos * now_post_cat            # Transformed Flow (Pos -> Cat)
                group_ff_ident_pos[sub, d_idx, i_t] = past_ant_pos * now_post_pos  # Identical Control 1 (Pos -> Pos)
                group_ff_ident_cat[sub, d_idx, i_t] = past_ant_cat * now_post_cat  # Identical Control 2 (Cat -> Cat)

# --- 5. RDM VISUALIZATION CHECK ---
RDM_POS_FULL = squareform(pdist(pos_labels, metric='hamming'))[:N_MORPH, :N_MORPH]
RDM_CAT_FULL = squareform(pdist(cat_labels, metric='hamming'))[:N_MORPH, :N_MORPH]

fig_rdm, axes_rdm = plt.subplots(1, 2, figsize=(10, 4.5))

im1 = axes_rdm[0].imshow(RDM_POS_FULL, cmap='viridis', origin='upper')
axes_rdm[0].set_title(f"Model Template: Position ({N_MORPH}x{N_MORPH})")
axes_rdm[0].set_xticks(np.arange(16))
axes_rdm[0].set_yticks(np.arange(16))
axes_rdm[0].set_xticklabels(np.arange(1, 17))
axes_rdm[0].set_yticklabels(np.arange(1, 17))
fig_rdm.colorbar(im1, ax=axes_rdm[0], fraction=0.046, pad=0.04)

im2 = axes_rdm[1].imshow(RDM_CAT_FULL, cmap='viridis', origin='upper')
axes_rdm[1].set_title(f"Model Template: Category ({N_MORPH}x{N_MORPH})")
axes_rdm[1].set_xticks(np.arange(16))
axes_rdm[1].set_yticks(np.arange(16))
axes_rdm[1].set_xticklabels(np.arange(1, 17))
axes_rdm[1].set_yticklabels(np.arange(1, 17))
fig_rdm.colorbar(im2, ax=axes_rdm[1], fraction=0.046, pad=0.04)

plt.tight_layout()
fig_rdm.savefig('model_templates.pdf', dpi=300, bbox_inches='tight')

# --- 6. TIMELINE & FLOW PLOTS ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Plot A: Time-Resolved Informational Traces
axes[0].plot(np.mean(group_c_ant_pos, axis=0), label='Area A: Position Coding', color='darkblue', lw=2.5)
axes[0].plot(np.mean(group_c_post_cat, axis=0), label='Area B: Category Coding', color='darkred', lw=2.5)
axes[0].plot(np.mean(group_c_ant_cat, axis=0), label='Area A: Category Coding', color='blue', ls='--', alpha=0.6)
axes[0].plot(np.mean(group_c_post_pos, axis=0), label='Area B: Position Coding', color='red', ls='--', alpha=0.6)
axes[0].set_title("Information Coding Timelines Across Regions")
axes[0].set_xlabel("Time Points")
axes[0].set_ylabel("Partial Correlation Profile")
axes[0].legend(loc='upper left')
axes[0].grid(True, ls='--', alpha=0.5)

# Plot B: Time-Resolved Connectivity Flow (Evaluated at the True Simulated Lag)
lag_idx = np.where(DELAYS_MS == LAG)[0][0]
flow_trans_time = np.mean(group_ff[:, lag_idx, :], axis=0)
flow_ident_pos_time = np.mean(group_ff_ident_pos[:, lag_idx, :], axis=0)
flow_ident_cat_time = np.mean(group_ff_ident_cat[:, lag_idx, :], axis=0)

axes[1].plot(flow_trans_time, label='Transformed RCA (Pos -> Cat)', color='purple', lw=2.5)
axes[1].plot(flow_ident_pos_time, label='Identical RCA (Pos -> Pos)', color='navy', ls='--', alpha=0.7)
axes[1].plot(flow_ident_cat_time, label='Identical RCA (Cat -> Cat)', color='crimson', ls='--', alpha=0.7)
axes[1].set_title(f"Time-Resolved Connectivity Flow (Delay = {LAG} time points)")
axes[1].set_xlabel("Time Points")
axes[1].set_ylabel("Covariance Flow Product Metric")
axes[1].legend(loc='upper left')
axes[1].grid(True, ls='--', alpha=0.5)

plt.tight_layout()
fig.savefig('connectivity_timelines.pdf', dpi=300, bbox_inches='tight')
plt.show()