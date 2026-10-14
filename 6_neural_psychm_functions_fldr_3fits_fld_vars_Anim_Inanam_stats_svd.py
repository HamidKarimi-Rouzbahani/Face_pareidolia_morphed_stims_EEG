import numpy as np
import matplotlib.pyplot as plt
import os
import re
import gc
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind

# --- 1. CONFIGURATION ---
INPUT_DIR = r'C:\Users\uqhkarim\OneDrive - The University of Queensland\Documents\Scientific\Projects\Pareidolia\EEG\Main data\Processed_Outputs'
TOPO_TIMES = [200]
WIN_HALF = 25 
# TOPO_TIMES = [-100, 0, 100, 200, 300, 400, 500, 600]
# WIN_HALF = 25 
MORPH_X = np.arange(1, 22)
MORPH_X_FINE = np.linspace(1, 21, 100)

OUTPUT_BASE = os.path.join(INPUT_DIR, 'Integrated_Psychometric_Results')
if not os.path.exists(OUTPUT_BASE): os.makedirs(OUTPUT_BASE)
SAVE_FILE_PATH = os.path.join(OUTPUT_BASE, 'topo_analysis_results.npy')

SUBJECT_SETTINGS = {
    "P00": {"exclude_blocks": []}, "P01": {"exclude_blocks": []}, "P02": {"exclude_blocks": []},
    "P03": {"exclude_blocks": []}, "P04": {"exclude_blocks": []}, "P05": {"exclude_blocks": []},
    "P06": {"exclude_blocks": []}, "P07": {"exclude_blocks": []}, "P08": {"exclude_blocks": []},
    "P09": {"exclude_blocks": [7, 8]}, "P10": {"exclude_blocks": []}, "P11": {"exclude_blocks": []},
    "P12": {"exclude_blocks": []}, "P13": {"exclude_blocks": []}, "P14": {"exclude_blocks": [1]},
    "P15": {"exclude_blocks": [9]}, "P16": {"exclude_blocks": []}, "P17": {"exclude_blocks": []},
    "P18": {"exclude_blocks": []}, "P19": {"exclude_blocks": []}, "P20": {"exclude_blocks": []},
    "P21": {"exclude_blocks": []}, "P22": {"exclude_blocks": []}, "P23": {"exclude_blocks": [1, 2, 3, 4, 5, 6, 7]},
    "P24": {"exclude_blocks": []}, "P25": {"exclude_blocks": []}, "P26": {"exclude_blocks": []},
    "P27": {"exclude_blocks": []}, "P28": {"exclude_blocks": []}, "P29": {"exclude_blocks": []}
}

ANIMATE_FOLDERS = [3, 7, 8, 9, 13, 15, 17, 19, 26, 27, 28]
INANIMATE_FOLDERS = [4, 5, 6, 10, 11, 12, 14, 16, 18, 20, 21, 22, 23, 24, 25]
ALL_FOLDERS = ANIMATE_FOLDERS + INANIMATE_FOLDERS

# --- 2. MODELS & HELPERS ---
def sigmoid_func(x, L, x0, k, b): return L / (1 + np.exp(-k * (x - x0))) + b
# def sigmoid_func(x, x0, k): return 1 / (1 + np.exp(-k * (x - x0)))

def linear_func(x, m, c): return m * x + c
def step_func(x, x0, low, high): return np.where(x < x0, low, high)

def get_adj_r2(y_true, y_pred, n_params):
    n = len(y_true)
    if n <= n_params + 1: return -1.0
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_params - 1)

def parse_metadata(names):
    parsed = []
    for name in names:
        match = re.search(r'F(\d+)_S(\d+)', name)
        if match:
            f = int(match.group(1)); s = int(match.group(2))
            m = (s - 100) if 3 <= f <= 28 else None
            parsed.append({'f': f, 'm': m})
        else: parsed.append({'f': -1, 'm': None})
    return parsed

# --- 3. PROCESSING ---
folder_raw_accumulator = {t: {f_id: [] for f_id in ALL_FOLDERS} for t in TOPO_TIMES}

print(f"Starting analysis for {len(SUBJECT_SETTINGS)} subjects...")
for s_idx, (sub_id, settings) in enumerate(SUBJECT_SETTINGS.items()):
    file_path = os.path.join(INPUT_DIR, f"{sub_id}.npy")
    if not os.path.exists(file_path): continue
    
    print(f"[{s_idx+1}/{len(SUBJECT_SETTINGS)}] Processing Subject: {sub_id}")
    data = np.load(file_path, allow_pickle=True).item()
    data_4d, time_ms = data['data_4d'], data['time_vector'] * 1000
    meta = parse_metadata(list(data['condition_names']))
    n_chan = data_4d.shape[0]
    keep_idx = [i for i in range(data_4d.shape[3]) if (i + 1) not in settings['exclude_blocks']]
    
    train_obj_f = [i for i, m in enumerate(meta) if m['f'] in [29, 30]]
    train_face_f = [i for i, m in enumerate(meta) if m['f'] in [1, 2]]
    y_train = np.concatenate([np.full(len(train_obj_f)*len(keep_idx), -1), np.ones(len(train_face_f)*len(keep_idx))])

    for t_target in TOPO_TIMES:
        t_mask = (time_ms >= t_target - WIN_HALF) & (time_ms <= t_target + WIN_HALF)
        if not np.any(t_mask): continue
        
        X_train_avg = np.nanmean(data_4d[:, t_mask, :, :], axis=1)
        X_train = np.nan_to_num(np.vstack([
            X_train_avg[:, train_obj_f, :][:, :, keep_idx].reshape(n_chan, -1).T,
            X_train_avg[:, train_face_f, :][:, :, keep_idx].reshape(n_chan, -1).T
        ]))
        
        clf = make_pipeline(SimpleImputer(), StandardScaler(), LinearSVC(dual='auto', max_iter=2000))
        clf.fit(X_train, y_train)

        for f_id in ALL_FOLDERS:
            f_profile = np.zeros(21)
            for m in range(1, 22):
                m_idx_list = [i for i, mt in enumerate(meta) if mt['f'] == f_id and mt['m'] == m]
                if m_idx_list:
                    X_test_window = data_4d[:, t_mask, m_idx_list[0], :]
                    X_test_final = np.nanmean(np.nanmean(X_test_window, axis=1)[:, keep_idx], axis=1)
                    f_profile[m-1] = clf.decision_function(np.nan_to_num(X_test_final).reshape(1, -1))[0]
            folder_raw_accumulator[t_target][f_id].append(f_profile)
    del data; gc.collect()

# --- 3.5 DATA SAVING SECTION ---
print(f"Saving compiled analysis outputs to {SAVE_FILE_PATH}...")
save_dict = {
    'folder_raw_accumulator': folder_raw_accumulator,
    'TOPO_TIMES': TOPO_TIMES
}
np.save(SAVE_FILE_PATH, save_dict)
print("Save complete.")

# --- 4. VISUALIZATION & PARAMETER EXTRACTION ---
if os.path.exists(SAVE_FILE_PATH):
    print(f"Loading results from historical run: {SAVE_FILE_PATH}...")
    loaded_data = np.load(SAVE_FILE_PATH, allow_pickle=True).item()
    folder_raw_accumulator = loaded_data['folder_raw_accumulator']
    TOPO_TIMES = loaded_data['TOPO_TIMES']
else:
    print("Warning: Save file not found. Visualizing current script memory parameters instead.")

# Ensure we are executing targeting a single active time window snippet
t = TOPO_TIMES[0]
print(f"\nFinalizing results and fitting curves for window: {t}ms...")

# 1. Single row, two columns configuration layout
fig, (ax_curve, ax_stat) = plt.subplots(1, 2, figsize=(22, 9))

# Storage for systematic parameter comparison
sig_params_comp = {t: {'anim': {'x0': [], 'k': []}, 'inanim': {'x0': [], 'k': []}}}

ANIM_COLOR = 'darkorange'
INANIM_COLOR = 'indigo'

anim_profiles, inanim_profiles = [], []
    
for f_id in ALL_FOLDERS:
    if not folder_raw_accumulator[t][f_id]: continue
    y_raw = np.mean(folder_raw_accumulator[t][f_id], axis=0)
    mn, mx = np.min(y_raw), np.max(y_raw)
    curr_color = ANIM_COLOR if f_id in ANIMATE_FOLDERS else INANIM_COLOR
    
    if f_id in ANIMATE_FOLDERS: anim_profiles.append(y_raw)
    else: inanim_profiles.append(y_raw)

    ax_curve.plot(MORPH_X, y_raw, color='black', alpha=0.25, lw=0.7, zorder=1)
    
    try:
        p_sig, _ = curve_fit(sigmoid_func, MORPH_X, y_raw, p0=[mx-mn, 11, 1, mn], bounds=([0, 5, -10, -15], [30, 16, 10, 15]))
        
        cat_key = 'anim' if f_id in ANIMATE_FOLDERS else 'inanim'
        sig_params_comp[t][cat_key]['x0'].append(p_sig[1])
        sig_params_comp[t][cat_key]['k'].append(p_sig[2])

        r2_s = get_adj_r2(y_raw, sigmoid_func(MORPH_X, *p_sig), 4)
        p_lin, _ = curve_fit(linear_func, MORPH_X, y_raw)
        r2_l = get_adj_r2(y_raw, linear_func(MORPH_X, *p_lin), 2)
        
        step_errs = []
        for ts in range(7, 15):
            pred = step_func(MORPH_X, ts, mn, mx)
            step_errs.append((ts, np.sum((y_raw-pred)**2)))
        best_ts = min(step_errs, key=lambda x: x[1])[0]
        r2_st = get_adj_r2(y_raw, step_func(MORPH_X, best_ts, mn, mx), 2)

        winner = 'Sig' # Set fixed or max dictionary lookups
        
        if winner == 'Sig': 
            ax_curve.plot(MORPH_X_FINE, sigmoid_func(MORPH_X_FINE, *p_sig), color=curr_color, alpha=0.6, lw=2.5, zorder=2)
        elif winner == 'Lin': 
            ax_curve.plot(MORPH_X, linear_func(MORPH_X, *p_lin), color=curr_color, alpha=0.6, lw=2.5, zorder=2)
        else: 
            ax_curve.step(MORPH_X, step_func(MORPH_X, best_ts, mn, mx), where='post', color=curr_color, alpha=0.6, lw=2.5, zorder=2)
    except: pass

# 2. Increased global font selections and removed mean curves mapping
ax_curve.set_title(f"Psychometric Model Fits ({t} ms)", fontweight='bold', fontsize=18, pad=12)
ax_curve.set_xlabel("Morph Level", fontsize=15, labelpad=10)
ax_curve.set_ylabel("Neural Distances / Evidence", fontsize=15, labelpad=10)

# 4. Strict discrete x-tick array placement mapping 1:21 directly
ax_curve.set_xticks(range(1, 22))
ax_curve.tick_params(axis='both', labelsize=13)

# --- 5. STATISTICAL BAR PLOTS PANEL ---
def add_sig_bracket(ax, x1, x2, p_val, means, sems, x_pos):
    if p_val >= 0.05: return
    stars = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else '*')
    idx1, idx2 = x_pos.index(x1), x_pos.index(x2)
    max_y = max(means[idx1] + sems[idx1], means[idx2] + sems[idx2])
    y_base = max_y + (max_y * 0.08 if max_y > 0 else 0.5)
    h = (max_y * 0.03 if max_y > 0 else 0.1)
    ax.plot([x1, x1, x2, x2], [y_base - h, y_base, y_base, y_base - h], lw=1.5, color='black')
    ax.text((x1 + x2) * 0.5, y_base, stars, ha='center', va='bottom', color='black', fontsize=16, fontweight='bold')

a_x0, i_x0 = sig_params_comp[t]['anim']['x0'], sig_params_comp[t]['inanim']['x0']
a_k, i_k = sig_params_comp[t]['anim']['k'], sig_params_comp[t]['inanim']['k']

t_x0, p_x0 = ttest_ind(a_x0, i_x0) if (a_x0 and i_x0) else (0, 1)
t_k, p_k = ttest_ind(a_k, i_k) if (a_k and i_k) else (0, 1)

means = [np.mean(a_x0) if a_x0 else 0, np.mean(i_x0) if i_x0 else 0, 
         np.mean(a_k) if a_k else 0, np.mean(i_k) if i_k else 0]

# 5. Verified Standard Error (SE) implementation structure across observations
sems = [np.std(a_x0)/np.sqrt(len(a_x0)) if a_x0 else 0, np.std(i_x0)/np.sqrt(len(i_x0)) if i_x0 else 0,
        np.std(a_k)/np.sqrt(len(a_k)) if a_k else 0, np.std(i_k)/np.sqrt(len(i_k)) if i_k else 0]

x_pos = [0, 1, 3, 4]
ax_stat.bar(x_pos, means, yerr=sems, color=[ANIM_COLOR, INANIM_COLOR, ANIM_COLOR, INANIM_COLOR], capsize=6, error_kw={'lw':1.5}, alpha=0.85)

add_sig_bracket(ax_stat, 0, 1, p_x0, means, sems, x_pos) 
add_sig_bracket(ax_stat, 3, 4, p_k, means, sems, x_pos)  

ax_stat.set_xticks(x_pos)
ax_stat.set_xticklabels(['x0 (Anim)', 'x0 (Inan)', 'k (Anim)', 'k (Inan)'], rotation=15, fontsize=13, fontweight='bold')
ax_stat.tick_params(axis='y', labelsize=13)
ax_stat.set_title(f"Parameter Comparison Metrics\nx0 p={p_x0:.3f} | k p={p_k:.3f}", fontsize=18, fontweight='bold', pad=12)
ax_stat.grid(axis='y', linestyle='--', alpha=0.4)

plt.suptitle(f"Psychometric Profile & Statistical Parameter Breakdown at {t}ms", fontsize=22, fontweight='bold', y=0.98)
plt.tight_layout()

# 6. Save as clean vector PDF for Illustrator scaling layouts
pdf_save_path = os.path.join(OUTPUT_BASE, f"psychometric_parameter_analysis_{t}ms.pdf")
print(f"Exporting publication-quality vector graphic to: {pdf_save_path}")
plt.savefig(pdf_save_path, format='pdf', bbox_inches='tight', dpi=600)

plt.show()