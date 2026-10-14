import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import mne
from matplotlib.colors import ListedColormap

# --- 1. CONFIGURATION ---
INPUT_DIR = r'C:\Users\uqhkarim\OneDrive - The University of Queensland\Documents\Scientific\Projects\Pareidolia\EEG\Main data\Processed_Outputs'
# TOPO_TIMES = [-100, 0, 50, 80, 120, 160, 200, 260, 320, 400, 500, 600]
TOPO_TIMES = [-100, 0, 100, 200, 300, 400, 500, 600]
WIN_HALF = 25 
MORPH_X = np.arange(1, 22)

OUTPUT_BASE = os.path.join(INPUT_DIR, 'Integrated_Psychometric_Results')
if not os.path.exists(OUTPUT_BASE): os.makedirs(OUTPUT_BASE)
SAVE_FILE_PATH = os.path.join(OUTPUT_BASE, 'sensor_topo_analysis_results.npy')

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

# --- 2. HELPERS ---
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

def sigmoid_func(x, x0, k): return 1 / (1 + np.exp(-k * (x - x0)))
def linear_func(x, m, c): return m * x + c
def step_func(x, x0): return np.where(x < x0, 0, 1)

def get_adj_r2(y_true, y_pred, n_params):
    n = len(y_true)
    if n <= n_params + 1: return 0.0
    r2 = r2_score(y_true, y_pred)
    return max(0, 1 - (1 - r2) * (n - 1) / (n - n_params - 1))

# --- 3. PROCESSING ---
storage = {t: {
    'adj_r2_lin': [], 'adj_r2_sig': [], 'adj_r2_stp': [],
    'lin_slope': [], 'sig_steep': [], 'step_center': []
} for t in TOPO_TIMES}
ch_names_found = None

for sub_id, settings in SUBJECT_SETTINGS.items():
    file_path = os.path.join(INPUT_DIR, f"{sub_id}.npy")
    if not os.path.exists(file_path): continue
    
    print(f"Processing Subject Data: {sub_id}")
    data = np.load(file_path, allow_pickle=True).item()
    data_4d, time_ms = data['data_4d'], data['time_vector'] * 1000
    meta = parse_metadata(list(data['condition_names']))
    if ch_names_found is None: ch_names_found = list(data['channel_names'])
    
    n_chan = data_4d.shape[0]
    keep_idx = [i for i in range(data_4d.shape[3]) if (i + 1) not in settings['exclude_blocks']]
    
    train_obj = [i for i, m in enumerate(meta) if m['f'] in [29, 30]]
    train_face = [i for i, m in enumerate(meta) if m['f'] in [1, 2]]
    y_train = np.concatenate([np.full(len(train_obj)*len(keep_idx), -1), np.ones(len(train_face)*len(keep_idx))])

    for t_target in TOPO_TIMES:
        t_mask = (time_ms >= t_target - WIN_HALF) & (time_ms <= t_target + WIN_HALF)
        if not np.any(t_mask): continue
        
        res = {k: np.zeros(n_chan) for k in ['lin_s', 'sig_k', 'step_x', 'r2_l', 'r2_s', 'r2_st']}

        for ch in range(n_chan):
            profile = np.zeros(21)
            for m_level in range(1, 22):
                m_idx = [i for i, mt in enumerate(meta) if 3 <= mt['f'] <= 28 and mt['m'] == m_level]
                if m_idx:
                    profile[m_level-1] = np.nanmean(data_4d[ch, :, :, :][t_mask, :, :][:, m_idx, :][:, :, keep_idx])
            
            if np.ptp(profile) == 0: continue
            y_norm = (profile - np.min(profile)) / np.ptp(profile)
            
            try:
                p_lin, _ = curve_fit(linear_func, MORPH_X, y_norm)
                res['r2_l'][ch] = get_adj_r2(y_norm, linear_func(MORPH_X, *p_lin), 2)
                res['lin_s'][ch] = p_lin[0]
                
                p_sig, _ = curve_fit(sigmoid_func, MORPH_X, y_norm, p0=[11, 1], bounds=([5, 0.1], [16, 10]))
                res['r2_s'][ch] = get_adj_r2(y_norm, sigmoid_func(MORPH_X, *p_sig), 2)
                res['sig_k'][ch] = p_sig[1]

                step_errs = [np.sum((y_norm - step_func(MORPH_X, t))**2) for t in range(5, 17)]
                best_t = range(5, 17)[np.argmin(step_errs)]
                adj_r2_st = get_adj_r2(y_norm, step_func(MORPH_X, best_t), 1)
                res['r2_st'][ch] = adj_r2_st
                
                # REVISED STEP LOCATION: Mask with NaN if model does not outperform baseline flat line
                if adj_r2_st > 0:
                    res['step_x'][ch] = best_t
                else:
                    res['step_x'][ch] = np.nan
            except: pass

        storage[t_target]['lin_slope'].append(res['lin_s'])
        storage[t_target]['sig_steep'].append(res['sig_k'])
        storage[t_target]['step_center'].append(res['step_x'])
        storage[t_target]['adj_r2_lin'].append(res['r2_l'])
        storage[t_target]['adj_r2_sig'].append(res['r2_s'])
        storage[t_target]['adj_r2_stp'].append(res['r2_st'])

# --- 3.5 DATA SAVING SECTION ---
print(f"Saving compiled analysis outputs to {SAVE_FILE_PATH}...")
save_dict = {
    'storage': storage,
    'ch_names_found': ch_names_found,
    'TOPO_TIMES': TOPO_TIMES
}
np.save(SAVE_FILE_PATH, save_dict)
print("Save complete.")

# --- 4. REVISED VISUALIZATION (STABLE SCALING FIX) ---
# Loading snippet for module jumping
if os.path.exists(SAVE_FILE_PATH):
    print(f"Loading results from historical run: {SAVE_FILE_PATH}...")
    loaded_data = np.load(SAVE_FILE_PATH, allow_pickle=True).item()
    storage = loaded_data['storage']
    ch_names_found = loaded_data['ch_names_found']
    TOPO_TIMES = loaded_data['TOPO_TIMES']
else:
    print("Warning: Save file not found. Visualizing current script memory parameters instead.")

info = mne.create_info(ch_names=ch_names_found, sfreq=1000, ch_types='eeg')
info.set_montage('biosemi64')
WINNER_CMAP = ListedColormap(['#1f77b4', '#d62728', '#2ca02c'])

def plot_final_topos_stable(data_keys, row_labels, row_cmaps, main_title, is_r2=False):
    n_times = len(TOPO_TIMES)
    
    # 1. Tighter layout allocation to minimize gaps and maximize topo sizes
    fig, axes = plt.subplots(4, n_times + 1, figsize=(24, 13), 
                             gridspec_kw={'width_ratios': [1]*n_times + [0.04], 
                                          'wspace': 0.02, 'hspace': 0.15})
    
    R2_LIMITS = (0.0, 0.6) 

    for col, t in enumerate(TOPO_TIMES):
        avg_r2_l = np.nanmean(storage[t]['adj_r2_lin'], axis=0)
        avg_r2_s = np.nanmean(storage[t]['adj_r2_sig'], axis=0)
        avg_r2_st = np.nanmean(storage[t]['adj_r2_stp'], axis=0)
        
        winner_map = np.zeros(len(ch_names_found))
        for ch in range(len(ch_names_found)):
            winner_map[ch] = np.argmax([avg_r2_l[ch], avg_r2_s[ch], avg_r2_st[ch]]) + 1

        for row in range(4):
            ax = axes[row, col]
            if row < 3:
                d = np.array(np.nanmean(storage[t][data_keys[row]], axis=0), dtype=float)
                
                if is_r2:
                    curr_vlim = R2_LIMITS
                    if np.all(np.isnan(d)): d = np.zeros_like(d)
                else:
                    curr_vlim = (5, 16) if data_keys[row] == 'step_center' else (None, None)
                    if np.all(np.isnan(d)): d = np.full_like(d, 10.5)

                im, _ = mne.viz.plot_topomap(d, info, axes=ax, show=False, cmap=row_cmaps[row], 
                                             vlim=curr_vlim, contours=6, sensors=True, extrapolate='head')
                
                if col == n_times - 1:
                    cax = axes[row, -1]
                    norm = plt.Normalize(vmin=curr_vlim[0], vmax=curr_vlim[1]) if curr_vlim[0] is not None else None
                    sm = plt.cm.ScalarMappable(cmap=row_cmaps[row], norm=norm)
                    cb = plt.colorbar(sm, cax=cax)
                    cb.ax.tick_params(labelsize=12) # Clear colorbar numbers
            else:
                im, _ = mne.viz.plot_topomap(winner_map, info, axes=ax, show=False, cmap=WINNER_CMAP, 
                                             vlim=(1, 3), contours=0, sensors=True)
                if col == n_times - 1:
                    cax = axes[row, -1]
                    cb = plt.colorbar(im, cax=cax, ticks=[1.33, 2, 2.66])
                    cb.ax.set_yticklabels(['Lin', 'Sig', 'Step'], fontsize=14, fontweight='bold')
            
            # 2. Clearer, larger publication-grade font handling
            if row == 0: 
                ax.set_title(f"{t} ms", fontsize=20, fontweight='bold', pad=10)
            if col == 0: 
                ax.set_ylabel(row_labels[row], fontsize=16, fontweight='bold', labelpad=20)

    plt.suptitle(main_title, fontsize=24, fontweight='bold', y=0.97)
    
    # 3. Save as clean vector PDF for Adobe Illustrator layers
    sanitized_title = main_title.lower().replace(":", "").replace(" ", "_").replace("(", "").replace(")", "")
    pdf_save_path = os.path.join(OUTPUT_BASE, f"{sanitized_title}.pdf")
    
    print(f"Exporting publication-quality vector graphic to: {pdf_save_path}")
    plt.savefig(pdf_save_path, format='pdf', bbox_inches='tight', dpi=600)
    plt.show()

# --- EXECUTION ---
# Figure 1: Parameters
plot_final_topos_stable(
    ['lin_slope', 'sig_steep', 'step_center'],
    ["Linear Slope (m)", "Sigmoid Steep (k)", "Step Center (x0)", "Winning Strategy"],
    ['Blues', 'Reds', 'Greens'], 
    "Figure 1: Neural Psychometric Parameters",
    is_r2=False
)

# Figure 2: Adjusted R2
plot_final_topos_stable(
    ['adj_r2_lin', 'adj_r2_sig', 'adj_r2_stp'],
    ["R² Linear", "R² Sigmoid", "R² Step", "Winning Strategy"],
    ['Blues', 'Reds', 'Greens'], 
    "Figure 2: Model Goodness-of-Fit (Adjusted R²)",
    is_r2=True
)