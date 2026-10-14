import numpy as np
import matplotlib.pyplot as plt
import mne
import os
import re
import gc
import pandas as pd
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# --- 1. CONFIGURATION & FUNCTIONS ---
INPUT_DIR = r'C:\Users\uqhkarim\OneDrive - The University of Queensland\Documents\Scientific\Projects\Pareidolia\EEG\Main data\Processed_Outputs'

# Combined folder IDs
ALL_FOLDERS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

OUTPUT_BASE = os.path.join(INPUT_DIR, 'Integrated_Psychometric_Results')
if not os.path.exists(OUTPUT_BASE): os.makedirs(OUTPUT_BASE)
SAVE_FILE_PATH = os.path.join(OUTPUT_BASE, 'behavioral_correlation_results.npy')

SUBJECT_SETTINGS = {
    "P00": {"exclude_blocks": [], "interpolate_channels": []}, "P01": {"exclude_blocks": [], "interpolate_channels": []}, 
    "P02": {"exclude_blocks": [], "interpolate_channels": []}, "P03": {"exclude_blocks": [], "interpolate_channels": []},
    "P04": {"exclude_blocks": [], "interpolate_channels": []}, "P05": {"exclude_blocks": [], "interpolate_channels": []},
    "P06": {"exclude_blocks": [], "interpolate_channels": []}, "P07": {"exclude_blocks": [], "interpolate_channels": []},
    "P08": {"exclude_blocks": [], "interpolate_channels": []}, "P09": {"exclude_blocks": [7, 8], "interpolate_channels": []},
    "P10": {"exclude_blocks": [], "interpolate_channels": []}, "P11": {"exclude_blocks": [], "interpolate_channels": []},
    "P12": {"exclude_blocks": [], "interpolate_channels": []}, "P13": {"exclude_blocks": [], "interpolate_channels": []},
    "P14": {"exclude_blocks": [1], "interpolate_channels": []}, "P15": {"exclude_blocks": [9], "interpolate_channels": []},
    "P16": {"exclude_blocks": [], "interpolate_channels": []}, "P17": {"exclude_blocks": [], "interpolate_channels": []},
    "P18": {"exclude_blocks": [], "interpolate_channels": []}, "P19": {"exclude_blocks": [], "interpolate_channels": []},
    "P20": {"exclude_blocks": [], "interpolate_channels": []}, "P21": {"exclude_blocks": [], "interpolate_channels": []},
    "P22": {"exclude_blocks": [], "interpolate_channels": []}, "P23": {"exclude_blocks": [1, 2, 3, 4, 5, 6, 7], "interpolate_channels": []},
    "P24": {"exclude_blocks": [], "interpolate_channels": []}, "P25": {"exclude_blocks": [], "interpolate_channels": []},
    "P26": {"exclude_blocks": [], "interpolate_channels": []}, "P27": {"exclude_blocks": [], "interpolate_channels": []},
    "P28": {"exclude_blocks": [], "interpolate_channels": []}, "P29": {"exclude_blocks": [], "interpolate_channels": []}
}

CORR_TIMES = [200]
WIN_HALF = 150
def sigmoid_norm(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0) / 21.0))

def parse_metadata(names):
    parsed = []
    for name in names:
        match = re.search(r'F(\d+)_S(\d+)', name)
        if match:
            f = int(match.group(1)); s = int(match.group(2))
            m = (s - 100) if 3 <= f <= 28 else None
            cat = 'morph' if f in ALL_FOLDERS else 'baseline'
            parsed.append({'f': f, 'm': m, 'cat': cat})
        else: parsed.append({'f': -1, 'm': None, 'cat': None})
    return parsed

# --- 2. INITIALIZATION ---
first_sub = next(iter(SUBJECT_SETTINGS))
temp = np.load(os.path.join(INPUT_DIR, f"{first_sub}.npy"), allow_pickle=True).item()
time_vec = temp['time_vector'] * 1000
all_block_points = []
    
# NEW DIAGNOSTIC LOGGER STRUCTURE
fit_diagnostics = {}

# Lists to aggregate subject-level catch trial summary stats
sub_catch_acc = []
sub_catch_rt = []

# --- 3. MAIN PROCESSING LOOP ---
for sub_id, settings in SUBJECT_SETTINGS.items():
    file_path = os.path.join(INPUT_DIR, f"{sub_id}.npy")
    if not os.path.exists(file_path): 
        print(f"Skipping {sub_id}: File not found on disk.")
        continue
    
    print(f">>> Processing {sub_id}...")
    data = np.load(file_path, allow_pickle=True).item()
    data_4d, meta = data['data_4d'], parse_metadata(list(data['condition_names']))
    catch_info = data.get('Catch_trials_info')
    
    # Initialize dictionary record for this subject
    fit_diagnostics[sub_id] = {'Passed': [], 'Failed_No_Catch': [], 'Failed_Curve_Fit': [], 'Excluded_By_Settings': settings['exclude_blocks']}

    if catch_info is None: 
        print(f"   -> EXCLUDE: {sub_id} missing Catch_trials_info data container entirely.")
        continue 

    total_blocks = data_4d.shape[3]
    valid_blocks = [b for b in range(1, total_blocks + 1) if b not in settings['exclude_blocks']]

    # --- CATCH TRIALS SUMMARY COMPONENT (PER SUBJECT) ---
    valid_catch = catch_info[catch_info['BlockNum'].isin(valid_blocks)]
    if not valid_catch.empty:
        # Subject accuracy across all catch trials in valid blocks
        acc = valid_catch['CorrectResponseFlag'].mean()
        sub_catch_acc.append(acc)
        
        # Subject mean RT for correctly answered catch trials in valid blocks
        correct_catch = valid_catch[valid_catch['CorrectResponseFlag'] == 1]
        if not correct_catch.empty:
            sub_catch_rt.append(correct_catch['RT'].mean())

    for b_num in valid_blocks:
        b_idx = b_num - 1 
        
        block_behavior = catch_info[(catch_info['BlockNum'] == b_num) & (catch_info['CorrectResponseFlag'] == 1)]
        if block_behavior.empty: 
            fit_diagnostics[sub_id]['Failed_No_Catch'].append(b_num)
            continue
        block_rt = block_behavior['RT'].mean()

        for target_t in CORR_TIMES:
            win_mask = (time_vec >= target_t - WIN_HALF) & (time_vec <= target_t + WIN_HALF)
            if not np.any(win_mask): continue

            best_slope_in_block = -np.inf
            
            train_obj_idx = [i for i, m in enumerate(meta) if m['f'] in [29, 30]]
            train_face_idx = [i for i, m in enumerate(meta) if m['f'] in [1, 2]]
            
            if train_obj_idx and train_face_idx:
                X_train_obj = np.nanmean(data_4d[:, win_mask, :, b_idx][:, :, train_obj_idx], axis=1).T
                X_train_face = np.nanmean(data_4d[:, win_mask, :, b_idx][:, :, train_face_idx], axis=1).T
                X_train = np.nan_to_num(np.vstack([X_train_obj, X_train_face]))
                y_train = np.concatenate([np.full(len(train_obj_idx), -1), np.ones(len(train_face_idx))])
                
                clf = make_pipeline(StandardScaler(), LinearSVC(dual='auto', max_iter=2000))
                clf.fit(X_train, y_train)
                
                m_distances = []
                for m in range(1, 22):
                    m_idx = [idx for idx, mt in enumerate(meta) if mt['m'] == m]
                    if not m_idx:
                        m_distances.append(0); continue
                        
                    pattern = np.nanmean(data_4d[:, win_mask, :, b_idx][:, :, m_idx], axis=(1, 2))
                    dist = clf.decision_function(np.nan_to_num(pattern).reshape(1, -1))[0]
                    m_distances.append(dist)
                
                ptp = np.ptp(m_distances)
                if ptp > 0:
                    norm_y = (m_distances - np.min(m_distances)) / ptp
                    try:
                        popt, _ = curve_fit(sigmoid_norm, np.arange(1, 22), norm_y, p0=[11, 1.0], maxfev=2000)
                        best_slope_in_block = popt[1]
                    except: 
                        pass
            
            if best_slope_in_block != -np.inf:
                all_block_points.append({
                    'Subject': sub_id, 'Block': b_num, 'Time': target_t, 'Slope': best_slope_in_block, 'RT': block_rt
                })
                fit_diagnostics[sub_id]['Passed'].append(b_num)
            else:
                fit_diagnostics[sub_id]['Failed_Curve_Fit'].append(b_num)

    # Output detailed evaluation log to terminal window instantly
    diag = fit_diagnostics[sub_id]
    print(f"   -> SUMMARY {sub_id}: Fits Kept={len(diag['Passed'])} | Failed Optimization={diag['Failed_Curve_Fit']} | Failed Behavioral Catch={diag['Failed_No_Catch']}")
    
    del data; gc.collect()

# --- REPORT CATCH TRIALS GROUP STATISTICS TO CONSOLE ---
print("\n==================================================")
print("             CATCH TRIALS PERFORMANCE             ")
print("==================================================")
if sub_catch_acc:
    mean_acc = np.mean(sub_catch_acc) * 100
    se_acc = (np.std(sub_catch_acc, ddof=1) / np.sqrt(len(sub_catch_acc))) * 100
    print(f"Catch Trials Accuracy : {mean_acc:.2f}% ± {se_acc:.2f}% (Mean ± SE, N={len(sub_catch_acc)})")
else:
    print("Catch Trials Accuracy : N/A")

if sub_catch_rt:
    mean_rt = np.mean(sub_catch_rt)
    se_rt = np.std(sub_catch_rt, ddof=1) / np.sqrt(len(sub_catch_rt))
    print(f"Catch Trials RT       : {mean_rt:.3f} s ± {se_rt:.3f} s (Mean ± SE, N={len(sub_catch_rt)})")
else:
    print("Catch Trials RT       : N/A")
print("==================================================\n")

# --- 3.5 DATA SAVING SECTION ---
print(f"Saving compiled analysis outputs to {SAVE_FILE_PATH}...")
save_dict = {
    'all_block_points': all_block_points,
    'CORR_TIMES': CORR_TIMES
}
np.save(SAVE_FILE_PATH, save_dict)
print("Save complete.")

# --- 4. VISUALIZATION ---
if os.path.exists(SAVE_FILE_PATH):
    print(f"Loading results from historical run: {SAVE_FILE_PATH}...")
    loaded_data = np.load(SAVE_FILE_PATH, allow_pickle=True).item()
    all_block_points = loaded_data['all_block_points']
    CORR_TIMES = loaded_data['CORR_TIMES']
else:
    print("Warning: Save file not found. Visualizing current script memory parameters instead.")

df_corr = pd.DataFrame(all_block_points)
n_times = len(CORR_TIMES)
n_cols = 4
n_rows = int(np.ceil(n_times / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
axes_flat = np.atleast_1d(axes).flatten()

for i, t in enumerate(CORR_TIMES):
    ax = axes_flat[i]
    subset = df_corr[df_corr['Time'] == t]
    
    if len(subset) > 0:
        sub_subset = subset.groupby('Subject', as_index=False)[['Slope', 'RT']].mean()
        
        if len(sub_subset) > 2:
            r, p = pearsonr(sub_subset['Slope'], sub_subset['RT'])
            
            sns.regplot(data=sub_subset, x='Slope', y='RT', ax=ax, 
                        scatter_kws={'color': 'black', 'alpha': 0.7, 's': 40}, 
                        line_kws={'color': 'black', 'lw': 2},
                        seed=42)
            
            for poly in ax.collections:
                if isinstance(poly, plt.matplotlib.collections.PolyCollection):
                    poly.set_color('gray')
                    poly.set_alpha(0.15)
                    
            ax.set_title(f"Time: {t}ms (Subject-level)\nN={len(sub_subset)} subjects, r={r:.3f}, p={p:.4e}", fontweight='bold')
    
    ax.set_xlabel("Subject Mean Neural Slope (k)")
    ax.set_ylabel("Subject Mean RT (s)")

for j in range(i + 1, len(axes_flat)): 
    fig.delaxes(axes_flat[j])

plt.tight_layout()

pdf_save_path = os.path.join(OUTPUT_BASE, "subject_slope_rt_correlation_plots.pdf")
print(f"Saving publication-grade vector graphic to: {pdf_save_path}")
plt.savefig(pdf_save_path, format='pdf', bbox_inches='tight', dpi=600)

plt.show()