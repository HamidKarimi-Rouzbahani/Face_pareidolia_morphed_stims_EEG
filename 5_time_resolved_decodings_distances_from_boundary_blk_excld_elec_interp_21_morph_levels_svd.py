import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.ndimage import gaussian_filter1d 
from scipy.optimize import curve_fit
import os
import re

# --- 1. CONFIGURATION ---
INPUT_DIR = r'C:\Users\uqhkarim\OneDrive - The University of Queensland\Documents\Scientific\Projects\Pareidolia\EEG\Main data\Processed_Outputs'
OUTPUT_BASE = os.path.join(INPUT_DIR, 'Integrated_Psychometric_Results')
if not os.path.exists(OUTPUT_BASE): os.makedirs(OUTPUT_BASE)
SAVE_FILE_PATH = os.path.join(OUTPUT_BASE, 'group_analysis_results.npy')

# --- SUBJECT-SPECIFIC SETTINGS ---
# Format: "Subject_ID": {"exclude_blocks": [list], "interpolate_channels": [list]}
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

SUBJECTS_TO_PROCESS = list(SUBJECT_SETTINGS.keys())

TIME_START_MS, TIME_END_MS, TIME_STEP_MS = -100, 600, 5
SMOOTHING_SIGMA = 3 
PSYCH_WINDOW = [50, 500] 

# Morph levels for test folders (3-28)
N_MORPH_LEVELS = 21

def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))

def parse_condition_names(names):
    parsed = []
    for name in names:
        match = re.search(r'F(\d+)_S(\d+)', name)
        if match:
            f_num = int(match.group(1))
            s_num = int(match.group(2))
            if 3 <= f_num <= 28:
                morph_lvl = s_num - 100 
            else:
                morph_lvl = None 
            parsed.append({'f': f_num, 'm': morph_lvl})
        else:
            parsed.append({'f': -1, 'm': -1})
    return parsed

group_3d_dist = []     
group_split_curves = [] 
group_psych_vectors = [] 

# --- 2. SUBJECT PROCESSING ---
for sub_id in SUBJECTS_TO_PROCESS:
    file_path = os.path.join(INPUT_DIR, f"{sub_id}.npy")
    if not os.path.exists(file_path): continue
    print(f"--- Analyzing Category Evidence: {sub_id} ---")

    data = np.load(file_path, allow_pickle=True).item()
    data_4d, time_vec = data['data_4d'], data['time_vector']
    condition_names = data['condition_names']
    chan_labels = data.get('chan_labels', []) # Assumes labels are stored in .npy
    
    n_chan, n_times, n_conds, n_trials = data_4d.shape
    time_ms = time_vec * 1000
    
    # A. INTERPOLATION LOGIC
    interp_list = SUBJECT_SETTINGS[sub_id].get("interpolate_channels", [])
    if interp_list and len(chan_labels) > 0:
        for ch_name in interp_list:
            if ch_name in chan_labels:
                idx = list(chan_labels).index(ch_name)
                # Simple interpolation: average of all other (presumably clean) channels
                other_indices = [i for i in range(n_chan) if i != idx]
                data_4d[idx, :, :, :] = np.mean(data_4d[other_indices, :, :, :], axis=0)
                print(f"  > Interpolated channel: {ch_name}")

    # B. BLOCK EXCLUSION LOGIC
    exclude_blocks = SUBJECT_SETTINGS[sub_id].get("exclude_blocks", [])
    exclude_idx = [b - 1 for b in exclude_blocks if 1 <= b <= n_trials]
    keep_idx = [i for i in range(n_trials) if i not in exclude_idx]
    n_trials_clean = len(keep_idx)
    
    meta = parse_condition_names(condition_names)
    
    train_f_idx = [i for i, m in enumerate(meta) if m['f'] in [1, 2]]
    train_o_idx = [i for i, m in enumerate(meta) if m['f'] in [29, 30]]
    train_idx = np.concatenate([train_o_idx, train_f_idx])
    
    y_train = np.repeat(np.concatenate([
        np.full(len(train_o_idx), -1), 
        np.ones(len(train_f_idx))
    ]), n_trials_clean)

    step = int(TIME_STEP_MS * (1 / (time_vec[1] - time_vec[0])) / 1000)
    t_idxs = np.where((time_ms >= TIME_START_MS) & (time_ms <= TIME_END_MS))[0][::step]
    decoded_time_ms = time_ms[t_idxs]

    test_folder_nums = range(3, 29)
    N_TEST_FOLDERS = len(test_folder_nums)
    sub_dist_res = np.zeros((len(t_idxs), N_TEST_FOLDERS, N_MORPH_LEVELS))
    
    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=42, max_iter=5000))

    for t_idx, t_sample in enumerate(t_idxs):
        X_train = data_4d[:, t_sample, train_idx, :][:, :, keep_idx].reshape(n_chan, -1).T
        train_keep = ~np.isnan(X_train).any(axis=1)
        clf.fit(X_train[train_keep], y_train[train_keep])
        
        for f_idx, f_num in enumerate(test_folder_nums):
            for m_lvl in range(1, N_MORPH_LEVELS + 1):
                idx = [i for i, m in enumerate(meta) if m['f'] == f_num and m['m'] == m_lvl]
                if idx:
                    X_test = data_4d[:, t_sample, idx[0], keep_idx].T
                    test_keep = ~np.isnan(X_test).any(axis=1)
                    if np.any(test_keep):
                        distances = clf.decision_function(X_test[test_keep])
                        sub_dist_res[t_idx, f_idx, m_lvl-1] = np.mean(distances)

    group_3d_dist.append(sub_dist_res)
    
    MORPH_SPLITS = [[1,7], [8,14], [15,21]] 
    sub_splits = np.zeros((3, len(t_idxs)))
    for s_i, (low, high) in enumerate(MORPH_SPLITS):
        sub_splits[s_i, :] = np.mean(sub_dist_res[:, :, low-1:high], axis=(1, 2))
    group_split_curves.append(sub_splits)

    psych_t_mask = (decoded_time_ms >= PSYCH_WINDOW[0]) & (decoded_time_ms <= PSYCH_WINDOW[1])
    sub_psych_raw = np.mean(sub_dist_res[psych_t_mask, :, :], axis=(0, 1))
    v_min, v_max = np.min(sub_psych_raw), np.max(sub_psych_raw)
    sub_psych_norm = (sub_psych_raw - v_min) / (v_max - v_min) if v_max > v_min else sub_psych_raw
    group_psych_vectors.append(sub_psych_norm)

# --- 2.5 DATA SAVING SECTION ---
print(f"Saving compiled analysis outputs to {SAVE_FILE_PATH}...")
save_dict = {
    'group_3d_dist': group_3d_dist,
    'group_split_curves': group_split_curves,
    'group_psych_vectors': group_psych_vectors,
    'decoded_time_ms': decoded_time_ms
}
np.save(SAVE_FILE_PATH, save_dict)
print("Save complete.")

# --- 3. FINAL VISUALIZATION ---
# Loading snippet for module jumping
if os.path.exists(SAVE_FILE_PATH):
    print(f"Loading results from historical run: {SAVE_FILE_PATH}...")
    loaded_data = np.load(SAVE_FILE_PATH, allow_pickle=True).item()
    group_3d_dist = loaded_data['group_3d_dist']
    group_split_curves = loaded_data['group_split_curves']
    group_psych_vectors = loaded_data['group_psych_vectors']
    decoded_time_ms = loaded_data['decoded_time_ms']
else:
    print("Warning: Save file not found. Visualizing current script memory parameters instead.")

if group_3d_dist:
    # Average across subjects: shape (time, folders, morph_levels)
    avg_3d = np.mean(group_3d_dist, axis=0) 
    # Average across folders for each morph level: shape (time, morph_levels)
    avg_per_level = np.mean(avg_3d, axis=1)
    
    # Calculate Standard Error for each level across subjects
    # First, average folders within each subject: result is (subjects, time, morph_levels)
    subject_avg_levels = [np.mean(sub, axis=1) for sub in group_3d_dist]
    se_per_level = np.std(subject_avg_levels, axis=0) / np.sqrt(len(group_3d_dist))

    # 1. Dropped bottom panel: Converted layout to a single dedicated plot window
    fig, ax = plt.subplots(figsize=(14, 9))

    # Create a custom colormap from Light Blue -> Navy Blue -> Black
    from matplotlib.colors import LinearSegmentedColormap
    custom_colors = ['#9ecae1', '#08519c', '#000000']
    cmap = LinearSegmentedColormap.from_list('morph_spectrum', custom_colors, N=N_MORPH_LEVELS)
    colors = [cmap(i) for i in np.linspace(0, 1, N_MORPH_LEVELS)]

    # Determine plotting order: sort indices based on distance from the mid-level morph (Level 11 / index 10)
    # This plots mid-levels first, leaving extremes on the top layer stack
    mid_idx = (N_MORPH_LEVELS - 1) // 2
    sorted_indices = sorted(range(N_MORPH_LEVELS), key=lambda x: abs(x - mid_idx))

    for i in sorted_indices:
        mu = gaussian_filter1d(avg_per_level[:, i], SMOOTHING_SIGMA)
        
        # Label selection for extremes and middle to avoid legend clutter
        if i in [0, mid_idx, N_MORPH_LEVELS - 1]:
            label_text = f"Level {i+1}"
        else:
            label_text = ""
            
        # Plot every level with uniform thickness, no shading, and 80% non-transparency
        ax.plot(decoded_time_ms, mu, color=colors[i], lw=3.0, alpha=0.8, label=label_text)

    ax.axhline(0, color='black', ls='-', lw=1, alpha=0.5)
    ax.axvline(0, color='black', ls='--', alpha=0.5)
    
    # 2. Significantly scaled up font size parameters for maximum visibility
    ax.set_title("Neural Category Evidence: 21 Morph Levels Spectrum", fontweight='bold', fontsize=20, pad=15)
    ax.set_ylabel("Distance to Boundary (Face Evidence)", fontsize=16, labelpad=12)
    ax.set_xlabel("Time (ms)", fontsize=16, labelpad=12)
    
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(loc='upper left', fontsize=14)

    # 3. Enhanced color bar formatting to list every individual tick from 1 to 21
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=21))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, ticks=range(1, 22))
    cbar.set_label('Morph Level (Object -> Face)', fontsize=16, labelpad=12)
    cbar.ax.tick_params(labelsize=13)

    plt.tight_layout()
    
    # 4. Save file out as high-resolution vector PDF for Illustrator scaling
    pdf_save_path = os.path.join(OUTPUT_BASE, "neural_category_evidence_spectrum.pdf")
    print(f"Saving publication-grade vector graphic to: {pdf_save_path}")
    plt.savefig(pdf_save_path, format='pdf', bbox_inches='tight', dpi=600)
    
    plt.show()