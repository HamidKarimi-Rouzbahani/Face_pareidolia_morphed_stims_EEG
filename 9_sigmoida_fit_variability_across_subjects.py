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
from matplotlib.lines import Line2D

# =============================================================================
# --- 1. CONFIGURATION ---
# =============================================================================
INPUT_DIR = r'C:\Users\uqhkarim\OneDrive - The University of Queensland\Documents\Scientific\Projects\Pareidolia\EEG\Main data\Processed_Outputs'
SAVE_PATH_FITS = os.path.join(INPUT_DIR, "pareidolia_fitted_results.npy")
SAVE_PATH_PDF = os.path.join(INPUT_DIR, "sigmoidal_fits_single_panel.pdf")

TOPO_TIMES = [200]
WIN_HALF = 25 
MORPH_X = np.arange(1, 22)
MORPH_X_FINE = np.linspace(1, 21, 100) # Smoother line for plot rendering

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

# =============================================================================
# --- 2. HELPER FUNCTIONS ---
# =============================================================================
def parse_metadata(names):
    parsed = []
    for name in names:
        match = re.search(r'F(\d+)_S(\d+)', name)
        if match:
            f = int(match.group(1)); s = int(match.group(2))
            m = (s - 100) if 3 <= f <= 28 else None
            parsed.append({'f': f, 'm': m})
        else: 
            parsed.append({'f': -1, 'm': None})
    return parsed

def sigmoid_func(x, L, x0, k, b): 
    return L / (1 + np.exp(-k * (x - x0))) + b

def linear_func(x, m, c): 
    return m * x + c

def step_func(x, x0, low, high): 
    return np.where(x < x0, low, high)

def get_adj_r2(y_true, y_pred, n_params):
    n = len(y_true)
    if n <= n_params + 1: 
        return -1.0
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_params - 1)

# =============================================================================
# --- 3. PROCESSING & DATA SAVING (DATA PIPELINE) ---
# =============================================================================
def run_processing_pipeline(save_path=SAVE_PATH_FITS):
    print("Starting processing and curve fitting pipeline...")
    results = {t: {} for t in TOPO_TIMES}

    for sub_id, settings in SUBJECT_SETTINGS.items():
        file_path = os.path.join(INPUT_DIR, f"{sub_id}.npy")
        if not os.path.exists(file_path): 
            continue
        
        print(f">>> Processing Subject: {sub_id}")
        data = np.load(file_path, allow_pickle=True).item()
        data_4d, time_ms = data['data_4d'], data['time_vector'] * 1000
        meta = parse_metadata(list(data['condition_names']))
        
        n_chan = data_4d.shape[0]
        keep_idx = [i for i in range(data_4d.shape[3]) if (i + 1) not in settings['exclude_blocks']]
        
        train_obj_f = [i for i, m in enumerate(meta) if m['f'] in [29, 30]]
        train_face_f = [i for i, m in enumerate(meta) if m['f'] in [1, 2]]
        y_train = np.concatenate([np.full(len(train_obj_f)*len(keep_idx), -1), 
                                  np.ones(len(train_face_f)*len(keep_idx))])

        for t_target in TOPO_TIMES:
            t_mask = (time_ms >= t_target - WIN_HALF) & (time_ms <= t_target + WIN_HALF)
            if not np.any(t_mask): 
                continue
            
            # Train Classifier
            X_train_avg = np.nanmean(data_4d[:, t_mask, :, :], axis=1)
            X_obj = X_train_avg[:, train_obj_f, :][:, :, keep_idx].reshape(n_chan, -1).T
            X_face = X_train_avg[:, train_face_f, :][:, :, keep_idx].reshape(n_chan, -1).T
            X_train = np.nan_to_num(np.vstack([X_obj, X_face]))
            
            clf = make_pipeline(SimpleImputer(), StandardScaler(), LinearSVC(dual='auto', max_iter=2000))
            clf.fit(X_train, y_train)

            # Test Raw across morphs
            sub_profile = np.zeros(21)
            for m in range(1, 22):
                m_idx_list = [i for i, mt in enumerate(meta) if mt['m'] == m]
                if m_idx_list:
                    X_test_raw = np.nanmean(data_4d[:, t_mask, :, :][:, :, m_idx_list, :], axis=(1, 2))
                    sub_profile[m-1] = np.mean(clf.decision_function(np.nan_to_num(X_test_raw[:, keep_idx].T)))
            
            y_raw = sub_profile
            mn, mx = np.min(y_raw), np.max(y_raw)
            
            try:
                # 1. Sigmoid Fit
                p_sig, _ = curve_fit(sigmoid_func, MORPH_X, y_raw, 
                                     p0=[mx-mn, 11, 1, mn], 
                                     bounds=([0, 5, 0.1, -10], [20, 16, 10, 10]))
                r2_s = get_adj_r2(y_raw, sigmoid_func(MORPH_X, *p_sig), 4)
                
                # 2. Linear Fit
                p_lin, _ = curve_fit(linear_func, MORPH_X, y_raw)
                r2_l = get_adj_r2(y_raw, linear_func(MORPH_X, *p_lin), 2)
                
                # 3. Step Fit
                step_params = []
                for t_step in range(7, 15):
                    pred = step_func(MORPH_X, t_step, mn, mx)
                    step_params.append({'t': t_step, 'err': np.sum((y_raw-pred)**2), 'low': mn, 'high': mx})
                best_s = min(step_params, key=lambda x: x['err'])
                r2_st = get_adj_r2(y_raw, step_func(MORPH_X, best_s['t'], mn, mx), 2)

                # Determine Winner
                r2_map = {
                    'Sigmoid': (r2_s, p_sig), 
                    'Linear': (r2_l, p_lin), 
                    'Step': (r2_st, best_s)
                }
                winner_key = max(r2_map, key=lambda k: r2_map[k][0])
                
                results[t_target][sub_id] = {
                    'y_raw': y_raw, 
                    'winner': winner_key, 
                    'sig_params': p_sig,  # Stores [L, x0, k, b]
                    'win_params': r2_map[winner_key][1],
                    'all_r2': {'Sigmoid': r2_s, 'Linear': r2_l, 'Step': r2_st}
                }
            except Exception as e:
                print(f"Fitting failed for subject {sub_id}: {e}")

        del data; gc.collect()

    # Calculate Slope Statistics (Mean and Variance of k across participants)
    for t_target in TOPO_TIMES:
        slopes = [res['sig_params'][2] for res in results[t_target].values() if 'sig_params' in res]
        if slopes:
            mean_k = np.mean(slopes)
            var_k = np.var(slopes, ddof=1)
            print(f"\n--- Fitting Summary for {t_target}ms ---")
            print(f"Mean Slope (k): {mean_k:.4f}")
            print(f"Variance of Slopes (k): {var_k:.4f}")
            results[t_target]['summary_stats'] = {'mean_k': mean_k, 'var_k': var_k, 'all_k': slopes}

    # Save dictionary to disk via safe atomic replacement
    tmp_save_path = save_path.replace('.npy', '_tmp.npy')
    np.save(tmp_save_path, results)
    os.replace(tmp_save_path, save_path)
    print(f"Processing complete. Results saved to: {save_path}\n")
    return results

# =============================================================================
# --- 4. INDEPENDENT PLOTTING FUNCTION ---
# =============================================================================
def plot_sigmoidal_single_panel(data_path=SAVE_PATH_FITS, pdf_out_path=SAVE_PATH_PDF, target_time=200):
    """Loads pre-saved processed data from disk and generates the single-panel figure."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}. Run processing first.")
        
    print(f"Loading data for plotting from {data_path}...")
    results = np.load(data_path, allow_pickle=True).item()
    t_data = results[target_time]
    
    # Extract subjects
    subjects_data = {k: v for k, v in t_data.items() if k != 'summary_stats'}
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    all_raw_y = []
    all_sig_fits = []
    
    # 1. Plot individual raw profiles & fitted sigmoids
    for sub_id, res in subjects_data.items():
        y_raw = res['y_raw']
        p_sig = res['sig_params']
        all_raw_y.append(y_raw)
        
        # Individual raw points/lines
        ax.plot(MORPH_X, y_raw, color='gray', alpha=0.35, lw=0.8)
        
        # Individual Sigmoidal Fit
        sig_fit_fine = sigmoid_func(MORPH_X_FINE, *p_sig)
        ax.plot(MORPH_X_FINE, sig_fit_fine, color='crimson', alpha=0.35, lw=1.0)
        all_sig_fits.append(sig_fit_fine)

    # 2. Grand Average Data and Mean Sigmoidal Fit
    if all_raw_y:
        grand_mean_raw = np.mean(all_raw_y, axis=0)
        grand_mean_sig_fit = np.mean(all_sig_fits, axis=0)
        
        # Plot Grand Mean Raw Data Points
        ax.plot(MORPH_X, grand_mean_raw, color='black', lw=2.5, marker='o', 
                markersize=6, label='Grand Mean Raw Data', zorder=10)
        
        # Plot Grand Mean Fitted Sigmoid
        ax.plot(MORPH_X_FINE, grand_mean_sig_fit, color='darkred', lw=3.5, 
                linestyle='-', label='Grand Mean Sigmoidal Fit', zorder=11)

    # Format Axes and Labels
    ax.set_title(f"Sigmoidal Fits to Decision Profiles ({target_time} ms)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Morph Level", fontsize=12)
    ax.set_ylabel("Decision Boundary Distance (a.u.)", fontsize=12)
    ax.set_xticks(MORPH_X)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Display slope statistics on plot
    if 'summary_stats' in t_data:
        stats = t_data['summary_stats']
        stat_text = f"Mean Slope ($k$): {stats['mean_k']:.3f}\nVariance ($k$): {stats['var_k']:.3f}"
        ax.text(0.05, 0.88, stat_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Custom Legend
    legend_elements = [
        Line2D([0], [0], color='black', lw=2.5, marker='o', label='Grand Mean Raw Data'),
        Line2D([0], [0], color='darkred', lw=3.5, label='Grand Mean Sigmoidal Fit'),
        Line2D([0], [0], color='crimson', alpha=0.5, lw=1.2, label='Individual Sigmoidal Fit'),
        Line2D([0], [0], color='gray', alpha=0.5, lw=0.8, label='Individual Raw Data')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=10)
    
    plt.tight_layout()
    
    # Save output plot as PDF
    plt.savefig(pdf_out_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {pdf_out_path}")
    plt.show()

# =============================================================================
# --- 5. MAIN EXECUTION WITH RESUME / LOAD LOGIC ---
# =============================================================================
if __name__ == "__main__":
    # Check if the saved numpy file exists; if not, execute the processing pipeline
    if not os.path.exists(SAVE_PATH_FITS):
        print(f"Saved dataset not found at {SAVE_PATH_FITS}. Running computation...")
        run_processing_pipeline()
    else:
        print(f"Found processed dataset at {SAVE_PATH_FITS}. Skipping computation.")

    # Always execute independent plotting block using saved data
    plot_sigmoidal_single_panel()