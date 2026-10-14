import numpy as np
import os
import re
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, ttest_1samp, ttest_rel
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION ---
# INPUT_DIR = '/QRISdata/Q7503/Pareidolea/EEG/Main data/Processed_Outputs'
INPUT_DIR = r'C:\Users\uqhkarim\OneDrive - The University of Queensland\Documents\Scientific\Projects\Pareidolia\EEG\Main data\Processed_Outputs'

SAVE_PATH_FLOW = os.path.join(INPUT_DIR, 'group_RCA_TRANS_FOLDER_LEVEL_Scalar_all_stats_HR_Granger_params.npy')
TIME_POINTS = np.arange(-100, 601,5)
DELAYS_MS = np.arange(5, 201, 20) 
N_MORPH = 21
FIT_WINDOW_HALF = 5 

ANT_PREFIXES = ('Fp', 'AF', 'F', 'FC')
POST_PREFIXES = ('CP', 'P', 'PO', 'O')

X0_RANGE = np.arange(5, 16) 
K_RANGE = [0.1, 0.5, 1, 2, 5] 

# --- 2. RDM HELPERS ---
def get_triu(mat): return mat[np.triu_indices(N_MORPH, k=1)]

def make_linear_rdm(x0):
    vals = (np.arange(N_MORPH) / x0).reshape(-1, 1)
    return get_triu(squareform(pdist(vals, metric='cityblock')))

def make_step_rdm(x0):
    labels = np.where(np.arange(N_MORPH) < x0, 0, 1).reshape(-1, 1)
    return get_triu(squareform(pdist(labels, metric='hamming')))

def make_sigmoid_rdm(x0, k):
    vals = (1 / (1 + np.exp(-k * (np.arange(N_MORPH) - x0)))).reshape(-1, 1)
    return get_triu(squareform(pdist(vals, metric='cityblock')))

# --- 3. CORE ANALYTICAL HELPERS ---
def get_partial_corr(target_vec, nuisance_vecs, neural_vec):
    try:
        X = np.column_stack([c.flatten() for c in nuisance_vecs])
        reg_t = LinearRegression().fit(X, target_vec.flatten())
        res_t = target_vec.flatten() - reg_t.predict(X)
        reg_n = LinearRegression().fit(X, neural_vec.flatten())
        res_n = neural_vec.flatten() - reg_n.predict(X)
        rho, _ = spearmanr(res_t, res_n)
        return rho if not np.isnan(rho) else 0
    except: return 0

def get_bf10(data):
    """JZS Bayes Factor approximation against zero."""
    try:
        t_stats, _ = ttest_1samp(data, 0, nan_policy='omit')
        bf = np.exp((t_stats**2 - np.log(len(data))) / 2)
        return bf if not np.isnan(bf) else 1.0
    except: return 1.0

# --- 4. PROCESSING WITH RESUME LOGIC ---
if os.path.exists(SAVE_PATH_FLOW):
    print(f"Resuming from {SAVE_PATH_FLOW}...")
    group_data = np.load(SAVE_PATH_FLOW, allow_pickle=True).item()
    processed_count = len(group_data['ff'])
    
    # Backward compatibility checking for parameter tracking keys
    if 'best_params' not in group_data:
        group_data['best_params'] = {
            'linear_x0': [],
            'step_x0': [],
            'sigmoid_x0': [],
            'sigmoid_k': []
        }
else:
    group_data = {'ff': [], 'fb': [], 
                  'ff_linear': [], 'fb_linear': [],
                  'ff_sigmoid': [], 'fb_sigmoid': [],
                  'ff_step': [], 'fb_step': [],
                  'coding_whole': {m: [] for m in ['Linear', 'Sigmoid', 'Step']},
                  'coding_ant': {m: [] for m in ['Linear', 'Sigmoid', 'Step']},
                  'coding_post': {m: [] for m in ['Linear', 'Sigmoid', 'Step']},
                  'best_params': {
                      'linear_x0': [],
                      'step_x0': [],
                      'sigmoid_x0': [],
                      'sigmoid_k': []
                  }}
    processed_count = 0

subs = [f"P{i:02d}" for i in range(30)]

for idx, sub_id in enumerate(subs):
    if idx < processed_count:
        continue

    f_path = os.path.join(INPUT_DIR, f"{sub_id}.npy")
    if not os.path.exists(f_path): continue
    
    print(f"Processing Folder-Level CV: {sub_id}")
    d = np.load(f_path, allow_pickle=True).item()
    data_4d, time_ms = d['data_4d'], d['time_vector'] * 1000
    n_trials = data_4d.shape[3]
    ch_names = list(d['channel_names'])
    
    # 1. Identify Folders and Morph Levels
    folder_ids = np.array([int(re.search(r'F(\d+)', n).group(1)) for n in d['condition_names']])
    morph_levels = np.array([int(re.search(r'S(\d+)', n).group(1)) - 100 for n in d['condition_names']])
    test_idx = np.where((folder_ids >= 3) & (folder_ids <= 28))[0]
    
    # 2. Regional Indices
    a_ch = [i for i, n in enumerate(ch_names) if any(n.startswith(p) for p in ANT_PREFIXES)]
    p_ch = [i for i, n in enumerate(ch_names) if any(n.startswith(p) for p in POST_PREFIXES)]
    w_ch = np.arange(len(ch_names))
    
    # 3. Even/Odd Trial Splits
    t_half1, t_half2 = np.arange(0, n_trials, 2), np.arange(1, n_trials, 2)

    sub_res = {'ff': np.zeros((len(DELAYS_MS), len(TIME_POINTS))), 'fb': np.zeros((len(DELAYS_MS), len(TIME_POINTS))),
               'ff_linear': np.zeros((len(DELAYS_MS), len(TIME_POINTS))), 'fb_linear': np.zeros((len(DELAYS_MS), len(TIME_POINTS))),
               'ff_sigmoid': np.zeros((len(DELAYS_MS), len(TIME_POINTS))), 'fb_sigmoid': np.zeros((len(DELAYS_MS), len(TIME_POINTS))),
               'ff_step': np.zeros((len(DELAYS_MS), len(TIME_POINTS))), 'fb_step': np.zeros((len(DELAYS_MS), len(TIME_POINTS))),
               'c_ant': {m: np.zeros(len(TIME_POINTS)) for m in ['Linear', 'Sigmoid', 'Step']},
               'c_post': {m: np.zeros(len(TIME_POINTS)) for m in ['Linear', 'Sigmoid', 'Step']},
               'c_whole': {m: np.zeros(len(TIME_POINTS)) for m in ['Linear', 'Sigmoid', 'Step']},
               'linear_x0': np.zeros(len(TIME_POINTS)),
               'step_x0': np.zeros(len(TIME_POINTS)),
               'sigmoid_x0': np.zeros(len(TIME_POINTS)),
               'sigmoid_k': np.zeros(len(TIME_POINTS))}

    # Pre-compute Neural RDMs
    neural_rdms = {f: {reg: {} for reg in ['A', 'P', 'W']} for f in [0, 1]}
    for f_idx, trials_idx in enumerate([t_half1, t_half2]):
        for t in TIME_POINTS:
            t_mask = (time_ms >= t - FIT_WINDOW_HALF) & (time_ms <= t + FIT_WINDOW_HALF)
            def get_reg_rdm(c_idx):
                patterns = []
                for m in range(1, N_MORPH + 1):
                    m_idx = test_idx[morph_levels[test_idx] == m]
                    p = np.nanmean(data_4d[c_idx, :, :, :][:, t_mask, :, :][:, :, m_idx, :][:, :, :, trials_idx], axis=(1, 2, 3))
                    patterns.append(p)
                return pdist(np.array(patterns), metric='euclidean')
            neural_rdms[f_idx]['A'][t] = get_reg_rdm(a_ch)
            neural_rdms[f_idx]['P'][t] = get_reg_rdm(p_ch)
            neural_rdms[f_idx]['W'][t] = get_reg_rdm(w_ch)

    # 4. Grid Search and Analysis Block
    scalar_waveforms = {f: {reg: {m: np.zeros(len(TIME_POINTS)) for m in ['Linear', 'Sigmoid', 'Step']} for reg in ['A', 'P', 'W']} for f in [0, 1]}
    
    for i_t, t in enumerate(TIME_POINTS):
        for train_f, test_f in [(0, 1), (1, 0)]:
            best_models = {'Linear': None, 'Step': None, 'Sigmoid': None}
            max_rs = {m: -1.0 for m in best_models}
            fit_neural = neural_rdms[train_f]['W'][t]
            
            # Temporary holders to track parameters for this cross-validation iteration
            best_p = {'Linear': {'x0': None}, 'Step': {'x0': None}, 'Sigmoid': {'x0': None, 'k': None}}

            for x0 in X0_RANGE:
                l_rdm = make_linear_rdm(x0); r_l = spearmanr(l_rdm, fit_neural)[0]
                if r_l > max_rs['Linear']: 
                    max_rs['Linear'], best_models['Linear'] = r_l, l_rdm
                    best_p['Linear']['x0'] = x0
                    
                s_rdm = make_step_rdm(x0); r_s = spearmanr(s_rdm, fit_neural)[0]
                if r_s > max_rs['Step']: 
                    max_rs['Step'], best_models['Step'] = r_s, s_rdm
                    best_p['Step']['x0'] = x0
                    
                for k in K_RANGE:
                    sig_rdm = make_sigmoid_rdm(x0, k); r_sig = spearmanr(sig_rdm, fit_neural)[0]
                    if r_sig > max_rs['Sigmoid']: 
                        max_rs['Sigmoid'], best_models['Sigmoid'] = r_sig, sig_rdm
                        best_p['Sigmoid']['x0'] = x0
                        best_p['Sigmoid']['k'] = k

            # Accumulate and average model parameters over cross-validation cycles (split 1 and split 2)
            sub_res['linear_x0'][i_t] += best_p['Linear']['x0'] / 2
            sub_res['step_x0'][i_t] += best_p['Step']['x0'] / 2
            sub_res['sigmoid_x0'][i_t] += best_p['Sigmoid']['x0'] / 2
            sub_res['sigmoid_k'][i_t] += best_p['Sigmoid']['k'] / 2

            # Calculate and store localized, isolated scalars into tracking waveforms
            for reg, res_k in [('A', 'c_ant'), ('P', 'c_post'), ('W', 'c_whole')]:
                eval_neural = neural_rdms[test_f][reg][t]
                
                s_lin = get_partial_corr(best_models['Linear'], [best_models['Sigmoid'], best_models['Step']], eval_neural)
                s_sig = get_partial_corr(best_models['Sigmoid'], [best_models['Linear'], best_models['Step']], eval_neural)
                s_stp = get_partial_corr(best_models['Step'], [best_models['Linear'], best_models['Sigmoid']], eval_neural)
                
                # Save into Row 1 main tracking dictionary
                sub_res[res_k]['Linear'][i_t] += s_lin / 2
                sub_res[res_k]['Sigmoid'][i_t] += s_sig / 2
                sub_res[res_k]['Step'][i_t] += s_stp / 2
                
                # Populate execution timeline maps
                scalar_waveforms[test_f][reg]['Linear'][i_t] = s_lin
                scalar_waveforms[test_f][reg]['Sigmoid'][i_t] = s_sig
                scalar_waveforms[test_f][reg]['Step'][i_t] = s_stp

    # 5. Scalar Directed Flow Tracking Analysis (Row 2 Calculations with Raw Arrays)
    for i_t, t in enumerate(TIME_POINTS):
        for train_f, test_f in [(0, 1), (1, 0)]:
            
            # Re-generate models to guarantee structural alignment with the current time step's selection parameters
            best_models = {'Linear': None, 'Step': None, 'Sigmoid': None}
            max_rs = {m: -1.0 for m in best_models}
            fit_neural = neural_rdms[train_f]['W'][t]
            for x0 in X0_RANGE:
                l_rdm = make_linear_rdm(x0); r_l = spearmanr(l_rdm, fit_neural)[0]
                if r_l > max_rs['Linear']: max_rs['Linear'], best_models['Linear'] = r_l, l_rdm
                s_rdm = make_step_rdm(x0); r_s = spearmanr(s_rdm, fit_neural)[0]
                if r_s > max_rs['Step']: max_rs['Step'], best_models['Step'] = r_s, s_rdm
                for k in K_RANGE:
                    sig_rdm = make_sigmoid_rdm(x0, k); r_sig = spearmanr(sig_rdm, fit_neural)[0]
                    if r_sig > max_rs['Sigmoid']: max_rs['Sigmoid'], best_models['Sigmoid'] = r_sig, sig_rdm

            for d_idx, dly in enumerate(DELAYS_MS):
                t_p = t - dly
                if t_p in TIME_POINTS:
                    i_t_past = np.where(TIME_POINTS == t_p)[0][0]
                    
                    # --- Core Destination Specificity Controls: Isolate updates from destination past ---
                    past_dest_rdm_ant = neural_rdms[test_f]['A'][t_p].flatten().reshape(-1, 1)
                    past_dest_rdm_post = neural_rdms[test_f]['P'][t_p].flatten().reshape(-1, 1)
                    
                    eval_neural_ant = neural_rdms[test_f]['A'][t].flatten()
                    eval_neural_post = neural_rdms[test_f]['P'][t].flatten()
                    
                    # Regress destination past out of destination present
                    res_neural_ant = eval_neural_ant - LinearRegression().fit(past_dest_rdm_ant, eval_neural_ant).predict(past_dest_rdm_ant)
                    res_neural_post = eval_neural_post - LinearRegression().fit(past_dest_rdm_post, eval_neural_post).predict(past_dest_rdm_post)
                    
                    # Regress destination past out of current structural model representations
                    res_models_ant = {}
                    res_models_post = {}
                    for m_name, m_vec in best_models.items():
                        v_flat = m_vec.flatten()
                        res_models_ant[m_name] = v_flat - LinearRegression().fit(past_dest_rdm_ant, v_flat).predict(past_dest_rdm_ant)
                        res_models_post[m_name] = v_flat - LinearRegression().fit(past_dest_rdm_post, v_flat).predict(past_dest_rdm_post)
                    
                    # Recalculate unique partial variances on the history-stripped residuals
                    now_ant_linear = get_partial_corr(res_models_ant['Linear'], [res_models_ant['Sigmoid'], res_models_ant['Step']], res_neural_ant)
                    now_ant_sigmoid = get_partial_corr(res_models_ant['Sigmoid'], [res_models_ant['Linear'], res_models_ant['Step']], res_neural_ant)
                    now_ant_step = get_partial_corr(res_models_ant['Step'], [res_models_ant['Linear'], res_models_ant['Sigmoid']], res_neural_ant)
                    
                    now_post_linear = get_partial_corr(res_models_post['Linear'], [res_models_post['Sigmoid'], res_models_post['Step']], res_neural_post)
                    now_post_sigmoid = get_partial_corr(res_models_post['Sigmoid'], [res_models_post['Linear'], res_models_post['Step']], res_neural_post)
                    now_post_step = get_partial_corr(res_models_post['Step'], [res_models_post['Linear'], res_models_post['Sigmoid']], res_neural_post)
                    
                    # --- Transformed Flow Elements ---
                    past_post_linear = scalar_waveforms[test_f]['P']['Linear'][i_t_past]
                    past_ant_sigmoid = scalar_waveforms[test_f]['A']['Sigmoid'][i_t_past]
                    
                    sub_res['ff'][d_idx, i_t] += (past_post_linear * now_ant_sigmoid) / 2
                    sub_res['fb'][d_idx, i_t] += (past_ant_sigmoid * now_post_linear) / 2

                    # --- Non-Transformed Flow Channels ---
                    # 1. Linear -> Linear
                    sub_res['ff_linear'][d_idx, i_t] += (scalar_waveforms[test_f]['P']['Linear'][i_t_past] * now_ant_linear) / 2
                    sub_res['fb_linear'][d_idx, i_t] += (scalar_waveforms[test_f]['A']['Linear'][i_t_past] * now_post_linear) / 2
                    
                    # 2. Sigmoid -> Sigmoid
                    sub_res['ff_sigmoid'][d_idx, i_t] += (scalar_waveforms[test_f]['P']['Sigmoid'][i_t_past] * now_ant_sigmoid) / 2
                    sub_res['fb_sigmoid'][d_idx, i_t] += (scalar_waveforms[test_f]['A']['Sigmoid'][i_t_past] * now_post_sigmoid) / 2
                    
                    # 3. Step -> Step
                    sub_res['ff_step'][d_idx, i_t] += (scalar_waveforms[test_f]['P']['Step'][i_t_past] * now_ant_step) / 2
                    sub_res['fb_step'][d_idx, i_t] += (scalar_waveforms[test_f]['A']['Step'][i_t_past] * now_post_step) / 2

    # Append subject results and Save
    for k in group_data['coding_whole']:
        group_data['coding_whole'][k].append(sub_res['c_whole'][k])
        group_data['coding_ant'][k].append(sub_res['c_ant'][k])
        group_data['coding_post'][k].append(sub_res['c_post'][k])
    group_data['ff'].append(sub_res['ff'])
    group_data['fb'].append(sub_res['fb'])
    group_data['ff_linear'].append(sub_res['ff_linear'])
    group_data['fb_linear'].append(sub_res['fb_linear'])
    group_data['ff_sigmoid'].append(sub_res['ff_sigmoid'])
    group_data['fb_sigmoid'].append(sub_res['fb_sigmoid'])
    group_data['ff_step'].append(sub_res['ff_step'])
    group_data['fb_step'].append(sub_res['fb_step'])
    
    # Save parameter tracks safely into the dictionary
    group_data['best_params']['linear_x0'].append(sub_res['linear_x0'])
    group_data['best_params']['step_x0'].append(sub_res['step_x0'])
    group_data['best_params']['sigmoid_x0'].append(sub_res['sigmoid_x0'])
    group_data['best_params']['sigmoid_k'].append(sub_res['sigmoid_k'])
    
    # --- FIXED ATOMIC SAVE ---
    tmp_save_path = SAVE_PATH_FLOW.replace('.npy', '_tmp.npy') 
    np.save(tmp_save_path, group_data)
    os.replace(tmp_save_path, SAVE_PATH_FLOW)
    print(f"Subject {sub_id} complete and saved.")
    
# --- 5. VISUALIZATION ---
if os.path.exists(SAVE_PATH_FLOW):
    BF_threshold = 3
    n = 5  # Define your smoothing points here (n * 5ms)
    r_scale = 0.5  # Narrower Cauchy prior scale (default JZS is sqrt(2)/2 approx 0.707)
                   # Lower values make the test more lenient/sensitive to small-to-moderate effects.
    
    # Identify indices corresponding to the pre-stimulus baseline window
    baseline_indices = np.where((TIME_POINTS >= -100) & (TIME_POINTS <= 0))[0]
    
    # Automatic color-averaging helper function
    def get_blended_color(c1, c2):
        if c1 == c2:
            return 'black'
        try:
            from matplotlib.colors import to_rgb, to_hex
            rgb1, rgb2 = np.array(to_rgb(c1)), np.array(to_rgb(c2))
            return to_hex((rgb1 + rgb2) / 2.0)
        except:
            return 'tab:purple' # Safe fallback

    # Updated true JZS Bayes Factor calculation engine with parameterized prior scale r
    def get_bf10_jzs(t_stats, N, r=r_scale):
        if N <= 1 or np.isnan(t_stats) or np.isinf(t_stats):
            return 1.0
        
        df = N - 1
        numerator = (1 + (t_stats**2) / df) ** (-(df + 1) / 2)
        
        def jzs_integrand(g):
            scale_factor = 1 + N * (r**2) * g
            return (
                ((scale_factor) ** -0.5) * ((1 + (t_stats**2) / ((scale_factor) * df)) ** (-(df + 1) / 2)) * (2 * np.pi) ** -0.5 * (g ** -1.5) * np.exp(-1 / (2 * g))
            )
        
        try:
            from scipy.integrate import quad
            denominator, _ = quad(jzs_integrand, 0, np.inf)
            if denominator > 0:
                return np.clip(denominator / numerator, 0.0, 1e10)
        except:
            pass
        return 1.0

    # Helper function to smooth line traces without shifting phase/time
    def smooth_trace(trace, window_size):
        if window_size <= 1:
            return trace
        padded = np.pad(trace, pad_width=window_size//2, mode='edge')
        smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
        if len(smoothed) > len(trace):
            smoothed = smoothed[:len(trace)]
        elif len(smoothed) < len(trace):
            smoothed = np.pad(smoothed, (0, len(trace)-len(smoothed)), mode='edge')
        return smoothed

    group_data = np.load(SAVE_PATH_FLOW, allow_pickle=True).item()
    N = len(group_data['ff'])
    colors = {'Linear': 'tab:blue', 'Sigmoid': 'tab:red', 'Step': 'tab:green'}
    fig_out_dir = os.path.dirname(SAVE_PATH_FLOW)
    
    # ==========================================
    # FIGURE 1: INFORMATION CODING DYNAMICS (Scaled down)
    # ==========================================
    fig1 = plt.figure(figsize=(11.3, 7.8)) # Multiplied 16x11 by 0.707
    gs1 = fig1.add_gridspec(2, 2, height_ratios=[2, 1])
    regs = [('coding_post', 'Posterior'), ('coding_ant', 'Anterior')]
    
    ax_top_axes1 = []
    ax_bot_axes1 = []
    for idx, (rk, title) in enumerate(regs):
        if idx == 0:
            ax_top = fig1.add_subplot(gs1[0, idx])
            ax_bot = fig1.add_subplot(gs1[1, idx], sharex=ax_top)
        else:
            ax_top = fig1.add_subplot(gs1[0, idx], sharey=ax_top_axes1[0])
            ax_bot = fig1.add_subplot(gs1[1, idx], sharex=ax_top, sharey=ax_bot_axes1[0])
        ax_top_axes1.append(ax_top)
        ax_bot_axes1.append(ax_bot)
        
        # Stimulus onset line (Reduced thickness)
        ax_top.axvline(0, color='gray', ls='--', lw=1.2, alpha=0.7)
        ax_bot.axvline(0, color='gray', ls='--', lw=1.2, alpha=0.7)
        
        model_data_dict = {}
        
        for m in colors:
            d = np.array(group_data[rk][m])
            model_data_dict[m] = d
            mu, se = np.nanmean(d, 0), np.nanstd(d, 0)/np.sqrt(N)
            ax_top.fill_between(TIME_POINTS, mu-se, mu+se, color=colors[m], alpha=0.1)
            
            bfs = np.zeros(len(TIME_POINTS))
            for t in range(len(TIME_POINTS)):
                t_stat, _ = ttest_1samp(d[:, t], 0, nan_policy='omit')
                bfs[t] = get_bf10_jzs(t_stat, N)
                
            mu_smoothed = smooth_trace(mu, n)
            bfs_smoothed = smooth_trace(bfs, n)
            
            for t_idx in range(len(TIME_POINTS) - 1):
                x_seg = TIME_POINTS[t_idx:t_idx+2]
                y_seg = mu_smoothed[t_idx:t_idx+2]
                # Scaled line widths down proportionally (6->4.2, 3->2.1)
                lw_val = 4.2 if (bfs[t_idx] > BF_threshold and mu[t_idx] > 0) else 2.1
                ax_top.plot(x_seg, y_seg, color=colors[m], lw=lw_val, label=m if t_idx == 0 else "")
            
            ax_bot.plot(TIME_POINTS, bfs_smoothed, color=colors[m], lw=2.1, label=f"{m} BF10" if idx == 0 else "")
            
        pairs = [('Linear', 'Sigmoid'), ('Linear', 'Step'), ('Sigmoid', 'Step')]
        
        y_offset = 0.02  
        for m1, m2 in pairs:
            p_color = get_blended_color(colors[m1], colors[m2])
            diff_data = model_data_dict[m1] - model_data_dict[m2]
            
            bfs_diff = np.zeros(len(TIME_POINTS))
            for t in range(len(TIME_POINTS)):
                t_stat, _ = ttest_rel(model_data_dict[m1][:, t], model_data_dict[m2][:, t], nan_policy='omit')
                bfs_diff[t] = get_bf10_jzs(t_stat, N)
            
            bfs_diff_smoothed = smooth_trace(bfs_diff, n)
            ax_bot.plot(TIME_POINTS, bfs_diff_smoothed, color=p_color, linestyle='-', lw=1.8, label=f'{m1}-{m2} BF10' if idx == 0 else '')
            
            sig_mask = bfs_diff > BF_threshold
            if np.any(sig_mask):
                y_pos = np.nanmin([np.nanmean(model_data_dict[m][0]) for m in colors]) - y_offset
                line_y = np.full_like(TIME_POINTS, y_pos, dtype=float)
                line_y[~sig_mask] = np.nan
                # Tightened significance bars (4.5 -> 3.0, markersize 5 -> 3.5)
                ax_top.plot(TIME_POINTS, line_y, color=p_color, lw=3.0, 
                            marker='s', markersize=3.5, solid_capstyle='projecting',
                            label=f"{m1} vs {m2} Sig" if idx == 0 else "")
                y_offset += 0.015  

        ax_top.set_title(f"{title} Unique Coding", fontweight='bold', fontsize=11)
        if idx == 0:
            ax_top.legend(frameon=False, loc='upper left', fontsize=9)
        ax_top.axhline(0, color='k', ls='--', lw=0.8)
        ax_top.set_ylabel('Partial Corr (rho)', fontweight='bold', fontsize=10)
        
        ax_bot.axhline(1, color='gray', ls='--', alpha=0.5, lw=0.8, label='H0/H1 Bound (BF10=1)' if idx == 0 else '')
        ax_bot.axhline(BF_threshold, color='k', ls=':', alpha=0.7, lw=0.8, label=f'Threshold (BF10={BF_threshold})' if idx == 0 else '')
        ax_bot.set_yscale('log')
        ax_bot.set_ylabel('Bayes Factor (BF10)', fontweight='bold', fontsize=10)
        ax_bot.set_xlabel('Time (ms)', fontweight='bold', fontsize=10)
        if idx == 0:
            ax_bot.legend(frameon=False, loc='upper left', fontsize=8)
        ax_top.tick_params(labelsize=9)
        ax_bot.tick_params(labelsize=9)
        
    fig1.tight_layout()
    fig1.savefig(os.path.join(fig_out_dir, 'Figure1_Coding_Dynamics.pdf'), format='pdf', dpi=600, bbox_inches='tight')

    # ==========================================
    # FIGURE 2: NON-TRANSFORMED FLOW PATHWAYS (Scaled down)
    # ==========================================
    fig2 = plt.figure(figsize=(10.3, 7.8)) # Adjusted width proportionally from 15.5 to ~10.3 for 2 columns instead of 3
    gs2 = fig2.add_gridspec(2, 2, height_ratios=[2, 1])
    
    raw_ff_lin = np.array(group_data['ff_linear'])
    raw_fb_lin = np.array(group_data['fb_linear'])
    raw_ff_sig = np.array(group_data['ff_sigmoid'])
    raw_fb_sig = np.array(group_data['fb_sigmoid'])

    non_trans_plots = [
        (raw_ff_lin, raw_fb_lin, 'Linear -> Linear Flow Channels', 'tab:blue'),
        (raw_ff_sig, raw_fb_sig, 'Sigmoid -> Sigmoid Flow Channels', 'tab:red')
    ]

    ax_top_axes2 = []
    ax_bot_axes2 = []
    for i, (ff_raw, fb_raw, title, color) in enumerate(non_trans_plots):
        if i == 0:
            ax_top = fig2.add_subplot(gs2[0, i])
            ax_bot = fig2.add_subplot(gs2[1, i], sharex=ax_top)
        else:
            ax_top = fig2.add_subplot(gs2[0, i], sharey=ax_top_axes2[0])
            ax_bot = fig2.add_subplot(gs2[1, i], sharex=ax_top, sharey=ax_bot_axes2[0])
        ax_top_axes2.append(ax_top)
        ax_bot_axes2.append(ax_bot)
        
        # Stimulus onset line (Reduced thickness)
        ax_top.axvline(0, color='gray', ls='--', lw=1.2, alpha=0.7)
        ax_bot.axvline(0, color='gray', ls='--', lw=1.2, alpha=0.7)
        
        sub_ff = np.nanmean(ff_raw, axis=1)
        sub_fb = np.nanmean(fb_raw, axis=1)
        
        mu_ff, se_ff = np.nanmean(sub_ff, 0), np.nanstd(sub_ff, 0)/np.sqrt(N)
        mu_fb, se_fb = np.nanmean(sub_fb, 0), np.nanstd(sub_fb, 0)/np.sqrt(N)
        
        # FIXED: Synchronized shading transparency to match pale version identically
        ax_top.fill_between(TIME_POINTS, mu_ff-se_ff, mu_ff+se_ff, color=color, alpha=0.15)
        ax_top.fill_between(TIME_POINTS, mu_fb-se_fb, mu_fb+se_fb, color=color, alpha=0.15)
        
        bfs_ff = np.zeros(len(TIME_POINTS))
        bfs_fb = np.zeros(len(TIME_POINTS))
        for t in range(len(TIME_POINTS)):
            t_stat_ff, _ = ttest_1samp(sub_ff[:, t], 0, nan_policy='omit')
            t_stat_fb, _ = ttest_1samp(sub_fb[:, t], 0, nan_policy='omit')
            bfs_ff[t] = get_bf10_jzs(t_stat_ff, N)
            bfs_fb[t] = get_bf10_jzs(t_stat_fb, N)
        
        mu_ff_smoothed = smooth_trace(mu_ff, n)
        mu_fb_smoothed = smooth_trace(mu_fb, n)
        bfs_ff_smoothed = smooth_trace(bfs_ff, n)
        bfs_fb_smoothed = smooth_trace(bfs_fb, n)
        
        # Render the foundational background traces cleanly across the full span
        # FIXED: Continuous background dashed trace for FB allows proper dash distribution
        ax_top.plot(TIME_POINTS, mu_ff_smoothed, color=color, lw=2.5, label='FF (Post->Ant)' if i == 0 else "")
        ax_top.plot(TIME_POINTS, mu_fb_smoothed, color=color, linestyle='--', lw=2.1, label='FB (Ant->Post)' if i == 0 else "")
        
        # Overlay thickened solid segments precisely on significant points
        for t_idx in range(len(TIME_POINTS) - 1):
            x_seg = TIME_POINTS[t_idx:t_idx+2]
            if bfs_ff[t_idx] > BF_threshold and mu_ff[t_idx] > 0:
                ax_top.plot(x_seg, mu_ff_smoothed[t_idx:t_idx+2], color=color, lw=4.2, linestyle='-')
            if bfs_fb[t_idx] > BF_threshold and mu_fb[t_idx] > 0:
                ax_top.plot(x_seg, mu_fb_smoothed[t_idx:t_idx+2], color=color, lw=4.2, linestyle='-')
        
        diff_ff_fb = sub_ff - sub_fb
        
        bfs_diff = np.zeros(len(TIME_POINTS))
        for t in range(len(TIME_POINTS)):
            t_stat_diff, _ = ttest_1samp(diff_ff_fb[:, t], 0, nan_policy='omit')
            bfs_diff[t] = get_bf10_jzs(t_stat_diff, N)
            
        bfs_diff_smoothed = smooth_trace(bfs_diff, n)
        
        p_color = get_blended_color(color, color)
        
        ax_bot.plot(TIME_POINTS, bfs_ff_smoothed, color=color, lw=2.1, label='FF BF10' if i == 0 else '')
        ax_bot.plot(TIME_POINTS, bfs_fb_smoothed, color=color, linestyle='--', lw=1.8, label='FB BF10' if i == 0 else '')
        ax_bot.plot(TIME_POINTS, bfs_diff_smoothed, color=p_color, linestyle='-', lw=1.8, label='FF-FB Diff BF10' if i == 0 else '')
        
        sig_mask = bfs_diff > BF_threshold
        if np.any(sig_mask):
            y_min, y_max = ax_top.get_ylim()
            if y_min == y_max:
                y_min, y_max = y_min - 0.1, y_max + 0.1
            y_pos = y_min + (y_max - y_min) * 0.08
            line_y = np.full_like(TIME_POINTS, y_pos, dtype=float)
            line_y[~sig_mask] = np.nan
            ax_top.plot(TIME_POINTS, line_y, color=p_color, lw=3.0, 
                        marker='s', markersize=3.5, solid_capstyle='projecting',
                        label='FF vs FB Diff Sig' if i == 0 else "")

        ax_top.set_title(title, fontweight='bold', fontsize=11)
        ax_top.axhline(0, color='k', ls='--', lw=0.8)
        ax_top.set_ylabel('Covariance Flow', fontweight='bold', fontsize=10)
        if i == 0:
            ax_top.legend(frameon=False, loc='upper left', fontsize=9)
        
        ax_bot.axhline(1, color='gray', ls='--', alpha=0.5, lw=0.8)
        ax_bot.axhline(BF_threshold, color='k', ls=':', alpha=0.7, lw=0.8)
        ax_bot.set_yscale('log')
        ax_bot.set_ylabel('Bayes Factor (BF10)', fontweight='bold', fontsize=10)
        ax_bot.set_xlabel('Time (ms)', fontweight='bold', fontsize=10)
        if i == 0:
            ax_bot.legend(frameon=False, loc='upper left', fontsize=8)
        ax_top.tick_params(labelsize=9)
        ax_bot.tick_params(labelsize=9)

    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_out_dir, 'Figure2_NonTransformed_Flow.pdf'), format='pdf', dpi=600, bbox_inches='tight')

    # ==========================================
    # FIGURE 3: TRANSFORMED FLOW CHANNELS (Scaled down)
    # ==========================================
    fig3 = plt.figure(figsize=(6.4, 7.1)) # Multiplied 9x10 by 0.707
    gs3 = fig3.add_gridspec(2, 1, height_ratios=[2, 1])
    
    raw_ff = np.array(group_data['ff'])  
    raw_fb = np.array(group_data['fb'])
    
    sub_ff_avg = np.nanmean(raw_ff, axis=1) 
    sub_fb_avg = np.nanmean(raw_fb, axis=1) 

    ff_avg, fb_avg = np.nanmean(sub_ff_avg, axis=0), np.nanmean(sub_fb_avg, axis=0)
    
    ax_top = fig3.add_subplot(gs3[0, 0])
    ax_bot = fig3.add_subplot(gs3[1, 0], sharex=ax_top)
    
    ax_top.axvline(0, color='gray', ls='--', lw=1.2, alpha=0.7)
    ax_bot.axvline(0, color='gray', ls='--', lw=1.2, alpha=0.7)
    
    color_ff_fb = 'purple'
    p_color = 'black' 
    
    # FIXED: Synchronized shading transparency identically here as well
    ax_top.fill_between(TIME_POINTS, ff_avg-(np.nanstd(sub_ff_avg,0)/np.sqrt(N)), ff_avg+(np.nanstd(sub_ff_avg,0)/np.sqrt(N)), color=color_ff_fb, alpha=0.15)
    ax_top.fill_between(TIME_POINTS, fb_avg-(np.nanstd(sub_fb_avg,0)/np.sqrt(N)), fb_avg+(np.nanstd(sub_fb_avg,0)/np.sqrt(N)), color=color_ff_fb, alpha=0.15)
    
    bfs_ff = np.zeros(len(TIME_POINTS))
    bfs_fb = np.zeros(len(TIME_POINTS))
    for t in range(len(TIME_POINTS)):
        t_stat_ff, _ = ttest_1samp(sub_ff_avg[:, t], 0, nan_policy='omit')
        t_stat_fb, _ = ttest_1samp(sub_fb_avg[:, t], 0, nan_policy='omit')
        bfs_ff[t] = get_bf10_jzs(t_stat_ff, N)
        bfs_fb[t] = get_bf10_jzs(t_stat_fb, N)
        
    ff_mu_smoothed = smooth_trace(ff_avg, n)
    fb_mu_smoothed = smooth_trace(fb_avg, n)
    bfs_ff_smoothed = smooth_trace(bfs_ff, n)
    bfs_fb_smoothed = smooth_trace(bfs_fb, n)
    
    # Render foundational full-length lines
    # FIXED: Background trace provides long continuous timeline execution for clean dashes
    ax_top.plot(TIME_POINTS, ff_mu_smoothed, color=color_ff_fb, lw=2.5, label='FF (Post->Ant)' if t_idx == 0 else "")
    ax_top.plot(TIME_POINTS, fb_mu_smoothed, color=color_ff_fb, linestyle='--', lw=2.1, label='FB (Ant->Post)' if t_idx == 0 else "")
    
    # Overlay thickened significant parts
    for t_idx in range(len(TIME_POINTS) - 1):
        x_seg = TIME_POINTS[t_idx:t_idx+2]
        if bfs_ff[t_idx] > BF_threshold and ff_avg[t_idx] > 0:
            ax_top.plot(x_seg, ff_mu_smoothed[t_idx:t_idx+2], color=color_ff_fb, lw=4.2, linestyle='-')
        if bfs_fb[t_idx] > BF_threshold and fb_avg[t_idx] > 0:
            ax_top.plot(x_seg, fb_mu_smoothed[t_idx:t_idx+2], color=color_ff_fb, lw=4.2, linestyle='-')
    
    diff_transformed = sub_ff_avg - sub_fb_avg
    
    bfs_diff = np.zeros(len(TIME_POINTS))
    for t in range(len(TIME_POINTS)):
        t_stat_diff, _ = ttest_1samp(diff_transformed[:, t], 0, nan_policy='omit')
        bfs_diff[t] = get_bf10_jzs(t_stat_diff, N)
        
    bfs_diff_smoothed = smooth_trace(bfs_diff, n)
    
    ax_bot.plot(TIME_POINTS, bfs_ff_smoothed, color=color_ff_fb, lw=2.1, label='FF BF10')
    ax_bot.plot(TIME_POINTS, bfs_fb_smoothed, color=color_ff_fb, linestyle='--', lw=1.8, label='FB BF10')
    ax_bot.plot(TIME_POINTS, bfs_diff_smoothed, color=p_color, linestyle='-', lw=1.8, label='FF-FB Diff BF10')
    
    sig_mask = bfs_diff > BF_threshold
    if np.any(sig_mask):
        y_min, y_max = ax_top.get_ylim()
        if y_min == y_max:
            y_min, y_max = y_min - 0.1, y_max + 0.1
        y_pos = y_min + (y_max - y_min) * 0.08
        line_y = np.full_like(TIME_POINTS, y_pos, dtype=float)
        line_y[~sig_mask] = np.nan
        ax_top.plot(TIME_POINTS, line_y, color=p_color, lw=3.0, 
                    marker='s', markersize=3.5, solid_capstyle='projecting',
                    label='FF vs FB Diff Sig')
    
    ax_top.set_title("Transformed Flow Overlaid Channels", fontweight='bold', fontsize=11)
    ax_top.axhline(0, color='k', ls='--', lw=0.8)
    ax_top.set_ylabel('Covariance Flow', fontweight='bold', fontsize=10)
    ax_top.legend(frameon=False, loc='upper left', fontsize=9)
    
    ax_bot.axhline(1, color='gray', ls='--', alpha=0.5, lw=0.8)
    ax_bot.axhline(BF_threshold, color='k', ls=':', alpha=0.7, lw=0.8)
    ax_bot.set_yscale('log')
    ax_bot.set_ylabel('Bayes Factor (BF10)', fontweight='bold', fontsize=10)
    ax_bot.set_xlabel('Time (ms)', fontweight='bold', fontsize=10)
    ax_bot.legend(frameon=False, loc='upper left', fontsize=8)
    ax_top.tick_params(labelsize=9)
    ax_bot.tick_params(labelsize=9)

    fig3.tight_layout()
    fig3.savefig(os.path.join(fig_out_dir, 'Figure3_Transformed_Flow_Overlay.pdf'), format='pdf', dpi=600, bbox_inches='tight')

    # ==========================================
    # FIGURE 4: BEST MODEL PARAMETERS (Scaled down)
    # ==========================================
    fig4, (ax_x0, ax_k) = plt.subplots(1, 2, figsize=(11.3, 4.2)) # Multiplied 16x6 by 0.707
    ax_x0.axvline(0, color='gray', ls='--', lw=1.2, alpha=0.7)
    ax_k.axvline(0, color='gray', ls='--', lw=1.2, alpha=0.7)
    
    param_keys_x0 = [('linear_x0', 'Linear $x_0$', 'tab:blue'), 
                     ('step_x0', 'Step $x_0$', 'tab:green'), 
                     ('sigmoid_x0', 'Sigmoid $x_0$', 'tab:red')]
    
    for key, label, col in param_keys_x0:
        p_data = np.array(group_data['best_params'][key])
        mu_p = smooth_trace(np.nanmean(p_data, axis=0), n)
        se_p = np.nanstd(p_data, axis=0) / np.sqrt(N)
        ax_x0.fill_between(TIME_POINTS, mu_p - se_p, mu_p + se_p, color=col, alpha=0.15)
        ax_x0.plot(TIME_POINTS, mu_p, color=col, lw=2.1, label=label)
        
    ax_x0.set_title('Fitted Inflection Boundaries ($x_0$)', fontweight='bold', fontsize=11)
    ax_x0.set_xlabel('Time (ms)', fontweight='bold', fontsize=10)
    ax_x0.set_ylabel('Morph Category Units', fontweight='bold', fontsize=10)
    ax_x0.legend(frameon=False, loc='upper left', fontsize=9)
    ax_x0.grid(True, ls='--', alpha=0.3, lw=0.6)

    k_data = np.array(group_data['best_params']['sigmoid_k'])
    mu_k = smooth_trace(np.nanmean(k_data, axis=0), n)
    se_k = np.nanstd(k_data, axis=0) / np.sqrt(N)
    ax_k.fill_between(TIME_POINTS, mu_k - se_k, mu_k + se_k, color='tab:orange', alpha=0.15)
    ax_k.plot(TIME_POINTS, mu_k, color='tab:orange', lw=2.1, label='Sigmoid $k$')
    ax_k.set_yscale('log')
    ax_k.set_title('Fitted Sigmoid Slope Parameter ($k$)', fontweight='bold', fontsize=11)
    ax_k.set_xlabel('Time (ms)', fontweight='bold', fontsize=10)
    ax_k.set_ylabel('Slope Gain (Log Scale)', fontweight='bold', fontsize=10)
    ax_k.legend(frameon=False, loc='upper left', fontsize=9)
    ax_k.grid(True, which="both", ls='--', alpha=0.3, lw=0.6)
    ax_x0.tick_params(labelsize=9)
    ax_k.tick_params(labelsize=9)
    
    fig4.tight_layout()
    fig4.savefig(os.path.join(fig_out_dir, 'Figure4_Model_Parameters.pdf'), format='pdf', dpi=600, bbox_inches='tight')

    # ==========================================
    # FIGURE 5: SIGMOIDAL EVOLUTION AT CRITICAL TIMES (Scaled down)
    # ==========================================
    target_times = [-100, 0, 100, 200, 300, 500]
    sig_x0_arr = np.array(group_data['best_params']['sigmoid_x0'])
    sig_k_arr = np.array(group_data['best_params']['sigmoid_k'])
    mean_x0 = np.nanmean(sig_x0_arr, axis=0)
    mean_k = np.nanmean(sig_k_arr, axis=0)
    
    fig5, ax5 = plt.subplots(figsize=(7.1, 4.2)) # Multiplied 10x6 by 0.707
    morph_axis = np.arange(N_MORPH)
    time_colors = plt.cm.viridis(np.linspace(0, 1, len(target_times)))
    
    for t_val, color in zip(target_times, time_colors):
        if t_val in TIME_POINTS:
            t_idx = np.where(TIME_POINTS == t_val)[0][0]
            x0_t = mean_x0[t_idx]
            k_t = mean_k[t_idx]
            sigmoid_vals = 1 / (1 + np.exp(-k_t * (morph_axis - x0_t)))
            ax5.plot(morph_axis + 1, sigmoid_vals, label=f"{t_val} ms", color=color, lw=2.5)
            
    ax5.set_title("Evolution of Neural Sigmoid Functions", fontsize=11, fontweight='bold')
    ax5.set_xlabel("Morph Levels (1 to 21)", fontsize=10, fontweight='bold')
    ax5.set_ylabel("Normalized Response Model Value", fontsize=10, fontweight='bold')
    ax5.set_xticks(np.arange(1, N_MORPH + 1, 2))
    ax5.grid(True, linestyle=':', alpha=0.6, lw=0.6)
    ax5.legend(frameon=True, facecolor='white', edgecolor='none', loc='lower right', fontsize=8)
    ax5.tick_params(labelsize=9)
    
    fig5.tight_layout()
    fig5.savefig(os.path.join(fig_out_dir, 'Figure5_Sigmoid_Evolution.pdf'), format='pdf', dpi=600, bbox_inches='tight')
    
    plt.show()