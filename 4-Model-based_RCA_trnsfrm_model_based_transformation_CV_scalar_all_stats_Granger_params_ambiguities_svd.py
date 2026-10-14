import numpy as np
import os
import re
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, ttest_1samp, ttest_rel
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.integrate import quad

# --- 1. CONFIGURATION ---
# INPUT_DIR = '/QRISdata/Q7503/Pareidolea/EEG/Main data/Processed_Outputs'
INPUT_DIR = r'C:\Users\uqhkarim\OneDrive - The University of Queensland\Documents\Scientific\Projects\Pareidolia\EEG\Main data\Processed_Outputs'

SAVE_PATH_FLOW = os.path.join(INPUT_DIR, 'group_RCA_TRANS_FOLDER_LEVEL_Scalar_all_stats_HR_Granger_params_morph_spans.npy')
TIME_POINTS = np.arange(-100, 601, 5)
DELAYS_MS = np.arange(5, 201, 20) 
N_MORPH = 21
FIT_WINDOW_HALF = 5 

ANT_PREFIXES = ('Fp', 'AF', 'F', 'FC')
POST_PREFIXES = ('CP', 'P', 'PO', 'O')

X0_RANGE = np.arange(5, 16) 

# Morph levels definitions for the 3 categories
MORPH_SPANS = {
    'Morph 1-7': np.arange(1, 8),
    'Morph 8-14': np.arange(8, 15),
    'Morph 15-21': np.arange(15, 22)
}
SPAN_KEYS = list(MORPH_SPANS.keys())

# --- 2. RDM HELPERS ---
def get_triu(mat, n_items): return mat[np.triu_indices(n_items, k=1)]

def make_linear_rdm(x0, morph_list):
    # Map the selected subset of morph levels to calculate the RDM matrix correctly for smaller sizes
    vals = (np.array(morph_list) / x0).reshape(-1, 1)
    return get_triu(squareform(pdist(vals, metric='cityblock')), len(morph_list))

# --- 3. CORE ANALYTICAL HELPERS ---
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
    processed_count = len(group_data['ff_linear'][SPAN_KEYS[0]])
else:
    group_data = {
        'ff_linear': {span: [] for span in SPAN_KEYS},
        'fb_linear': {span: [] for span in SPAN_KEYS},
        'coding_whole': {span: [] for span in SPAN_KEYS},
        'coding_ant': {span: [] for span in SPAN_KEYS},
        'coding_post': {span: [] for span in SPAN_KEYS},
        'best_params': {span: [] for span in SPAN_KEYS}
    }
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

    # Initialize sub-structures to hold statistics dynamically for each separate morph span
    sub_res = {
        span: {
            'ff_linear': np.zeros((len(DELAYS_MS), len(TIME_POINTS))), 
            'fb_linear': np.zeros((len(DELAYS_MS), len(TIME_POINTS))),
            'c_ant': np.zeros(len(TIME_POINTS)),
            'c_post': np.zeros(len(TIME_POINTS)),
            'c_whole': np.zeros(len(TIME_POINTS)),
            'linear_x0': np.zeros(len(TIME_POINTS))
        } for span in SPAN_KEYS
    }

    # Pre-compute Neural RDMs independently for each morph span subset
    neural_rdms = {span: {f: {reg: {} for reg in ['A', 'P', 'W']} for f in [0, 1]} for span in SPAN_KEYS}
    
    for span in SPAN_KEYS:
        target_morphs = MORPH_SPANS[span]
        for f_idx, trials_idx in enumerate([t_half1, t_half2]):
            for t in TIME_POINTS:
                t_mask = (time_ms >= t - FIT_WINDOW_HALF) & (time_ms <= t + FIT_WINDOW_HALF)
                
                def get_reg_rdm(c_idx):
                    patterns = []
                    for m in target_morphs:
                        m_idx = test_idx[morph_levels[test_idx] == m]
                        p = np.nanmean(data_4d[c_idx, :, :, :][:, t_mask, :, :][:, :, m_idx, :][:, :, :, trials_idx], axis=(1, 2, 3))
                        patterns.append(p)
                    return pdist(np.array(patterns), metric='euclidean')
                
                neural_rdms[span][f_idx]['A'][t] = get_reg_rdm(a_ch)
                neural_rdms[span][f_idx]['P'][t] = get_reg_rdm(p_ch)
                neural_rdms[span][f_idx]['W'][t] = get_reg_rdm(w_ch)

    # 4. Grid Search and Representation Block per Morph Category
    scalar_waveforms = {span: {f: {reg: np.zeros(len(TIME_POINTS)) for reg in ['A', 'P', 'W']} for f in [0, 1]} for span in SPAN_KEYS}
    
    for span in SPAN_KEYS:
        target_morphs = MORPH_SPANS[span]
        for i_t, t in enumerate(TIME_POINTS):
            for train_f, test_f in [(0, 1), (1, 0)]:
                best_model = None
                max_r = -1.0
                fit_neural = neural_rdms[span][train_f]['W'][t]
                best_x0 = None

                for x0 in X0_RANGE:
                    l_rdm = make_linear_rdm(x0, target_morphs)
                    r_l = spearmanr(l_rdm, fit_neural)[0]
                    if r_l > max_r: 
                        max_r, best_model = r_l, l_rdm
                        best_x0 = x0

                sub_res[span]['linear_x0'][i_t] += best_x0 / 2

                # Isolated simple correlation mapping tracking 
                for reg, res_k in [('A', 'c_ant'), ('P', 'c_post'), ('W', 'c_whole')]:
                    eval_neural = neural_rdms[span][test_f][reg][t]
                    r_val, _ = spearmanr(best_model, eval_neural)
                    s_lin = r_val if not np.isnan(r_val) else 0
                    
                    sub_res[span][res_k][i_t] += s_lin / 2
                    scalar_waveforms[span][test_f][reg][i_t] = s_lin

    # 5. Scalar Directed Flow Tracking Analysis 
    for span in SPAN_KEYS:
        target_morphs = MORPH_SPANS[span]
        for i_t, t in enumerate(TIME_POINTS):
            for train_f, test_f in [(0, 1), (1, 0)]:
                best_model = None
                max_r = -1.0
                fit_neural = neural_rdms[span][train_f]['W'][t]
                
                for x0 in X0_RANGE:
                    l_rdm = make_linear_rdm(x0, target_morphs)
                    r_l = spearmanr(l_rdm, fit_neural)[0]
                    if r_l > max_r: max_r, best_model = r_l, l_rdm

                for d_idx, dly in enumerate(DELAYS_MS):
                    t_p = t - dly
                    if t_p in TIME_POINTS:
                        i_t_past = np.where(TIME_POINTS == t_p)[0][0]
                        
                        past_dest_rdm_ant = neural_rdms[span][test_f]['A'][t_p].flatten().reshape(-1, 1)
                        past_dest_rdm_post = neural_rdms[span][test_f]['P'][t_p].flatten().reshape(-1, 1)
                        
                        eval_neural_ant = neural_rdms[span][test_f]['A'][t].flatten()
                        eval_neural_post = neural_rdms[span][test_f]['P'][t].flatten()
                        
                        # History stripping residuals
                        res_neural_ant = eval_neural_ant - LinearRegression().fit(past_dest_rdm_ant, eval_neural_ant).predict(past_dest_rdm_ant)
                        res_neural_post = eval_neural_post - LinearRegression().fit(past_dest_rdm_post, eval_neural_post).predict(past_dest_rdm_post)
                        
                        v_flat = best_model.flatten()
                        res_model_ant = v_flat - LinearRegression().fit(past_dest_rdm_ant, v_flat).predict(past_dest_rdm_ant)
                        res_model_post = v_flat - LinearRegression().fit(past_dest_rdm_post, v_flat).predict(past_dest_rdm_post)
                        
                        rho_ant, _ = spearmanr(res_model_ant, res_neural_ant)
                        now_ant_linear = rho_ant if not np.isnan(rho_ant) else 0
                        
                        rho_post, _ = spearmanr(res_model_post, res_neural_post)
                        now_post_linear = rho_post if not np.isnan(rho_post) else 0
                        
                        # Un-transformed directional updates tracking
                        sub_res[span]['ff_linear'][d_idx, i_t] += (scalar_waveforms[span][test_f]['P'][i_t_past] * now_ant_linear) / 2
                        sub_res[span]['fb_linear'][d_idx, i_t] += (scalar_waveforms[span][test_f]['A'][i_t_past] * now_post_linear) / 2

    # Append subject results per categories and Save
    for span in SPAN_KEYS:
        group_data['coding_whole'][span].append(sub_res[span]['c_whole'])
        group_data['coding_ant'][span].append(sub_res[span]['c_ant'])
        group_data['coding_post'][span].append(sub_res[span]['c_post'])
        group_data['ff_linear'][span].append(sub_res[span]['ff_linear'])
        group_data['fb_linear'][span].append(sub_res[span]['fb_linear'])
        group_data['best_params'][span].append(sub_res[span]['linear_x0'])
        
    tmp_save_path = SAVE_PATH_FLOW.replace('.npy', '_tmp.npy') 
    np.save(tmp_save_path, group_data)
    os.replace(tmp_save_path, SAVE_PATH_FLOW)
    print(f"Subject {sub_id} complete and saved.")
    
# --- 5. VISUALIZATION ---
if os.path.exists(SAVE_PATH_FLOW):
    BF_threshold = 3
    n = 5  # Smoothing window scale
    r_scale = 0.5  # Narrower Cauchy prior scale (default JZS is sqrt(2)/2 approx 0.707)
                   # Lower values make the test more lenient/sensitive to small-to-moderate effects.
    fig_out_dir = os.path.dirname(SAVE_PATH_FLOW)
    
    # Identify indices corresponding to the pre-stimulus baseline window
    baseline_indices = np.where((TIME_POINTS >= -100) & (TIME_POINTS <= 0))[0]

    def get_bf10_jzs(t_stats, N, r=r_scale):
        """Calculates the true JZS Bayes Factor (BF10) from a given t-statistic with customizable prior scale r."""
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

    def calculate_time_resolved_bfs(data, baseline_idx):
        """Computes Bayes Factors across all time points using the selected null method strategy."""
        num_subs, num_times = data.shape
        bfs_series = np.zeros(num_times)
        
        sub_baselines = np.nanmean(data[:, baseline_idx], axis=1, keepdims=True)
        corrected_data = data - sub_baselines
        for t in range(num_times):
            t_stat, _ = ttest_1samp(corrected_data[:, t], 0, nan_policy='omit')
            bfs_series[t] = get_bf10_jzs(t_stat, num_subs)    
        return bfs_series

    def calculate_paired_diff_bfs(data1, data2):
        """Computes paired-sample JZS Bayes Factors between two conditions over time."""
        num_subs, num_times = data1.shape
        bfs_series = np.zeros(num_times)
        for t in range(num_times):
            t_stat, _ = ttest_rel(data1[:, t], data2[:, t], nan_policy='omit')
            bfs_series[t] = get_bf10_jzs(t_stat, num_subs)
        return bfs_series

    def smooth_trace(trace, window_size):
        if window_size <= 1: return trace
        padded = np.pad(trace, pad_width=window_size//2, mode='edge')
        smoothed = np.convolve(padded, np.ones(window_size)/window_size, mode='valid')
        return smoothed[:len(trace)] if len(smoothed) > len(trace) else np.pad(smoothed, (0, len(trace)-len(smoothed)), mode='edge')

    group_data = np.load(SAVE_PATH_FLOW, allow_pickle=True).item()
    N = len(group_data['ff_linear'][SPAN_KEYS[0]])
    
    span_colors = {'Morph 1-7': '#4292c6', 'Morph 8-14': '#08519c', 'Morph 15-21': '#000000'}
    pairs = [('Morph 1-7', 'Morph 8-14'), 
             ('Morph 1-7', 'Morph 15-21'), 
             ('Morph 8-14', 'Morph 15-21')]
    
    # Updated explicit distinct color mapping for pairwise comparison traces and indicators
    pair_colors = {
        ('Morph 1-7', 'Morph 8-14'): '#008080',   # Teal/Cyan
        ('Morph 1-7', 'Morph 15-21'): '#d95f02',  # Amber/Dark Gold
        ('Morph 8-14', 'Morph 15-21'): '#7570b3'  # Dark Purple
    }
    
    # ==========================================
    # FIGURE 1: INFORMATION CODING DYNAMICS
    # ==========================================
    fig1 = plt.figure(figsize=(11.3, 7.8))
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
        
        # Stimulus onset line
        ax_top.axvline(0, color='gray', ls='--', lw=1.2, alpha=0.7)
        ax_bot.axvline(0, color='gray', ls='--', lw=1.2, alpha=0.7)
        
        span_data_dict = {}
        
        for span in SPAN_KEYS:
            d = np.array(group_data[rk][span])
            
            d_plot = d - np.nanmean(d[:, baseline_indices], axis=1, keepdims=True)
                
            span_data_dict[span] = d_plot
            mu, se = np.nanmean(d_plot, 0), np.nanstd(d_plot, 0)/np.sqrt(N)
            
            alpha_val = 0.30 if span == 'Morph 1-7' else 0.15
            ax_top.fill_between(TIME_POINTS, mu-se, mu+se, color=span_colors[span], alpha=alpha_val)
            
            bfs = calculate_time_resolved_bfs(d, baseline_indices)
            mu_smoothed = smooth_trace(mu, n)
            
            for t_idx in range(len(TIME_POINTS) - 1):
                x_seg = TIME_POINTS[t_idx:t_idx+2]
                y_seg = mu_smoothed[t_idx:t_idx+2]
                lw_val = 5.0 if (bfs[t_idx] > BF_threshold and mu[t_idx] > 0) else 2.1
                ax_top.plot(x_seg, y_seg, color=span_colors[span], lw=lw_val, label=span if t_idx == 0 else "")
            
        # Pairwise cross-condition comparisons
        fig1_y_positions = {
            ('Morph 1-7', 'Morph 8-14'): -0.08,
            ('Morph 1-7', 'Morph 15-21'): -0.1,
            ('Morph 8-14', 'Morph 15-21'): -0.12  
        }
        
        for s1, s2 in pairs:
            p_color = pair_colors[(s1, s2)]
            bfs_diff = calculate_paired_diff_bfs(span_data_dict[s1], span_data_dict[s2])
            bfs_diff_smoothed = smooth_trace(bfs_diff, n)
            
            ax_bot.plot(TIME_POINTS, bfs_diff_smoothed, color=p_color, linestyle='-', lw=1.8, label=f'{s1} vs {s2} BF10' if idx == 0 else '')
            
            sig_mask = bfs_diff > BF_threshold
            if np.any(sig_mask):
                y_pos = fig1_y_positions.get((s1, s2), -0.1)
                line_y = np.full_like(TIME_POINTS, y_pos, dtype=float)
                line_y[~sig_mask] = np.nan
                ax_top.plot(TIME_POINTS, line_y, color=p_color, linestyle='-', lw=3.0, solid_capstyle='butt', label=f"{s1} vs {s2} Diff Sig" if idx == 0 else "")
        
        ax_top.set_title(f"{title} Linear Coding Dynamics", fontweight='bold', fontsize=11)
        if idx == 0: ax_top.legend(frameon=False, loc='upper left', fontsize=9)
        ax_top.axhline(0, color='k', ls='--', lw=0.8)
        ax_top.set_ylabel('Correlation Amplitude (rho)', fontweight='bold', fontsize=10)
        
        current_ylim_top = ax_top.get_ylim()
        ax_top.set_ylim(bottom=-0.15, top=current_ylim_top[1])
        
        ax_bot.axhline(1, color='gray', ls='--', alpha=0.5, lw=0.8)
        ax_bot.axhline(BF_threshold, color='k', ls=':', alpha=0.7, lw=0.8)
        ax_bot.set_yscale('log')
        ax_bot.set_ylabel('Bayes Factor (BF10)', fontweight='bold', fontsize=10)
        ax_bot.set_xlabel('Time (ms)', fontweight='bold', fontsize=10)
        if idx == 0: ax_bot.legend(frameon=False, loc='upper left', fontsize=8)
        ax_top.tick_params(labelsize=9)
        ax_bot.tick_params(labelsize=9)
        
    fig1.tight_layout()
    fig1.savefig(os.path.join(fig_out_dir, 'Figure1_Linear_Coding_Dynamics.pdf'), format='pdf', dpi=600, bbox_inches='tight')

    # ==========================================
    # FIGURE 2: LINEAR CHANNELS COVARIANCE FLOWS
    # ==========================================
    fig2 = plt.figure(figsize=(11.3, 7.8))
    gs2 = fig2.add_gridspec(2, 2, height_ratios=[2, 1])
    
    flow_directions = [('ff_linear', 'Feedforward (Post -> Ant)'), ('fb_linear', 'Feedback (Ant -> Post)')]
    ax_top_axes2 = []
    ax_bot_axes2 = []
    
    for idx, (fk, dir_title) in enumerate(flow_directions):
        if idx == 0:
            ax_top = fig2.add_subplot(gs2[0, idx])
            ax_bot = fig2.add_subplot(gs2[1, idx], sharex=ax_top)
        else:
            ax_top = fig2.add_subplot(gs2[0, idx], sharey=ax_top_axes2[0])
            ax_bot = fig2.add_subplot(gs2[1, idx], sharex=ax_top, sharey=ax_bot_axes2[0])
        ax_top_axes2.append(ax_top)
        ax_bot_axes2.append(ax_bot)
        
        # Stimulus onset line
        ax_top.axvline(0, color='gray', ls='--', lw=1.2, alpha=0.7)
        ax_bot.axvline(0, color='gray', ls='--', lw=1.2, alpha=0.7)
        
        flow_data_dict = {}
        
        for span in SPAN_KEYS:
            raw_flow = np.array(group_data[fk][span])
            sub_avg = np.nanmean(raw_flow, axis=1) 
            
            sub_avg_plot = sub_avg - np.nanmean(sub_avg[:, baseline_indices], axis=1, keepdims=True)
                
            flow_data_dict[span] = sub_avg_plot
            mu, se = np.nanmean(sub_avg_plot, 0), np.nanstd(sub_avg_plot, 0)/np.sqrt(N)
            
            alpha_val = 0.30 if span == 'Morph 1-7' else 0.15
            ax_top.fill_between(TIME_POINTS, mu-se, mu+se, color=span_colors[span], alpha=alpha_val)
            
            bfs = calculate_time_resolved_bfs(sub_avg, baseline_indices)
            mu_smoothed = smooth_trace(mu, n)
            
            for t_idx in range(len(TIME_POINTS) - 1):
                x_seg = TIME_POINTS[t_idx:t_idx+2]
                y_seg = mu_smoothed[t_idx:t_idx+2]
                lw_val = 6.0 if (bfs[t_idx] > BF_threshold and mu[t_idx] > 0) else 2.1
                ax_top.plot(x_seg, y_seg, color=span_colors[span], lw=lw_val, label=span if t_idx == 0 else "")
                
        # Pairwise cross-condition comparisons
        fig2_y_positions = {
            ('Morph 1-7', 'Morph 8-14'): -0.02,
            ('Morph 1-7', 'Morph 15-21'): -0.025,
            ('Morph 8-14', 'Morph 15-21'): -0.030  
        }
        
        for s1, s2 in pairs:
            p_color = pair_colors[(s1, s2)]
            bfs_diff = calculate_paired_diff_bfs(flow_data_dict[s1], flow_data_dict[s2])
            bfs_diff_smoothed = smooth_trace(bfs_diff, n)
            
            ax_bot.plot(TIME_POINTS, bfs_diff_smoothed, color=p_color, linestyle='-', lw=1.8, label=f'{s1} vs {s2} BF10' if idx == 0 else '')
            
            sig_mask = bfs_diff > BF_threshold
            if np.any(sig_mask):
                y_pos = fig2_y_positions.get((s1, s2), -0.02)
                line_y = np.full_like(TIME_POINTS, y_pos, dtype=float)
                line_y[~sig_mask] = np.nan
                ax_top.plot(TIME_POINTS, line_y, color=p_color, linestyle='-', lw=3.0, 
                            marker='s', markersize=3.5, solid_capstyle='projecting',
                            label=f"{s1} vs {s2} Diff Sig" if idx == 0 else "")

        ax_top.set_title(f"{dir_title} Linear Information Channels", fontweight='bold', fontsize=11)
        if idx == 0: ax_top.legend(frameon=False, loc='upper left', fontsize=9)
        ax_top.axhline(0, color='k', ls='--', lw=0.8)
        ax_top.set_ylabel('Covariance Flow Rate', fontweight='bold', fontsize=10)
        
        current_ylim_top2 = ax_top.get_ylim()
        ax_top.set_ylim(bottom=-0.035, top=current_ylim_top2[1])
        
        ax_bot.axhline(1, color='gray', ls='--', alpha=0.5, lw=0.8)
        ax_bot.axhline(BF_threshold, color='k', ls=':', alpha=0.7, lw=0.8)
        ax_bot.set_yscale('log')
        ax_bot.set_ylabel('Bayes Factor (BF10)', fontweight='bold', fontsize=10)
        ax_bot.set_xlabel('Time (ms)', fontweight='bold', fontsize=10)
        if idx == 0: ax_bot.legend(frameon=False, loc='upper left', fontsize=8)
        ax_top.tick_params(labelsize=9)
        ax_bot.tick_params(labelsize=9)

    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_out_dir, 'Figure2_Linear_Channels_Flow.pdf'), format='pdf', dpi=600, bbox_inches='tight')
    
    plt.show()