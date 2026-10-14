import mne
import pandas as pd
import numpy as np
import os
import traceback 

# --- 1. CONFIGURATION ---
BASE_DIR = r'C:\Users\uqhkarim\OneDrive - The University of Queensland\Documents\Scientific\Projects\Pareidolia\EEG\Main data'
OUTPUT_DIR = os.path.join(BASE_DIR, 'Processed_Outputs')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
# --- 2. STRICT DECODING LOGIC (Unchanged) ---
def decode_two_byte_triggers(events, sfreq):
    decoded_events = []
    max_pair_samples = int(sfreq * 0.050) 
    i = 0
    while i < len(events) - 1:
        code1 = events[i, 2]
        code2 = events[i+1, 2]
        
        # Standard trials: 1-30 and 101-200
        # Catch trials: 255 and 11-14
        is_standard = (1 <= code1 <= 30) and (101 <= code2 <= 200)
        is_catch = (code1 == 255) and (11 <= code2 <= 14)

        if is_standard or is_catch:
            sample_diff = events[i+1, 0] - events[i, 0]
            if sample_diff <= max_pair_samples:
                combined_code = code1 * 1000 + code2
                onset_sample = events[i, 0]
                decoded_events.append((onset_sample, 0, combined_code, code1, code2))
                i += 2 
                continue
        i += 1
        
    dtype = [('sample', 'int'), ('code', 'int'), ('combined_code', 'int'), ('byte1', 'int'), ('byte2', 'int')]
    return np.array(decoded_events, dtype=dtype)

# --- 3. PARTICIPANT LOOP ---
for p_num in range(1, 2): 
    p_id = f"P{p_num:02d}"
    p_folder = os.path.join(BASE_DIR, p_id)
    
    if not os.path.exists(p_folder):
        continue

    print(f"\n{'='*50}")
    print(f"PROCESSING SUBJECT: {p_id}")

    files_in_folder = os.listdir(p_folder)
    bdf_list = [f for f in files_in_folder if f.lower().endswith('.bdf')]
    xlsx_list = [f for f in files_in_folder if f.lower().endswith('.xlsx')]

    if not bdf_list or not xlsx_list:
        print(f"Warning: Missing files for {p_id}. Skipping.")
        continue

    EEG_FILE = os.path.join(p_folder, bdf_list[0])
    BEHAVIORAL_LOG = os.path.join(p_folder, xlsx_list[0])
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{p_id}.npy")
    
    try:
        raw = mne.io.read_raw_bdf(EEG_FILE, preload=True, verbose=False)
        
        montage = mne.channels.make_standard_montage('biosemi64')
        raw.set_montage(montage, on_missing='ignore')
        raw.set_eeg_reference('average', projection=False, verbose=False)
                
        print(f"Applying Filters: Band-pass (0.05-200 Hz), Notch (50 Hz)")
        raw.filter(l_freq=0.05, h_freq=200.0, fir_design='firwin', verbose=False)
        raw.notch_filter(freqs=50.0, fir_design='firwin', verbose=False)
               
        eeg_ch_names = np.array(raw.ch_names[:64], dtype=str)
        ch_positions = np.array([raw.info['chs'][ch_idx]['loc'][:3] for ch_idx in range(64)])
        sfreq = raw.info['sfreq']
    
        # 1. Extract EEG Triggers
        events = mne.find_events(raw, stim_channel='Status', shortest_event=1, verbose=False)
        decoded_all_events = decode_two_byte_triggers(events, sfreq)
        all_events_df = pd.DataFrame(decoded_all_events)
        
        # Separate Standard and Catch EEG triggers
        eeg_df = all_events_df[all_events_df['byte1'] <= 30].copy().reset_index(drop=True)
        eeg_catch_df = all_events_df[all_events_df['byte1'] == 255].copy().reset_index(drop=True)
        
        # 2. Extract Behavioral Log
        beh_df = pd.read_excel(BEHAVIORAL_LOG, sheet_name='RSVP')
        Catch_trials_info = pd.read_excel(BEHAVIORAL_LOG, sheet_name='Catch')
        
        beh_df_clean = beh_df[
            (beh_df['TriggerFolderID'] >= 1) & 
            (beh_df['TriggerFolderID'] <= 30)
        ].copy().reset_index(drop=True)
        
        # 3. Validation and Alignment
        min_len = min(len(eeg_df), len(beh_df_clean))
        excel_vals = beh_df_clean[['TriggerFolderID', 'TriggerStimID']].iloc[:min_len].values
        eeg_vals = eeg_df[['byte1', 'byte2']].iloc[:min_len].values
        
        match_mask = (excel_vals[:, 0] == eeg_vals[:, 0]) & (excel_vals[:, 1] == eeg_vals[:, 1])
        num_mismatches = np.sum(~match_mask)
        match_pct = (np.sum(match_mask) / min_len) * 100 if min_len > 0 else 0
        print(f"DEBUG: Found {len(eeg_df)} standard triggers in EEG.")
        print(f"DEBUG: Found {len(beh_df_clean)} valid trials in Excel.")
        print(f"VALIDATION: {num_mismatches} mismatched trials found ({match_pct:.2f}% match).")
    
        # If mismatch, use Excel labels for condition names
        if num_mismatches > 0:
            print(f"!!! Labels mismatch detected. Using Excel labels for file saving.")
            stim_events_df = eeg_df.iloc[:min_len].copy()
            # Map the combined codes in EEG to the intended Excel labels
            condition_labels = [f"F{int(excel_vals[i,0])}_S{int(excel_vals[i,1])}" for i in range(min_len)]
        else:
            stim_events_df = eeg_df.iloc[:min_len].copy()
            condition_labels = [f"F{int(row['byte1'])}_S{int(row['byte2'])}" for _, row in stim_events_df.iterrows()]
    
        # 4. Epoching Standard Trials
        mne_events = np.column_stack([
            stim_events_df['sample'], 
            np.zeros(min_len), 
            stim_events_df['combined_code']
        ]).astype(int)
        
        unique_codes = stim_events_df[['byte1', 'byte2']].drop_duplicates().values
        event_id_dict = {f"F{int(f)}_S{int(s)}": int(f*1000+s) for f, s in unique_codes}
    
        epochs = mne.Epochs(raw, mne_events, event_id=event_id_dict, tmin=-0.1, tmax=0.8, 
                            baseline=(None, 0), preload=True, verbose=False)
    
        # Epoching Catch Trials
        catch_events = np.column_stack([
            eeg_catch_df['sample'],
            np.zeros(len(eeg_catch_df)),
            eeg_catch_df['combined_code']
        ]).astype(int)
        
        catch_ids = {f"Catch_S{int(s)}": int(255*1000+s) for s in eeg_catch_df['byte2'].unique()}
        epochs_catch = mne.Epochs(raw, catch_events, event_id=catch_ids, tmin=-0.1, tmax=0.8,
                                  baseline=(None, 0), preload=True, verbose=False)
    
        # 5. Restructure Standard to 4D
        data_3d = epochs.get_data()
        cond_names = sorted(list(set(condition_labels)))
        n_reps = pd.Series(condition_labels).value_counts().max()
        
        data_4d = np.full((64, data_3d.shape[2], len(cond_names), n_reps), np.nan, dtype=np.float32)
        c_map = {name: i for i, name in enumerate(cond_names)}
        counters = {i: 0 for i in range(len(cond_names))}
    
        for i in range(len(epochs)):
            label = condition_labels[i]
            c_idx = c_map[label]
            r_idx = counters[c_idx]
            if r_idx < n_reps:
                data_4d[:, :, c_idx, r_idx] = data_3d[i, :64, :]
                counters[c_idx] += 1
               
        # 6. Save
        final_struct = {
            'data_4d': data_4d,
            'data_catch': epochs_catch.get_data()[:, :64, :],
            'Catch_trials_info': Catch_trials_info,
            'channel_names': eeg_ch_names,
            'channel_positions': ch_positions,
            'condition_names': cond_names,
            'event_id_map': event_id_dict,
            'time_vector': epochs.times,
            'sfreq': epochs.info['sfreq']
        }
        np.save(OUTPUT_FILE, final_struct)
        print(f"SUCCESS: {p_id} saved.")    

    except Exception as e:
        print(f"\nFAILED {p_id} with error: {e}")
        traceback.print_exc()

print("\nProcessing complete.")