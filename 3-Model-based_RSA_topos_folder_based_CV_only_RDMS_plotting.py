import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap

# --- 1. CONFIGURATION ---
N_MORPH = 21
f_x0 = 10  # Category boundary center (Morph Level 11)
f_k = 1.2   # Sigmoidal steepness
OUTPUT_DIR = r'C:\Users\uqhkarim\OneDrive - The University of Queensland\Documents\Scientific\Projects\Pareidolia\EEG\Main data\Processed_Outputs\Integrated_Psychometric_Results'

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

# --- 2. FUNCTIONS ---
def sigmoid_func(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))

def create_rdm(vals):
    return np.abs(vals[:, None] - vals)

# --- 3. DATA PREPARATION & CUSTOM MONOCHROME CMAPS ---
lvls = np.arange(N_MORPH)

# Base hex codes matching your models
blue_color = '#1f77b4'
red_color = '#d62728'
green_color = '#2ca02c'

# Custom 3-segment color maps: [Low Dissimilarity -> Mid -> High Dissimilarity]
cmap_linear = LinearSegmentedColormap.from_list("LinearCmap", [blue_color, "white", "black"])
cmap_sigmoid = LinearSegmentedColormap.from_list("SigmoidCmap", [red_color, "white", "black"])
cmap_step = LinearSegmentedColormap.from_list("StepCmap", [green_color, "white", "black"])

models = {
# Normalizing by (N_MORPH - 1) ensures index 20 / 20 = 1.0
    'Linear': {'vec': lvls / (N_MORPH - 1), 'color': blue_color, 'cmap': cmap_linear},
    'Sigmoid': {'vec': sigmoid_func(lvls, f_x0, f_k), 'color': red_color, 'cmap': cmap_sigmoid},
    'Step': {'vec': np.where(lvls < f_x0, 0, 1), 'color': green_color, 'cmap': cmap_step}
}

# --- 4. PLOTTING ---
# Adjusted figsize and grid spacing to comfortably accommodate 21 individual tick marks
fig, axes = plt.subplots(2, 3, figsize=(18, 12), gridspec_kw={'height_ratios': [1, 1.3]})

tick_positions = np.arange(N_MORPH)
tick_labels = [str(x) for x in range(1, N_MORPH + 1)]

for i, (name, data) in enumerate(models.items()):
    vec = data['vec']
    color = data['color']
    custom_cmap = data['cmap']
    
    # --- TOP ROW: Psychometric Curves ---
    ax_curve = axes[0, i]
    ax_curve.plot(lvls, vec, color=color, lw=4, marker='o', markersize=5)
    ax_curve.set_title(f"{name} Profile", fontweight='bold', fontsize=14)
    ax_curve.set_ylim([-0.1, 1.1])
    
    # 1. Update Top Panel Ticks to 1:21
    ax_curve.set_xticks(tick_positions)
    ax_curve.set_xticklabels(tick_labels, fontsize=8, rotation=45)
    ax_curve.grid(True, alpha=0.2)
    if i == 0:
        ax_curve.set_ylabel("Probability / Level (A.U.)", fontweight='bold', fontsize=12)
    
    # --- BOTTOM ROW: RDMs ---
    ax_rdm = axes[1, i]
    rdm = create_rdm(vec)
    
    # 2. Map custom colormap and force aspect='equal' to preserve perfect squares
    im = ax_rdm.imshow(rdm, cmap=custom_cmap, origin='lower', aspect='equal', vmin=0, vmax=1)
    ax_rdm.set_xlabel("Morph Level", fontweight='bold', fontsize=12)
    
    # 1. Update RDM X and Y axes ticks to 1:21
    ax_rdm.set_xticks(tick_positions)
    ax_rdm.set_xticklabels(tick_labels, fontsize=8, rotation=45)
    ax_rdm.set_yticks(tick_positions)
    ax_rdm.set_yticklabels(tick_labels, fontsize=8)
    
    if i == 0:
        ax_rdm.set_ylabel("Morph Level", fontweight='bold', fontsize=12)
        
    # Add localized colorbar for each RDM profile to accurately reflect its custom map scaling
    cbar = fig.colorbar(im, ax=ax_rdm, orientation='horizontal', pad=0.15, shrink=0.7)
    cbar.set_label('Dissimilarity Scale', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

# Global Layout Formatting
plt.suptitle("Neuro-psychometric Profiles & Resulting RSA RDMs", 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout()

# Save high-resolution vector PDF for Illustrator figure construction
pdf_save_path = os.path.join(OUTPUT_DIR, "neuro_psychometric_rsa_profiles.pdf")
print(f"Saving high-resolution vector layout to: {pdf_save_path}")
plt.savefig(pdf_save_path, format='pdf', bbox_inches='tight', dpi=600)

plt.show()