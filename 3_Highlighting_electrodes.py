import numpy as np
import matplotlib.pyplot as plt
import mne
import os

# Set high-resolution print standards
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600

# 1. Load standard BioSemi 64-channel montage and establish info structure
montage = mne.channels.make_standard_montage('biosemi64')
ch_names = montage.ch_names

info = mne.create_info(ch_names=ch_names, sfreq=1000, ch_types='eeg')
info.set_montage(montage)

# 2. Filter channels using your exact prefixes
ANT_PREFIXES = ('Fp', 'AF', 'F', 'FC')
POST_PREFIXES = ('CP', 'P', 'PO', 'O')

ant_channels = [ch for ch in ch_names if ch.startswith(ANT_PREFIXES)]
post_channels = [ch for ch in ch_names if ch.startswith(POST_PREFIXES)]
comb_channels = ant_channels + post_channels

# 3. Define standard exact colors matching your primary scripts
color_ant = 'tab:red'
color_post = 'tab:blue'

# ==========================================================================
# FIGURE 1: 3-PANEL LAYOUT (Scaled down by 70.7% for half spatial A4 area)
# ==========================================================================
fig, axes = plt.subplots(1, 3, figsize=(14.85, 4.95), facecolor='white') # Scaled down from 21x7

# Style parameters updated for clean publication ratios at half spatial size
marker_style = dict(marker='o', markerfacecolor='k', markeredgecolor='k', markersize=5.5, alpha=0.12)
selected_style = dict(marker='o', markerfacecolor='k', markeredgecolor='k', markersize=8.5, alpha=1.0)
title_font = {'fontsize': 11, 'fontweight': 'bold', 'pad': 10}

def render_channel_selection_panel(ax, selected_list, title_text):
    ax.set_title(title_text, **title_font)
    
    mne.viz.plot_topomap(
        np.zeros(len(ch_names)), 
        info, 
        axes=ax, 
        show=False, 
        cmap='binary', 
        vlim=(0, 0), 
        contours=0, 
        sensors=True, 
        extrapolate='head'
    )
    
    sensor_collection = ax.collections[0]
    projected_positions = sensor_collection.get_offsets()
    sensor_collection.set_visible(False)
    
    for idx, ch in enumerate(ch_names):
        x, y = projected_positions[idx]
        
        if ch in selected_list:
            ax.plot(x, y, **selected_style)
            ax.text(x, y + 0.004, ch, fontsize=7.5, fontweight='bold',
                    ha='center', va='bottom', alpha=1.0, color='black')
        else:
            ax.plot(x, y, **marker_style)
            ax.text(x, y + 0.004, ch, fontsize=6.0, fontweight='normal',
                    ha='center', va='bottom', alpha=0.2, color='gray')
            
    ax.axis('off')

# Generate Panels for Figure 1
render_channel_selection_panel(axes[0], ant_channels, "Anterior Channel Selection")
render_channel_selection_panel(axes[1], post_channels, "Posterior Channel Selection")
render_channel_selection_panel(axes[2], comb_channels, "Combined Channel Selection")

fig.tight_layout()
fig.savefig('EEG_Channel_Selection_3Panels_Scaled.pdf', format='pdf', bbox_inches='tight')
fig.savefig('EEG_Channel_Selection_3Panels_Scaled.png', format='png', dpi=600, bbox_inches='tight')


# ==========================================================================
# FIGURE 2: NEW SINGLE-PANEL COLOR-CODED LAYOUT (Scaled for Illustrator A4 import)
# ==========================================================================
fig_color, ax_color = plt.subplots(figsize=(6.0, 6.0), facecolor='white')
ax_color.set_title("Color-Coded ROI Topography", fontsize=12, fontweight='bold', pad=12)

mne.viz.plot_topomap(
    np.zeros(len(ch_names)), 
    info, 
    axes=ax_color, 
    show=False, 
    cmap='binary', 
    vlim=(0, 0), 
    contours=0, 
    sensors=True, 
    extrapolate='head'
)

sensor_collection_color = ax_color.collections[0]
projected_positions_color = sensor_collection_color.get_offsets()
sensor_collection_color.set_visible(False)

for idx, ch in enumerate(ch_names):
    x, y = projected_positions_color[idx]
    
    # Check if channel belongs to Ant (Red) or Post (Blue) groups
    if ch in ant_channels:
        current_style = dict(marker='o', markerfacecolor=color_ant, markeredgecolor=color_ant, markersize=8.5, alpha=1.0)
        ax_color.plot(x, y, **current_style)
        ax_color.text(x, y + 0.004, ch, fontsize=8.0, fontweight='bold',
                      ha='center', va='bottom', alpha=1.0, color='black')
    elif ch in post_channels:
        current_style = dict(marker='o', markerfacecolor=color_post, markeredgecolor=color_post, markersize=8.5, alpha=1.0)
        ax_color.plot(x, y, **current_style)
        ax_color.text(x, y + 0.004, ch, fontsize=8.0, fontweight='bold',
                      ha='center', va='bottom', alpha=1.0, color='black')
    else:
        # Gray out non-selected channels cleanly
        bg_style = dict(marker='o', markerfacecolor='k', markeredgecolor='k', markersize=5.5, alpha=0.10)
        ax_color.plot(x, y, **bg_style)
        ax_color.text(x, y + 0.004, ch, fontsize=6.0, fontweight='normal',
                      ha='center', va='bottom', alpha=0.18, color='gray')

ax_color.axis('off')

# Custom clean legend for presentation clarity
ax_color.plot([], [], 'o', color=color_ant, label='Anterior Region', markersize=8)
ax_color.plot([], [], 'o', color=color_post, label='Posterior Region', markersize=8)
ax_color.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9)

fig_color.tight_layout()
fig_color.savefig('EEG_ROI_Topography_ColorCoded.pdf', format='pdf', bbox_inches='tight')
fig_color.savefig('EEG_ROI_Topography_ColorCoded.png', format='png', dpi=600, bbox_inches='tight')


# ==========================================================================
# FIGURE 3: UNIFORM REGION TOPOGRAPHIES (All Blue vs All Red)
# ==========================================================================
fig_uniform, axes_uniform = plt.subplots(1, 2, figsize=(11.0, 5.5), facecolor='white')

uniform_configs = [
    {'color': color_post, 'title': 'All-Blue ROI Selection', 'filename_suffix': 'AllBlue'},
    {'color': color_ant, 'title': 'All-Red ROI Selection', 'filename_suffix': 'AllRed'}
]

for i, config in enumerate(uniform_configs):
    ax = axes_uniform[i]
    ax.set_title(config['title'], **title_font)
    
    mne.viz.plot_topomap(
        np.zeros(len(ch_names)), 
        info, 
        axes=ax, 
        show=False, 
        cmap='binary', 
        vlim=(0, 0), 
        contours=0, 
        sensors=True, 
        extrapolate='head'
    )
    
    sensor_coll = ax.collections[0]
    proj_pos = sensor_coll.get_offsets()
    sensor_coll.set_visible(False)
    
    for idx, ch in enumerate(ch_names):
        x, y = proj_pos[idx]
        
        # Color both anterior and posterior channels with the uniform color
        if ch in comb_channels:
            current_style = dict(marker='o', markerfacecolor=config['color'], markeredgecolor=config['color'], markersize=8.5, alpha=1.0)
            ax.plot(x, y, **current_style)
            ax.text(x, y + 0.004, ch, fontsize=8.0, fontweight='bold',
                    ha='center', va='bottom', alpha=1.0, color='black')
        else:
            ax.plot(x, y, **marker_style)
            ax.text(x, y + 0.004, ch, fontsize=6.0, fontweight='normal',
                    ha='center', va='bottom', alpha=0.18, color='gray')
            
    ax.axis('off')

fig_uniform.tight_layout()
fig_uniform.savefig('EEG_ROI_Topography_UniformRegions.pdf', format='pdf', bbox_inches='tight')
fig_uniform.savefig('EEG_ROI_Topography_UniformRegions.png', format='png', dpi=600, bbox_inches='tight')

plt.show()