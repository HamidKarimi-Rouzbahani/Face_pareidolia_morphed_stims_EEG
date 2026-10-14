import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define File Paths
data_dir = r"C:\Users\uqhkarim\OneDrive - The University of Queensland\Documents\Scientific\Projects\Pareidolia\Psycho\Experiment 2\Results"
json_filename = "images_faceness.json"
json_path = os.path.join(data_dir, json_filename)
pdf_output_path = os.path.join(data_dir, "psychometric_function.pdf")

# 2. Load and Parse JSON Data
with open(json_path, 'r') as f:
    data = json.load(f)

parsed_data = []

for key, val in data.items():
    try:
        morph_level = float(key.split("___")[-1])
    except ValueError:
        continue
    
    faceness_scores = val.get("facenesses", [])
    
    for sub_idx, score in enumerate(faceness_scores):
        if score is None or score == 'NaN' or np.isnan(float(score)):
            score_val = np.nan
        else:
            score_val = float(score)
            
        parsed_data.append({
            "morph_level": morph_level,
            "subject": f"Sub_{sub_idx + 1}",
            "faceness": score_val
        })

df = pd.DataFrame(parsed_data)

# 3. Pivot Data to get Matrix: Morph Levels x Subjects
pivot_df = df.groupby(["morph_level", "subject"])["faceness"].mean().unstack()

# Extract arrays for plotting
morph_levels = pivot_df.index.to_numpy()
individual_means = pivot_df.to_numpy()

# Calculate global metrics across subjects
grand_mean = pivot_df.mean(axis=1, skipna=True).to_numpy()
grand_sem = pivot_df.sem(axis=1, skipna=True).to_numpy()

# 4. Compute Slopes (Differences between consecutive points)
# Slope = ΔY / ΔX
dy = np.diff(grand_mean)
dx = np.diff(morph_levels)
slopes = dy / dx

# Midpoints for plotting slopes between the discrete morph levels
slope_midpoints = morph_levels[:-1] + (dx / 2.0)

# 5. Plotting Architecture
# Horizontally wide/elongated canvas layout
fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=300)

# --- Left Axis (Faceness Score) ---
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_color('black')
ax1.spines['bottom'].set_color('black')

ax1.xaxis.set_ticks_position('bottom')
ax1.yaxis.set_ticks_position('left')
ax1.tick_params(axis='both', direction='out', colors='black', length=6, width=1.2)
ax1.grid(False)

# Plot Individual Subject Variabilities (Faded background black paths)
for sub_i in range(individual_means.shape[1]):
    ax1.plot(
        morph_levels, 
        individual_means[:, sub_i], 
        color='black', 
        alpha=0.12, 
        linewidth=0.75, 
        zorder=1
    )

# Connect the Mean Dots directly with a solid black line
line_mean, = ax1.plot(
    morph_levels,
    grand_mean,
    color='black',
    linewidth=2.0,
    label='Grand Mean Path',
    zorder=2
)

# Plot Grand Mean Data Points with SEM Error Bars (Stark Black)
ax1.errorbar(
    morph_levels, 
    grand_mean, 
    yerr=grand_sem, 
    fmt='o', 
    color='black', 
    ecolor='black', 
    elinewidth=1.5, 
    capsize=0,         
    ms=6, 
    zorder=3
)

ax1.set_title("Psychometric Function & Slope Across Morph Levels", fontsize=12, fontweight='bold', pad=15, color='black')
ax1.set_xlabel("Morph Level", fontsize=10, labelpad=8, color='black')
ax1.set_ylabel("Faceness Score", fontsize=10, labelpad=8, color='black')

ax1.set_xticks(np.sort(morph_levels))
ax1.set_xlim(morph_levels.min() - 0.25, morph_levels.max() + 0.25)
ax1.set_ylim(-0.5, 10.5)


# --- Right Axis (Slope Changes) ---
ax2 = ax1.twinx()  # Shared X axis, independent Y axis
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_color('red')
ax2.spines['right'].set_visible(True)

ax2.yaxis.set_ticks_position('right')
ax2.tick_params(axis='y', direction='out', colors='red', length=6, width=1.2)
ax2.grid(False)

# Plot the slopes in red
line_slope, = ax2.plot(
    slope_midpoints,
    slopes,
    color='red',
    linewidth=1.75,
    linestyle='--',
    marker='s',
    markersize=5,
    label='Slope (ΔFaceness / ΔMorph)',
    zorder=4
)

ax2.set_ylabel("Slope (Rate of Change)", fontsize=10, labelpad=8, color='red')

# Dynamically buffer slope Y-limits for clean visual distribution
slope_min, slope_max = np.nanmin(slopes), np.nanmax(slopes)
slope_buffer = max((slope_max - slope_min) * 0.15, 0.5)
ax2.set_ylim(slope_min - slope_buffer, slope_max + slope_buffer)


# --- Consolidated Legend ---
lines = [line_mean, line_slope]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, frameon=False, loc='upper left', fontsize=9)

plt.tight_layout()

# Save Figure as Vector PDF
plt.savefig(pdf_output_path, format='pdf', bbox_inches='tight')

# Render figure interactively on screen
plt.show()

print(f"Successfully processed metrics and saved plot to:\n{pdf_output_path}")