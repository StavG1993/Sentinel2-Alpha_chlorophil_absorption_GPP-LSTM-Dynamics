#%%

# Step 1: Filtering based on spectral signature sharp features (mainly large cloud cover)

import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. Path settings
file_path = r"D:\Stav\research\Israel_polygons\final_excels\Sentinel2_Full_Data_80_Sites.csv"
output_dir = r"D:\Stav\research\Israel_polygons\final_excels\Conservative_Spectral_Dark"
if not os.path.exists(output_dir): os.makedirs(output_dir)

# 2. Loading and initial preprocessing
df = pd.read_csv(file_path)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['Site_ID'] = df['FORNAME'].astype(str) + "_" + df['REP'].astype(str)
bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

# 3. Conservative filtering logic (preserves fires and drought events)
def apply_conservative_logic(group):
    monthly_medians = group.groupby('month')[bands].transform('median')
    group['Slope'] = (group['B8'] - group['B4']) / (group['B8'] + group['B4'])
    
    # Conditions for removal (must be extreme)
    is_extreme_blue = group['B2'] > (monthly_medians['B2'] * 2.0)
    is_very_low_swir = group['B12'] < (monthly_medians['B12'] * 0.5)
    is_flat_signature = group['Slope'] < 0.1
    
    group['is_cloud'] = (is_extreme_blue & is_flat_signature) | (is_extreme_blue & is_very_low_swir)
    return group

df = df.groupby('Site_ID', group_keys=False).apply(apply_conservative_logic)

# 4. Generating plots with dark and clear colors
for site_id, group in df.groupby('Site_ID'):
    total_points = len(group)
    kept_points = len(group[group['is_cloud'] == False])
    percent_kept = (kept_points / total_points) * 100
    
    plt.figure(figsize=(12, 7))
    kept = group[group['is_cloud'] == False]
    removed = group[group['is_cloud'] == True]
    
    # Plot kept signatures - darker color and reduced opacity
    for _, row in kept.iterrows():
        plt.plot(bands, row[bands], color='darkgreen', alpha=0.15, linewidth=0.8)
    
    # Plot removed signatures - bold red
    for _, row in removed.iterrows():
        plt.plot(bands, row[bands], color='red', alpha=0.8, linewidth=1.2, label=f"Removed: {row['date'].date()}")
    
    # Highlighted median profile line in black
    plt.plot(bands, group[bands].median(), color='black', linewidth=2.5, linestyle='-', label='Site Median Profile')

    # Title and styling
    plt.title(f"Site: {site_id} | Data Kept: {percent_kept:.1f}% ({kept_points}/{total_points})", fontsize=14)
    plt.ylabel("Reflectance", fontsize=12)
    plt.xlabel("Bands", fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Legend handling
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(os.path.join(output_dir, f"Analysis_{site_id}.png"), bbox_inches='tight', dpi=150)
    plt.close()

# 5. Saving the CSV file
df_final = df[df['is_cloud'] == False].drop(columns=['Slope', 'is_cloud', 'month'])
df_final.to_csv(os.path.join(output_dir, "final_sentinel_cleaned.csv"), index=False)

print(f"Process completed! {df['is_cloud'].sum()} observations were removed.")

#%%
# Step 2 - Filtering based on NDVI time series for each site

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. Path settings - use the spectrally filtered file
input_file = r"D:\Stav\research\Israel_polygons\final_excels\Conservative_Spectral_Dark\final_sentinel_cleaned.csv"
output_dir = r"D:\Stav\research\Israel_polygons\final_excels\Final_Physiological_Cleaning"
if not os.path.exists(output_dir): os.makedirs(output_dir)

# 2. Loading and date handling (resolving the ValueError)
df = pd.read_csv(input_file)
df['date'] = pd.to_datetime(df['date'], dayfirst=True) # Fix for ValueError
df = df.sort_values(['Site_ID', 'date'])

# Calculate NDVI (ensure updated column exists)
df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'])

# 3. Filtering function based on rate of change (Slope) and distance from median
def filter_physiological_spikes(group):
    group = group.reset_index(drop=True)
    
    # Calculate time differences (in days) and NDVI differences
    group['days_diff'] = group['date'].diff().dt.days
    group['ndvi_diff'] = group['NDVI'].diff()
    
    # Daily NDVI rate of change
    group['rate'] = group['ndvi_diff'] / group['days_diff']
    group['rate_next'] = group['rate'].shift(-1)
    
    # Define "zigzag" pattern (sharp drop followed by sharp recovery in a short window)
    # A forest does not recover at 0.015 NDVI units per day
    is_spike = (group['rate'] < -0.015) & (group['rate_next'] > 0.015)
    
    # Additional check: distance from local median (3-observation moving window)
    rolling_med = group['NDVI'].rolling(window=3, center=True, min_periods=1).median()
    is_outlier = (group['NDVI'] < (rolling_med - 0.12))
    
    group['is_temporal_outlier'] = is_spike | is_outlier
    return group

# Run filtering for each site independently
df = df.groupby('Site_ID', group_keys=False).apply(filter_physiological_spikes)

# 4. Generating plots with dark and clear data points
for site_id, group in df.groupby('Site_ID'):
    total = len(group)
    kept_df = group[group['is_temporal_outlier'] == False]
    removed_df = group[group['is_temporal_outlier'] == True]
    percent_kept = (len(kept_df) / total) * 100
    
    plt.figure(figsize=(15, 6))
    
    # Removed points (red crosses)
    plt.scatter(removed_df['date'], removed_df['NDVI'], color='red', marker='x', s=70, 
                label=f'Removed Spikes ({len(removed_df)})', zorder=5)
    
    # Kept points (dark green with connecting line)
    plt.plot(kept_df['date'], kept_df['NDVI'], color='darkgreen', marker='o', 
             markersize=4, linewidth=1.2, alpha=0.8, label='Cleaned NDVI')
    
    # Faint grey background line for original series
    plt.plot(group['date'], group['NDVI'], color='grey', alpha=0.15, linestyle='--', zorder=1)

    # Title and styling
    plt.title(f"Physiological NDVI Cleaning: {site_id} | Kept: {percent_kept:.1f}% ({len(kept_df)}/{total})", fontsize=14)
    plt.ylabel("NDVI", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc='best')
    
    plt.savefig(os.path.join(output_dir, f"Final_TS_{site_id}.png"), bbox_inches='tight')
    plt.close()

# 5. Saving the final cleaned CSV file
# Drop auxiliary calculation columns
cols_to_drop = ['days_diff', 'ndvi_diff', 'rate', 'rate_next', 'is_temporal_outlier']
final_df = df[df['is_temporal_outlier'] == False].drop(columns=cols_to_drop)

final_df.to_csv(os.path.join(output_dir, "final_sentinel_physiologically_cleaned.csv"), index=False)

print(f"--- Filtering process completed ---")
print(f"Total noise points removed in this step: {df['is_temporal_outlier'].sum()}")
print(f"Files and plots saved at: {output_dir}")

#%%

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. Path settings
input_file = r"D:\Stav\research\Israel_polygons\final_excels\PAR_Final_All_80_Sites.csv"
output_dir = r"D:\Stav\research\Israel_polygons\final_excels\PAR_Temporal_Cleaning"
if not os.path.exists(output_dir): os.makedirs(output_dir)

# 2. Loading and consolidating sites - fix applied here
df = pd.read_csv(input_file)

# Using format='mixed' resolves unconverted data issues
df['date'] = pd.to_datetime(df['date'], format='mixed', dayfirst=False) 

# Create unique Site_ID
df['Site_ID'] = df['FORNAME'].astype(str) + "_" + df['REP'].astype(str)
df = df.sort_values(['Site_ID', 'date'])

# Define exact column name
par_col = 'PAR_daily_mol'

# 3. Spike filtering function for PAR (remains unchanged)
def filter_par_spikes(group):
    group = group.reset_index(drop=True)
    
    # 7-observation rolling window for trend calculation
    group['PAR_median'] = group[par_col].rolling(window=7, center=True, min_periods=1).median()
    
    # Condition 1: Relative deviation (30% lower than median) - typical for severe cloud cover or sensor errors
    group['rel_diff'] = (group[par_col] - group['PAR_median']) / group['PAR_median']
    is_low_outlier = group['rel_diff'] < -0.30
    
    # Condition 2: Rapid zigzag pattern
    group['par_diff_prev'] = group[par_col].diff()
    group['par_diff_next'] = group[par_col].shift(-1) - group[par_col]
    is_zigzag = (group['par_diff_prev'] < -5) & (group['par_diff_next'] > 5)
    
    group['is_par_spike'] = is_low_outlier | is_zigzag
    return group

# Run filtering
df = df.groupby('Site_ID', group_keys=False).apply(filter_par_spikes)

# 4. Visualization (with purple crosses)
print(f"Generating plots for {df['Site_ID'].nunique()} sites...")

for site_id, group in df.groupby('Site_ID'):
    total = len(group)
    kept_df = group[group['is_par_spike'] == False]
    removed_df = group[group['is_par_spike'] == True]
    
    # Protection against division by zero if site is empty
    percent_kept = (len(kept_df) / total) * 100 if total > 0 else 0
    
    plt.figure(figsize=(15, 6))
    
    # Original series (grey)
    plt.plot(group['date'], group[par_col], color='grey', alpha=0.2, label='Original PAR')
    
    # Kept data (orange)
    plt.plot(kept_df['date'], kept_df[par_col], color='darkorange', marker='o', 
             markersize=3, linewidth=1, label='Cleaned PAR')
    
    # Removed points (purple crosses)
    if not removed_df.empty:
        plt.scatter(removed_df['date'], removed_df[par_col], color='purple', marker='x', 
                    s=50, label=f'Removed Spikes ({len(removed_df)})', zorder=5)
    
    plt.title(f"PAR Time Series Cleaning: {site_id} | Kept: {percent_kept:.1f}%", fontsize=14)
    plt.ylabel("PAR (mol m-2 day-1)")
    plt.grid(True, alpha=0.2, linestyle='--')
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, f"PAR_Clean_{site_id}.png"), bbox_inches='tight')
    plt.close()

# 5. Saving the cleaned CSV file
cols_to_drop = ['PAR_median', 'rel_diff', 'par_diff_prev', 'par_diff_next', 'is_par_spike']
final_par_df = df[df['is_par_spike'] == False].drop(columns=cols_to_drop)
final_par_df.to_csv(os.path.join(output_dir, "final_PAR_cleaned.csv"), index=False)

print(f"Success! Filtered PAR saved. Total spikes removed: {df['is_par_spike'].sum()}")

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Path settings
base_path = r"D:\Stav\research\Israel_polygons\final_excels"
input_file = os.path.join(base_path, "Final_Master_Database.csv")
output_csv = os.path.join(base_path, "Final_Cleaned_alphaREPAR_Data.csv")
report_csv = os.path.join(base_path, "alphaREPAR_Cleaning_Report.csv")
plots_dir = os.path.join(base_path, "alphaREPAR_Cleaning_Plots")

# Create plots directory if it does not exist
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# 2. Loading and preparation
df = pd.read_csv(input_file)
df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed')
df['alphaRE'] = pd.to_numeric(df['alphaRE'], errors='coerce')
df['PAR_daily_mol'] = pd.to_numeric(df['PAR_daily_mol'], errors='coerce')
df = df.dropna(subset=['alphaRE', 'PAR_daily_mol'])
df['alphaREPAR_approx'] = df['alphaRE'] * df['PAR_daily_mol']

# 3. Processing and report generation function per site
def process_site(group):
    site_id = group['Site_ID'].iloc[0]
    group = group.sort_values('date')
    
    # Calculate outliers
    rolling_mean = group['alphaREPAR_approx'].rolling(window=5, center=True, min_periods=1).mean()
    rolling_std = group['alphaREPAR_approx'].rolling(window=5, center=True, min_periods=1).std()
    
    # Identify outliers (1.5 standard deviations)
    is_outlier = (np.abs(group['alphaREPAR_approx'] - rolling_mean) > (1.5 * rolling_std))
    group['is_outlier'] = is_outlier.fillna(False)
    
    # --- Plot creation and saving ---
    plt.figure(figsize=(12, 6))
    plt.scatter(group['date'], group['alphaREPAR_approx'], color='lightgray', alpha=0.6, label='Original Data')
    
    outliers = group[group['is_outlier']]
    plt.scatter(outliers['date'], outliers['alphaREPAR_approx'], color='red', marker='x', s=70, label='Removed')
    
    clean = group[~group['is_outlier']]
    plt.plot(clean['date'], clean['alphaREPAR_approx'], '-o', color='forestgreen', markersize=3, linewidth=1, label='Cleaned Path')
    
    plt.title(f"alphaREPAR Cleaning: Site {site_id}")
    plt.ylabel("alphaREPAR (alphaRE * PAR)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save to directory
    plt.savefig(os.path.join(plots_dir, f"Site_{site_id}_cleaning.png"))
    plt.close() # Close plot to free up memory
    
    return group

# 4. Run execution
df_processed = df.groupby('Site_ID', group_keys=False).apply(process_site)

# 5. Create numerical summary report (removal count per site)
report = df_processed.groupby('Site_ID').agg(
    Total_Points=('is_outlier', 'count'),
    Removed_Points=('is_outlier', 'sum')
).reset_index()

report['Removal_Rate_Percent'] = (report['Removed_Points'] / report['Total_Points'] * 100).round(2)
report.to_csv(report_csv, index=False)

# 6. Save final cleaned CSV file
df_clean = df_processed[~df_processed['is_outlier']].copy()
df_clean.to_csv(output_csv, index=False)

print("-" * 30)
print(f"1. Clean file saved at: {output_csv}")
print(f"2. Removal report (counts and percentages) saved at: {report_csv}")
print(f"3. All plots (80+) saved in directory: {plots_dir}")
print("-" * 30)
print(f"Overall average removal rate: {report['Removal_Rate_Percent'].mean():.2f}%")