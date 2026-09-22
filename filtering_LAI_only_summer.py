# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:54:06 2026

@author: gilbars
"""

"""
Created on Wed Jul  8 11:59:23 2026

@author: gilbars
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') # Prevent unnecessary warnings

# =============================================================================
# 1. General Path Settings
# =============================================================================
input_file = r"D:\Stav\research\Israel_polygons\LAI_data\LAI_filtering_identical_LAI_different_sites.csv"
base_dir = os.path.dirname(input_file)

# Paths for Step 2 (Filtering)
cleaned_csv = os.path.join(base_dir, "LAI_Summer_Cleaned_step_2.csv")
cleaning_report_csv = os.path.join(base_dir, "LAI_Summer_Cleaning_Report.csv")
cleaning_plots_dir = os.path.join(base_dir, "LAI_Summer_Cleaning_Plots")

# Paths for Step 3 (Gap Filling)
gapfilled_csv = os.path.join(base_dir, "LAI_Summer_GapFilled_step_3.csv")
gapfilling_report_csv = os.path.join(base_dir, "LAI_Summer_GapFilling_Report.csv")
gapfill_plots_dir = os.path.join(base_dir, "LAI_Summer_GapFilling_Plots")

# Create directories for plots
os.makedirs(cleaning_plots_dir, exist_ok=True)
os.makedirs(gapfill_plots_dir, exist_ok=True)

# =============================================================================
# 2. Data Preparation and Filtering for Summer Months
# =============================================================================
print("=" * 60)
print("Loading LAI data and filtering for summer months (June-October)...")
df = pd.read_csv(input_file)
df.columns = df.columns.str.strip()

df['date'] = pd.to_datetime(df['date'], dayfirst=True, format='mixed')
df['LAI_mean'] = pd.to_numeric(df['LAI_mean'], errors='coerce')
df = df.dropna(subset=['LAI_mean', 'ID'])

df_summer = df[df['date'].dt.month.isin([6, 7, 8, 9, 10])].copy()
df_summer['year'] = df_summer['date'].dt.year

# =============================================================================
# 3. Filtering Step (Calculation Per Year, Plot Per Site)
# =============================================================================
def process_cleaning_site(site_group):
    poly_id = site_group.name
    site_group = site_group.sort_values('date')
    
    # Mathematical calculation separately for each year to avoid connecting seasons
    processed_years = []
    for year, year_group in site_group.groupby('year'):
        rolling_mean = year_group['LAI_mean'].rolling(window=5, center=True, min_periods=1).mean()
        rolling_std = year_group['LAI_mean'].rolling(window=5, center=True, min_periods=1).std()
        
        is_outlier = (np.abs(year_group['LAI_mean'] - rolling_mean) > (1.5 * rolling_std))
        year_group['is_outlier'] = is_outlier.fillna(False)
        processed_years.append(year_group)
        
    # Recombine all years for the site back together
    site_processed = pd.concat(processed_years)
    
    # **Fix here**: Force-preserving the ID column so Pandas does not process it accidentally
    site_processed['ID'] = poly_id
    
    # Calculate percentages
    total_pts = len(site_processed)
    removed_pts = site_processed['is_outlier'].sum()
    removal_rate = (removed_pts / total_pts * 100) if total_pts > 0 else 0
    
    # --- Plotting overall site graph ---
    plt.figure(figsize=(15, 6))
    
    # Original data points and outliers
    plt.scatter(site_processed['date'], site_processed['LAI_mean'], color='lightgray', alpha=0.6, label='Original LAI')
    
    outliers = site_processed[site_processed['is_outlier']]
    if not outliers.empty:
        plt.scatter(outliers['date'], outliers['LAI_mean'], color='red', marker='x', s=70, label='Removed Outliers')
        
    clean = site_processed[~site_processed['is_outlier']]
    
    # Plot green line per year to leave gaps during winter
    for i, (year, clean_year) in enumerate(clean.groupby('year')):
        lbl = 'Cleaned LAI Path' if i == 0 else "" # Prevent duplicate legend entries
        plt.plot(clean_year['date'], clean_year['LAI_mean'], '-o', color='forestgreen', markersize=3, linewidth=1, label=lbl)
    
    plt.title(f"LAI Dynamic Cleaning: Polygon ID {poly_id} | Removed: {removal_rate:.2f}% ({removed_pts}/{total_pts})")
    plt.xlabel("Date")
    plt.ylabel("LAI Mean")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(cleaning_plots_dir, f"Polygon_{poly_id}_cleaning.png"))
    plt.close() 
    
    return site_processed

print("Running dynamic filtering and generating plots...")
df_cleaned_processed = df_summer.groupby('ID', group_keys=False).apply(process_cleaning_site)

# Ensure the index is clean and we have a proper ID column to work with
df_cleaned_processed = df_cleaned_processed.reset_index(drop=True)

# Cleaning report
report_clean = df_cleaned_processed.groupby('ID').agg(
    Total_Points=('is_outlier', 'count'),
    Removed_Points=('is_outlier', 'sum')
).reset_index()
report_clean['Removal_Rate_Percent'] = (report_clean['Removed_Points'] / report_clean['Total_Points'] * 100).round(2)
report_clean.to_csv(cleaning_report_csv, index=False)

# Save the cleaned file
df_clean_final = df_cleaned_processed[~df_cleaned_processed['is_outlier']].copy()
df_clean_final = df_clean_final[['ID', 'date', 'LAI_mean', 'year']]
df_clean_final['date_str'] = df_clean_final['date'].dt.strftime('%d/%m/%Y')
df_clean_final[['ID', 'date_str', 'LAI_mean']].rename(columns={'date_str': 'date'}).to_csv(cleaned_csv, index=False)

# =============================================================================
# 4. Gap Filling Step
# =============================================================================
def process_gapfill_site(site_group):
    # Fix here: Extracting the ID safely directly from group name
    poly_id = site_group.name 
    
    filled_years = []
    # Run gap filling separately for each year to avoid bridging across winter
    for year, year_group in site_group.groupby('year'):
        year_group = year_group.groupby('date')['LAI_mean'].mean().reset_index()
        year_group = year_group.sort_values('date').set_index('date')
        
        if len(year_group) == 0:
            continue
            
        full_date_range = pd.date_range(start=year_group.index.min(), end=year_group.index.max(), freq='D')
        year_reindexed = year_group.reindex(full_date_range)
        year_reindexed.index.name = 'date'
        
        year_reindexed['LAI_filtered'] = year_reindexed['LAI_mean']
        
        is_missing = year_reindexed['LAI_mean'].isna()
        missing_blocks = is_missing.cumsum()
        block_sizes = year_reindexed.groupby(missing_blocks).transform('count')['LAI_mean']
        too_long_gap_mask = is_missing & (block_sizes > 30)
        
        year_reindexed['LAI_filled'] = year_reindexed['LAI_mean'].interpolate(method='linear').bfill().ffill()
        year_reindexed['LAI_mean_smooth'] = year_reindexed['LAI_filled'].rolling(window=21, center=True, min_periods=1).mean()
        year_reindexed.loc[too_long_gap_mask, 'LAI_mean_smooth'] = np.nan
        year_reindexed['year'] = year
        
        filled_years.append(year_reindexed)
        
    if not filled_years:
        return pd.DataFrame()
        
    site_processed = pd.concat(filled_years)
    
    # --- Plotting overall site graph ---
    plt.figure(figsize=(15, 6))
    
    # Plot lines per year to keep explicit gaps during winter
    for i, (year, year_data) in enumerate(site_processed.groupby('year')):
        lbl_filt = 'Filtered LAI (Input)' if i == 0 else ""
        lbl_smooth = 'Ecological 21-Day Smooth Path' if i == 0 else ""
        
        plt.scatter(year_data.index, year_data['LAI_filtered'], color='forestgreen', alpha=0.7, s=15, label=lbl_filt)
        plt.plot(year_data.index, year_data['LAI_mean_smooth'], '-', color='darkorange', linewidth=1.5, label=lbl_smooth)
        
    # Highlight interpolated days
    is_filled_mask = site_processed['LAI_filtered'].isna() & ~site_processed['LAI_mean_smooth'].isna()
    filled_points = site_processed[is_filled_mask]
    
    if len(filled_points) > 0:
        plt.scatter(filled_points.index, filled_points['LAI_mean_smooth'], color='red', marker='.', s=10, alpha=0.3, label='Interpolated Points (<30d)')
            
    total_days = len(site_processed)
    interpolated_count = is_filled_mask.sum()
    unfilled_gap_count = (site_processed['LAI_filtered'].isna() & site_processed['LAI_mean_smooth'].isna()).sum()
    
    plt.title(f"LAI 21-Day Gap Filling: Polygon ID {poly_id}\nFilled: {interpolated_count}/{total_days} days | Skipped (Gap > 30d): {unfilled_gap_count} days")
    plt.xlabel("Date")
    plt.ylabel("LAI Mean")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(gapfill_plots_dir, f"Polygon_{poly_id}_gap_filling.png"))
    plt.close()
    
    # Prepare return DataFrame - ensuring the ID column is properly returned
    group_out = site_processed.reset_index()
    group_out['ID'] = poly_id
    group_out['LAI_mean'] = group_out['LAI_mean_smooth']
    group_out['is_interpolated'] = is_filled_mask.values
    
    return group_out[['ID', 'date', 'LAI_mean', 'is_interpolated']]

print("Running gap filling and generating plots...")
df_gapfilled_processed = df_clean_final.groupby('ID', group_keys=False).apply(process_gapfill_site)
df_gapfilled_processed = df_gapfilled_processed.reset_index(drop=True) 

# Gap filling report
report_gap = df_gapfilled_processed.groupby('ID').agg(
    Total_Days=('is_interpolated', 'count'),
    Interpolated_Days=('is_interpolated', 'sum')
).reset_index()
report_gap['Gap_Percentage'] = (report_gap['Interpolated_Days'] / report_gap['Total_Days'] * 100).round(2)
report_gap.to_csv(gapfilling_report_csv, index=False)

# Save final output
df_final = df_gapfilled_processed.dropna(subset=['LAI_mean']).copy()
df_final['date'] = df_final['date'].dt.strftime('%d/%m/%Y')
df_final[['ID', 'date', 'LAI_mean']].to_csv(gapfilled_csv, index=False)

print("-" * 50)
print("             Process completed successfully!")
print("-" * 50)
print(f"Cleaned file saved at: \n  {cleaned_csv}")
print(f"Gap-filled file saved at: \n  {gapfilled_csv}")
print(f"Plots (one per site with winter gaps) saved in directories: \n  {cleaning_plots_dir}\n  {gapfill_plots_dir}")

