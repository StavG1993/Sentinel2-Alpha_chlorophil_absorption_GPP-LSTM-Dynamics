# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:10:33 2026

@author: gilbars
"""





# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:10:33 2026

@author: gilbars
"""





#%%

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

print("⏳ Step 1: Loading the updated data file (24.5.26)...")
input_path = r"D:\Stav\research\Israel_polygons\final_excels\data_for_random_forest_lag_24.5.26.csv"
output_dir = r"D:\Stav\research\Israel_polygons\random forest"
output_path = os.path.join(output_dir, "boruta_confirmed_static_features.xlsx")

os.makedirs(output_dir, exist_ok=True)
df = pd.read_csv(input_path)
print(f"✅ File loaded. Found {len(df):,} rows and {len(df.columns)} columns.")

print("⏳ Step 2: Performing feature engineering and handling static categorical variables...")
# Converting categorical variables to numeric indicators within the same column (0, 1, ...)
static_categorical = ['TreeType', 'RockType', 'Aspect']
for col in static_categorical:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = pd.factorize(df[col])[0]

# Calculating forest age
date_col = 'date'
df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
if 'Planting_year' in df.columns:
    df['Forest_Age'] = df[date_col].dt.year - df['Planting_year']
    df['Forest_Age'] = df['Forest_Age'].fillna(df['Forest_Age'].mean())

print("⏳ Step 3: Filtering to keep only static and topographic variables...")
target_col = 'GPP'

# Expanded and full list of your static and topographic features
static_features_to_keep = [
    'TreeType', 'RockType', 'Aspect', 'Forest_Age', 
    'T/He_2022', 'Altitude', 'Slope', 'Rock cover %'
]

# Filtering features that actually exist in your dataset from the static list
existing_static_features = [col for col in static_features_to_keep if col in df.columns]
X_df = df[existing_static_features].copy()

# Ensuring all columns are numeric and dropping NaNs in sync with the Target variable
X_df[target_col] = df[target_col]
X_df = X_df.dropna()

y = X_df[target_col].values
X_df = X_df.drop(columns=[target_col])

feature_names = np.array(X_df.columns)
X = X_df.values

print("\n📋 Here is the list of all static and topographic features Boruta will evaluate:")
print("-" * 60)
for i, name in enumerate(feature_names, 1):
    print(f"{i}. {name}")
print("-" * 60)
print(f"   - Total selected static predictor features: {len(feature_names)}")

print("\n⏳ Step 4: Running the Boruta algorithm for static variables...")
rf_engine = RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
boruta_selector = BorutaPy(rf_engine, n_estimators='auto', verbose=2, random_state=42, max_iter=50)
boruta_selector.fit(X, y)
print("✅ Boruta algorithm execution completed!")

print("⏳ Step 5: Filtering and organizing confirmed static features only...")
confirmed_mask = boruta_selector.support_
selected_features = feature_names[confirmed_mask]

boruta_results_df = pd.DataFrame({'Confirmed_Static_Features_Boruta': selected_features})

print(f"\n🎯 Out of {len(feature_names)} static variables, Boruta confirmed and retained {len(selected_features)} features.")
print(boruta_results_df.to_string(index=False))

print("\n⏳ Step 6: Saving the list of confirmed features...")
boruta_results_df.to_excel(output_path, index=False)
print(f"✨ Expanded static feature selection file saved at:\n 📂 {output_path}")





#%%

## Dynamic Feature Selection Based on New Variables
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

print("⏳ Step 1: Loading the updated data file (24.5.26)...")
input_path = r"D:\Stav\research\Israel_polygons\final_excels\data_for_LSTM_final_with_features_8.6.26.csv" 
output_dir = r"D:\Stav\research\Israel_polygons\random forest"
output_path = os.path.join(output_dir, "boruta_confirmed_dynamic_features_8.6.26.xlsx")

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Loading data
df = pd.read_csv(input_path)
print(f"✅ File loaded. Found {len(df):,} rows and {len(df.columns)} columns.")

print("⏳ Step 2: Filtering out static variables, identifiers, and temporal variables...")
target_col = 'GPP'

# ❌ Explicit list of columns to remove: identifiers, temporal, and all static/topographic/structural variables
columns_to_drop = [
    'Date', 'Site_ID', 'ID', 'Planting_year',
    'TreeType', 'RockType', 'Aspect',
    'T/He_2022', 'Altitude', 'Slope', 'Rock cover %', 'is_HW'
]

# Dropping existing columns from dataset
existing_drops = [col for col in columns_to_drop if col in df.columns]
X_df = df.drop(columns=existing_drops)

# Ensuring all remaining columns are numeric (dynamic variables, lags, SPEI, heatwaves, dry days)
X_df = X_df.select_dtypes(include=[np.number])

print("⏳ Step 3: Cleaning NaN rows (synchronizing Lags and SPEI with Target)...")
# Synchronized NaN cleanup
X_df[target_col] = df[target_col]
X_df = X_df.dropna()

y = X_df[target_col].values
X_df = X_df.drop(columns=[target_col])

# Saving updated feature names and converting to NumPy array for Boruta
feature_names = np.array(X_df.columns)
X = X_df.values

# 📊 Printing full dynamic feature list in the Console for complete inspection:
print("\n📋 Here is the list of all dynamic variables Boruta will evaluate (verify all are present):")
print("-" * 75)
for i, name in enumerate(feature_names, 1):
    print(f"{i}. {name}")
print("-" * 75)
print(f"   - Total selected dynamic predictor features: {len(feature_names)}")
print(f"   - Total valid training rows after NaN cleanup: {X.shape[0]:,}")

print("\n⏳ Step 4: Running Boruta algorithm for dynamic variables... (This may take several minutes)")
rf_engine = RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42)
boruta_selector = BorutaPy(rf_engine, n_estimators='auto', verbose=2, random_state=42, max_iter=50)

# Running feature selection process
boruta_selector.fit(X, y)
print("✅ Boruta algorithm execution completed!")

print("⏳ Step 5: Filtering and organizing confirmed dynamic features only...")
confirmed_mask = boruta_selector.support_
selected_features = feature_names[confirmed_mask]

boruta_results_df = pd.DataFrame({
    'Confirmed_Dynamic_Features_Boruta': selected_features
})

print(f"\n🎯 Out of {len(feature_names)} dynamic variables, Boruta confirmed and retained {len(selected_features)} features.")
print("📋 List of final selected dynamic features:")
print(boruta_results_df.to_string(index=False))

print("\n⏳ Step 6: Saving the list of confirmed features to Excel...")
boruta_results_df.to_excel(output_path, index=False)

print(f"✨ Process completed successfully! New dynamic feature selection file saved at:\n 📂 {output_path}")