

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# --- 1. הגדרות נתיבים ---
PATH_GPP = r"D:\Stav\research\Israel_polygons\final_excels\Final_Cleaned_alphaREPAR_Data_8.6.26.csv"
PATH_SHP = r"D:\Stav\research\Israel_polygons\polygons\LTM_rock.shp"
BASE_DIR = r"D:\Stav\research\Israel_polygons\gap_filling_GPP_8.6.26"

BUFFER_DIST = 3000  # רדיוס למילוי מרחבי
SG_WINDOW = 11      # חלון החלקה
SG_POLY = 2        # דרגת פולינום

if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
    os.makedirs(os.path.join(BASE_DIR, "Plots"))

# --- 2. טעינה ויצירת קשר בין המרחב לנתונים (ID-to-Site_ID mapping) ---# --- 2. טעינה ויצירת קשר בין המרחב לנתונים (ID-to-Site_ID mapping) ---
gdf = gpd.read_file(PATH_SHP)

# תיקון אוטומטי לשם עמודת ה-ID:
# נבדוק אם יש עמודה שנקראת ID בצורות שונות, ואם לא - ניצור אותה מהאינדקס
if 'ID' in gdf.columns:
    pass
elif 'id' in gdf.columns:
    gdf = gdf.rename(columns={'id': 'ID'})
elif 'Id' in gdf.columns:
    gdf = gdf.rename(columns={'Id': 'ID'})
elif 'FID' in gdf.columns:
    gdf = gdf.rename(columns={'FID': 'ID'})
else:
    # מוצא אחרון: אם אין שום עמודת מפתח, ניצור אחת רצה על בסיס האינדקס
    print("Warning: No ID column found in SHP. Creating a new ID column based on index.")
    gdf['ID'] = gdf.index

# יצירת ה-Site_ID הזמני בשכבה המרחבית כדי להתאים ל-CSV הקיים
gdf['temporary_site_id'] = gdf['FORNAME'].astype(str).str.strip() + "_" + gdf['REP'].astype(str).str.strip()

# וידוא שרק עמודת ה-ID ועמודת המיפוי נשמרות לצורך החיבור
mapping_df = gdf[['ID', 'temporary_site_id']].drop_duplicates()

# טעינת קובץ ה-CSV המקורי וחיבור עמודת ה-ID המרחבית אליו
df_gpp = pd.read_csv(PATH_GPP)
df_gpp['date'] = pd.to_datetime(df_gpp['date'], dayfirst=True)
df_gpp = df_gpp.dropna(subset=['GPP', 'Site_ID']).drop_duplicates(subset=['Site_ID', 'date'])

# מיזוג כדי להוסיף את עמודת ID המרחבית לקובץ הלווייני
df_gpp = df_gpp.merge(mapping_df, left_on='Site_ID', right_on='temporary_site_id', how='inner')
df_gpp = df_gpp.drop(columns=['temporary_site_id'])

# יצירת מילון (Dictionary) שישמש אותנו בסוף לתרגום חזרה מ-ID לשם האתר המקורי מה-CSV
id_to_name_dict = df_gpp.set_index('ID')['Site_ID'].to_dict()

# הכנת ה-KDTree על בסיס ה-ID המרחבי
gdf = gdf.set_index('ID')
coords = np.array(list(gdf.geometry.centroid.apply(lambda p: (p.x, p.y))))
tree = cKDTree(coords)

# --- 3. פונקציית זיהוי אנומליות ---
def remove_outliers(df_site, threshold=2.5):
    if len(df_site) < 5: return df_site
    rolling_median = df_site['GPP'].rolling(window=5, center=True).median().bfill().ffill()
    rolling_std = df_site['GPP'].rolling(window=5, center=True).std().bfill().ffill()
    
    is_not_outlier = (df_site['GPP'] <= rolling_median + threshold * rolling_std) & \
                     (df_site['GPP'] >= rolling_median - threshold * rolling_std)
    return df_site[is_not_outlier]

# --- 4. פונקציית עיבוד לכל אתר (מבוססת ID) ---
def process_site_data(site_id):
    # א) שליפת נתונים ומיון לפי ה-ID החדש
    site_raw = df_gpp[df_gpp['ID'] == site_id].copy().sort_values('date')
    if site_raw.empty: return pd.DataFrame()

    # ב) הורדת אנומליות
    site_clean = remove_outliers(site_raw)
    
    # ג) הכנת ציר זמן יומי
    st, en = site_clean['date'].min(), site_clean['date'].max()
    full_df = pd.DataFrame({'date': pd.date_range(st, en, freq='D'), 'ID': site_id})
    
    # ד) שמירת הנתונים המקוריים ששרדו
    full_df = full_df.merge(site_clean[['date', 'GPP']], on='date', how='left')
    full_df = full_df.rename(columns={'GPP': 'GPP_original'})
    
    # ה) מילוי מרחבי (Spatial Fill) מבוסס מרחקי ה-KDTree מהפוליגון
    neighbor_indices = tree.query_ball_point(coords[gdf.index.get_loc(site_id)], BUFFER_DIST)
    neighbor_ids = [gdf.index[n] for n in neighbor_indices if gdf.index[n] != site_id]
    
    full_df['GPP_processed'] = full_df['GPP_original']
    mask_missing = full_df['GPP_processed'].isna()
    
    if mask_missing.any() and neighbor_ids:
        neighbors_pool = df_gpp[df_gpp['ID'].isin(neighbor_ids)]
        spatial_lookup = neighbors_pool.groupby('date')['GPP'].mean()
        full_df.loc[mask_missing, 'GPP_processed'] = full_df.loc[mask_missing, 'date'].map(spatial_lookup)

    # ו) מילוי ליניארי והחלקה (SG)
    full_df['GPP_processed'] = full_df['GPP_processed'].interpolate(method='linear', limit_direction='both')
    if len(full_df) >= SG_WINDOW:
        full_df['GPP_processed'] = savgol_filter(full_df['GPP_processed'], window_length=SG_WINDOW, polyorder=SG_POLY)

    return full_df

# --- 5. הרצה, החלפת שמות חזרה ושמירה ---
all_results = []
metrics = []

# רצים לפי ה-ID הייחודיים הקיימים בנתוני ה-GPP
for site_id in tqdm(df_gpp['ID'].unique()):
    res = process_site_data(site_id)
    if res.empty: continue

    # חישוב RMSE
    valid = res.dropna(subset=['GPP_original'])
    if len(valid) > 0:
        rmse = np.sqrt(mean_squared_error(valid['GPP_original'], valid['GPP_processed']))
    else:
        rmse = np.nan
        
    # החזרת שם האתר המקורי (Site_ID) מתוך המילון עבור קובץ התוצאות והגרפים
    original_site_name = id_to_name_dict[site_id]
    
    res['Site_ID'] = original_site_name
    res['RMSE_value'] = rmse
    all_results.append(res)
    
    metrics.append({'ID': site_id, 'Site_ID': original_site_name, 'RMSE': rmse})

    # הפקת פלוט מותאם עם השם המקורי
    plt.figure(figsize=(12, 5))
    plt.scatter(res['date'], res['GPP_original'], color='black', s=20, label='Cleaned Satellite Data')
    plt.plot(res['date'], res['GPP_processed'], color='green', linewidth=2, label=f'Daily GPP (RMSE: {rmse:.3f})')
    plt.title(f"Site: {original_site_name} (ID: {site_id}) | Pre-cleaned Anomalies & Processed")
    plt.legend()
    plt.savefig(os.path.join(BASE_DIR, "Plots", f"{original_site_name}.png"))
    plt.close()

# איחוד ושמירה סופית
final_df = pd.concat(all_results)
# סידור עמודות קצת יותר נקי לקובץ הסופי
cols_order = ['date', 'ID', 'Site_ID', 'GPP_original', 'GPP_processed', 'RMSE_value']
final_df = final_df[[c for c in cols_order if c in final_df.columns]]

final_df.to_csv(os.path.join(BASE_DIR, "Final_GPP_Comparison.csv"), index=False)
pd.DataFrame(metrics).to_csv(os.path.join(BASE_DIR, "RMSE_Summary.csv"), index=False)