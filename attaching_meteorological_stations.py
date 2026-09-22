#%%
import pandas as pd
import geopandas as gpd
import os

# --- File path definitions ---
path_filtered_stations = r"D:\Stav\research\Israel_polygons\meteorological_data\filtered_meteorological_stations.xls"
path_polygons = r"D:\Stav\research\Israel_polygons\polygons\LTM_rock.shp"
output_folder = r"D:\Stav\research\Israel_polygons\meteorological_data"
output_filename = "polygons_meteorological_summary_v5_elevation_optimized.xlsx"

def run_meteorological_analysis():
    print("--- Starting updated processing pipeline (elevation optimization) ---")
    
    # 1. Load and process station data from the climate sheet
    print("Loading filtered station data...")
    df_stations = pd.read_excel(path_filtered_stations, sheet_name='stns aclim תחנות אקלימיות')
    df_stations.columns = df_stations.columns.str.strip()
    
    # Column mapping
    col_map = {
        'station_name': [c for c in df_stations.columns if 'Hebrew station name' in c or 'שם התחנה בעברית' in c][0],
        'x': [c for c in df_stations.columns if 'ITM East' in c or 'מזרח' in c][0],
        'y': [c for c in df_stations.columns if 'ITM North' in c or 'צפון' in c][0],
        'altitude': [c for c in df_stations.columns if 'Altitude' in c or 'גובה' in c][0],
        'source': [c for c in df_stations.columns if 'מקור' in c or 'source' in c.lower()][0]
    }
    
    stations_clean = df_stations.rename(columns={
        col_map['station_name']: 'station_name',
        col_map['x']: 'x',
        col_map['y']: 'y',
        col_map['altitude']: 'altitude',
        col_map['source']: 'source_file'
    }).copy()
    
    stations_clean['altitude'] = pd.to_numeric(stations_clean['altitude'], errors='coerce')
    
    # Create GeoDataFrame
    stations_gdf = gpd.GeoDataFrame(
        stations_clean, 
        geometry=gpd.points_from_xy(stations_clean.x, stations_clean.y),
        crs="EPSG:2039"
    )
    
    # 2. Load polygons
    print("Loading Shapefile...")
    polygons = gpd.read_file(path_polygons)
    polygons = polygons.to_crs("EPSG:2039")
    polygons['FORNAME_REP'] = polygons['FORNAME'].astype(str) + "_" + polygons['REP'].astype(str)
    
    # 3. Associate stations within dynamic elevation-conditioned radius
    results_list = []
    print("Performing spatial calculations...")
    
    for idx, poly_row in polygons.iterrows():
        poly_geom = poly_row.geometry
        poly_name = poly_row['FORNAME_REP']
        poly_alt = poly_row['Altitude']
        
        # Define radius thresholds as requested
        radius_steps = [5500, 10000, 15000]
        final_stations_for_poly = None
        
        for radius in radius_steps:
            distances = stations_gdf.distance(poly_geom)
            nearby_mask = distances <= radius
            
            if nearby_mask.any():
                stations_found = stations_gdf[nearby_mask].copy()
                avg_stn_alt = stations_found['altitude'].mean()
                avg_diff = avg_stn_alt - poly_alt
                
                # Stop conditions: 
                # 1. If absolute elevation difference is <= 200 meters
                # 2. Or if maximum radius threshold is reached (15 km)
                if abs(avg_diff) <= 200 or radius == 15000:
                    stations_found['distance_to_poly_m'] = distances[nearby_mask]
                    stations_found['polygon_name_full'] = poly_name
                    stations_found['polygon_altitude'] = poly_alt
                    stations_found['avg_stations_altitude'] = avg_stn_alt
                    stations_found['avg_height_difference'] = avg_diff
                    stations_found['station_vs_poly_diff'] = stations_found['altitude'] - poly_alt
                    stations_found['final_search_radius'] = radius
                    stations_found['height_warning_gt_200m'] = abs(avg_diff) > 200
                    
                    final_stations_for_poly = stations_found
                    break # Sufficient stations found or maximum range reached
                else:
                    # Elevation difference exceeds 200m, proceed to next iteration (larger radius)
                    continue
            else:
                # No stations found within current radius, proceed to next radius
                continue
        
        if final_stations_for_poly is not None:
            results_list.append(final_stations_for_poly)

    # 4. Merge and save results
    if results_list:
        final_df = pd.concat(results_list, ignore_index=True)
        if 'geometry' in final_df.columns:
            final_df = final_df.drop(columns='geometry')
            
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        full_path = os.path.join(output_folder, output_filename)
        final_df.to_excel(full_path, index=False)
        print(f"--- Processing completed! File saved at: {full_path} ---")
    else:
        print("Error: No data found to process.")

if __name__ == "__main__":
    run_meteorological_analysis()