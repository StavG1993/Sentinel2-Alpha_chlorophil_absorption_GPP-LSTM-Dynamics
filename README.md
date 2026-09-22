# Sentinel2-Alpha_chlorophil_absorption_GPP-LSTM-Dynamics
Evaluating environmental controls on dryland forest GPP: combining Sentinel-2 chlorophyll absorption (α) for GPP estimation with an LSTM-SHAP framework for feature attribution.

**Stage 1: Development and Evaluation of the Remote-Sensing GPP Model
Objective: Calibrate and evaluate a high-resolution remote-sensing GPP model based on Sentinel-2-derived canopy chlorophyll absorption and remotely sensed PAR, validated against continuous Eddy Covariance (EC) measurements from seven dryland conifer forest sites.**

**Methodology & Scripts:**

**1. EC Data & Footprint Processing (foot_print_EC_ES_Hen2)**

Calculated daily 90% flux footprint source areas (based on Kljun et al.) to capture the primary signal.

Applied spatial filtering via QGIS to overlay daily footprints onto defined forest boundaries. Observations extending beyond the forest canopy (e.g., into adjacent roads or bare soil) were entirely excluded.

Files were merged to facilitate streamlined ingestion into Google Earth Engine (GEE).

**2. Sentinel-2 Pre-processing (Sentinel2_bands_global_sites_GEE)**

Extracted Sentinel-2 Level-2A surface reflectance data for each daily footprint.

Masked opaque and cirrus clouds using the QA60 band (bands 10 and 11) and scaled reflectance values.

Implemented a Spectral Angle Mapper (SAM) quality control procedure. Observations falling below a 0.9 similarity threshold against a theoretical vegetation reference spectrum were discarded to remove residual atmospheric artifacts.

**3. BRDF Correction (BRDF_for_spain_greece_GEE)**

Harmonized significant view-angle variability introduced by non-overlapping observations from different orbital swaths for the Spanish and Greek sites (2–3 day temporal resolution).

Applied a localized linear regression separately for each band between a reference 'master' orbit and 'slave' orbits to normalize Bidirectional Reflectance Distribution Function (BRDF) effects.

**4. PAR Extraction and Calibration (ERS5_MODIS_PAR_GEE)**

Integrated spaceborne PAR products from MODIS (MCD18C2) and ERA5-Land, synchronizing them with Sentinel-2 acquisition dates.

Converted values to common units [mol m⁻² day⁻¹] and calibrated the satellite-derived PAR against in-situ EC tower measurements using a linear transfer equation to correct systematic differences.

**5. GPP Calibration & Validation**

Computed chlorophyll absorption at 705 nm (α₇₀₅) from Band 8 (NIR) and Band 5 (Red-edge) using the Kubelka-Munk formulation.

Calculated footprint-level statistical summaries, selecting a 7-day centered moving average for optimal calibration.

Fitted the general GPP estimation model: GPP = a * (α_λ * PAR) + b.

Evaluated the model's accuracy sequentially—first using in-situ PAR, then substituting calibrated satellite-derived PAR—assessing performance across individual and pooled sites via R², RMSE, MAE, CV, NRMSE, and bias.

6. **Independent Model Application** 
   * Applied the calibrated GPP equation using **calibrated satellite-derived PAR** in place of in-situ EC measurements. 
   * This facilitates continuous GPP monitoring over extensive dryland forest areas, entirely independent of flux tower infrastructure.

<br>

### Stage 2: Construction of the Israeli Forest Dataset
**Objective:** Apply the validated remote-sensing GPP model across 70 KKL-JNF Long-Term Monitoring (LTM) pine forest sites in Israel (2018–2024) and compile a comprehensive spatial dataset.

**Methodology & Scripts:**

1. **Spectral Data Processing & Filtering** (`filtering_all_spectral_data`)
   * Extracted Sentinel-2 bands and normalized MODIS PAR for the center of each LTM site.
   * Applied empirically determined thresholds (Band 2 deviation >100%, red-NIR spectral slope <0.1, NDVI drops <−0.015) and a two-step temporal filter to mask severe cloud noise and "zig-zag" artifacts.
   * Calculated the initial satellite-derived GPP (GPPRS) time series (concluding in October 2024).

2. **GPP Gap-Filling & Smoothing** (`gap_filling_GPP`)
   * Replaced missing daily GPP observations by averaging available values from neighboring forest pixels within a **3,000 m radius**.
   * Completed the time series using linear interpolation followed by an **11-day, 2nd-degree Savitzky-Golay filter** to smooth data while retaining essential biological growth dynamics.

3. **LAI Extraction via SL2P** (`LAI_SL2P_GEE`)
   * Estimated Leaf Area Index (LAI) using the **Simplified Level 2 Prototype Processor (SL2P)** model in Google Earth Engine.
   * Restricted observations strictly to the dry season (June–October) to isolate stable canopy structure and minimize the contribution of the seasonal herbaceous understory.

4. **LAI Filtering & Structural Proxy Generation** (`filtering_LAI_only_summer`)
   * Removed outliers using a dynamic filter (1.5 standard deviations from a 5-observation moving mean).
   * Reconstructed short gaps (<30 days) via interpolation and a **21-day Savitzky-Golay filter**. 
   * Averaged the daily values over the dry season to generate a **single, static representative LAI value** per site, acting as a structural proxy for the woody canopy in the LSTM analysis.

5. **Meteorological Data Integration** (`attaching_meteorological_stations.py`)
   * Sourced daily meteorological data from up to three nearby stations, utilizing sequentially expanding radial buffers (up to 15 km) and a strict elevation difference limit (<200 m) to prevent topographical confounding.
   * Calculated specific **Vapor Pressure Deficit (VPD)** metrics (mean, min, and max) and standardized **FAO-56 reference evapotranspiration (ET₀)**.
   * Computed the **Cumulative Climatic Water Balance (CWB₆)**, representing accumulated precipitation minus ET₀ over the preceding six months, to characterize climatic water supply relative to atmospheric demand.
