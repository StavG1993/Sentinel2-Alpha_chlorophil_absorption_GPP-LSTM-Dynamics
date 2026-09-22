# Sentinel2-Alpha-LSTM-GPP

## Overview
This Github was built upon Ms. **Stav Gil bar**'s thesis: "_High-Resolution Remote Sensing of Gross Primary Productivity Reveals Climatic and Local Topo-Edaphic Controls in Dryland Conifer Forests_", 2026. Under the Supervision of **Prof. Tarin Paz-Kagan** and **Dr. Yagil Osem**.  Evaluating environmental controls on dryland forest Gross Primary Productivity (GPP): combining a Sentinel-2 chlorophyll absorption (α705) model for GPP estimation with an LSTM-SHAP framework for feature attribution.

## Project Workflow

This repository is structured around the four main stages of the research methodology:

### Stage 1: Development and Evaluation of the Remote-Sensing GPP Model
**Objective:** Calibrate and evaluate a high-spatial-resolution remote-sensing GPP model based on Sentinel-2-derived canopy chlorophyll absorption and remotely sensed PAR, validated against continuous Eddy Covariance (EC) measurements from seven dryland conifer forest sites.

**Methodology & Scripts:**
1. **EC Data & Footprint Processing** (`foot_print_EC_ES_Hen2`)
   * Calculated daily 90% flux footprint source areas (based on Kljun et al.) to capture the primary signal.
   * Applied spatial filtering via QGIS to overlay daily footprints onto defined forest boundaries. Observations extending beyond the forest canopy (e.g., into adjacent roads or bare soil) were entirely excluded.
   * Files were merged to facilitate streamlined ingestion into Google Earth Engine (GEE).
2. **Sentinel-2 Pre-processing** (`Sentinel2_bands_global_sites_GEE`)
   * Extracted Sentinel-2 Level-2A surface reflectance data for each daily footprint.
   * Masked opaque and cirrus clouds using the QA60 band (bands 10 and 11) and scaled reflectance values.
   * Implemented a Spectral Angle Mapper (SAM) quality control procedure. Observations falling below a 0.9 similarity threshold against a theoretical vegetation reference spectrum were discarded to remove residual atmospheric artifacts.
3. **BRDF Correction** (`BRDF_for_spain_greece_GEE`)
   * Harmonized significant view-angle variability introduced by non-overlapping observations from different orbital swaths for the Spanish and Greek sites (2–3 day temporal resolution).
   * Applied a localized linear regression separately for each band between a reference 'master' orbit and 'slave' orbits to normalize Bidirectional Reflectance Distribution Function (BRDF) effects.
4. **PAR Extraction and Calibration** (`ERS5_MODIS_PAR_GEE`)
   * Integrated spaceborne PAR products from MODIS (MCD18C2) and ERA5-Land, synchronizing them with Sentinel-2 acquisition dates.
   * Converted values to common units [mol m⁻² day⁻¹] and calibrated the satellite-derived PAR against in-situ EC tower measurements using a linear transfer equation to correct systematic differences.
5. **GPP Calibration & Validation**
   * Computed chlorophyll absorption at 705 nm (α₇₀₅) from Band 8 (NIR) and Band 5 (Red-edge) using the Kubelka-Munk formulation based on Gitelson et al., 2019, 2021.
   * Calculated footprint-level statistical summaries, selecting a 7-day centered moving average for optimal calibration.
   * Fitted the general GPP estimation model: `GPP = a * (α_λ * PAR) + b`.
   * Evaluated the model's accuracy sequentially—first using in-situ PAR, then substituting calibrated satellite-derived PAR—assessing performance across individual and pooled sites via R², RMSE, MAE, CV, NRMSE, and bias.
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

<br>

### Stage 3: Analysis of Environmental Controls on GPP
**Objective:** Evaluate the association between environmental conditions and spatio-temporal GPP variations using deep learning.

**Methodology & Scripts:**
1. **Feature Selection** (`boruta_algorythm_for_feature_selection`)
   * Applied a two-stage **Boruta feature selection algorithm** using a Random Forest Regressor (n_estimators=100, max_depth=5) to balance efficiency and robustness.
   * Selected features based exclusively on training/validation data. Excluded unconfirmed variables (RH, wind speed) while retaining correlated drivers (Air Temp, VPD, ET₀) due to their distinct eco-physiological mechanisms.
2. **Data Partitioning** 
   * Validated spatial and temporal generalization by applying an **80/20 site-level split** (entire sites assigned exclusively to train/val sets) and utilizing the complete **2024 hydrological year as an independent temporal test set**.
3. **LSTM Architecture & Optimization** (`running_LSTM_final_with_LAI_no_LAI`)
   * Built a **Late-Fusion Attention-LSTM framework** in PyTorch to process continuous daily meteorological sequences alongside static local topo-edaphic and stand characteristics.
   * Conducted hyperparameter optimization via **Optuna (TPE algorithm)** for hidden units, learning rate, and batch size.
   * Implemented regularization strategies including Dropout, weight decay, Huber Loss, Adam optimizer, and an early-stopping mechanism (patience = 50 epochs) to prevent overfitting.
4. **Model Execution** (`running_LSTM_final_with_LAI_no_LAI`)
   * Executed two alternative LSTM models in parallel: one incorporating the dry-season ecosystem LAI as a structural proxy, and one without.

<br>

### Stage 4: Interpretation of Model-Attributed Effects
**Objective:** Open the LSTM "black box" to characterize the relative model-attributed contribution, direction of environmental features, and influence of antecedent conditions.

**Methodology & Scripts:**
1. **Lag Analysis & Performance Metrics** (`running_LSTM_final_with_LAI_no_LAI`)
   * Executed the LSTM across **multiple antecedent temporal lags** (7, 14, 30, 60, 90, 120, 180, and 270 days) to evaluate the time window in which past meteorological conditions influence current daily GPP.
   * Assessed model performance across windows using R², RMSE, MAE, and Bias.
2. **SHAP Extraction** (`running_LSTM_final_with_LAI_no_LAI`)
   * Computed **Shapley Additive exPlanations (SHAP)** in parallel with LSTM predictions to quantify how the fitted model distributes attribution among features (noting that attribution may be shared among correlated features, rather than representing independent causal effects).
3. **SHAP Aggregation & Visualization** 
   * **Outlier Filtering:** Excluded rare, non-representative single-day meteorological extremes (e.g., precipitation > 80 mm d⁻¹, ET₀ > 12.5 mm d⁻¹) prior to aggregation to prevent artificial skewing.
   * **Site-Level Aggregation:** Derived robust overall feature importance by calculating the **median of absolute SHAP values** for each individual site, followed by computing the mean of these site-level medians across the study area.
   * **Directional Impact:** Evaluated directional relationships using linear regression on raw SHAP values, visualized via beeswarm plots for both dynamic daily meteorological variables and static site-level characteristics.
