# Sentinel2-Alpha_chlorophil_absorption_GPP-LSTM-Dynamics
Evaluating environmental controls on dryland forest GPP: combining Sentinel-2 chlorophyll absorption (α) for GPP estimation with an LSTM-SHAP framework for feature attribution.

**Stage 1: Development and Evaluation of the Remote-Sensing GPP Model
Objective: Calibrate and evaluate a high-resolution remote-sensing GPP model based on Sentinel-2-derived canopy chlorophyll absorption and remotely sensed PAR, validated against continuous Eddy Covariance (EC) measurements from seven dryland conifer forest sites.**

**Methodology & Scripts:**

**EC Data & Footprint Processing (foot_print_EC_ES_Hen2)**

Calculated daily 90% flux footprint source areas (based on Kljun et al.) to capture the primary signal.

Applied spatial filtering via QGIS to overlay daily footprints onto defined forest boundaries. Observations extending beyond the forest canopy (e.g., into adjacent roads or bare soil) were entirely excluded.

Files were merged to facilitate streamlined ingestion into Google Earth Engine (GEE).

**Sentinel-2 Pre-processing (Sentinel2_bands_global_sites_GEE)**

Extracted Sentinel-2 Level-2A surface reflectance data for each daily footprint.

Masked opaque and cirrus clouds using the QA60 band (bands 10 and 11) and scaled reflectance values.

Implemented a Spectral Angle Mapper (SAM) quality control procedure. Observations falling below a 0.9 similarity threshold against a theoretical vegetation reference spectrum were discarded to remove residual atmospheric artifacts.

**BRDF Correction (BRDF_for_spain_greece_GEE)**

Harmonized significant view-angle variability introduced by non-overlapping observations from different orbital swaths for the Spanish and Greek sites (2–3 day temporal resolution).

Applied a localized linear regression separately for each band between a reference 'master' orbit and 'slave' orbits to normalize Bidirectional Reflectance Distribution Function (BRDF) effects.

**PAR Extraction and Calibration (ERS5_MODIS_PAR_GEE)**

Integrated spaceborne PAR products from MODIS (MCD18C2) and ERA5-Land, synchronizing them with Sentinel-2 acquisition dates.

Converted values to common units [mol m⁻² day⁻¹] and calibrated the satellite-derived PAR against in-situ EC tower measurements using a linear transfer equation to correct systematic differences.

**GPP Calibration & Validation**

Computed chlorophyll absorption at 705 nm (α₇₀₅) from Band 8 (NIR) and Band 5 (Red-edge) using the Kubelka-Munk formulation.

Calculated footprint-level statistical summaries, selecting a 7-day centered moving average for optimal calibration.

Fitted the general GPP estimation model: GPP = a * (α_λ * PAR) + b.

Evaluated the model's accuracy sequentially—first using in-situ PAR, then substituting calibrated satellite-derived PAR—assessing performance across individual and pooled sites via R², RMSE, MAE, CV, NRMSE, and bias.
