# Neural Dynamics of Face Pareidolia: Spatiotemporal Transformation and Recurrent Ambiguity Resolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete preprocessing, representational similarity analysis (RSA), representational connectivity analysis (RCA), and behavioral correlation pipelines used to investigate the spatiotemporal neural dynamics of **face pareidolia** from high-density electroencephalography (EEG) data.

---

## Repository Structure & Pipeline Overview

The analysis pipeline is structured into sequential processing stages, from raw data cleaning to advanced connectivity modeling and behavioral correlation:

### 1. Preprocessing & Data Structuring
* **`0_extracting_triggers_preprocessing_saving_data_matrix_chan_posits_errExplnd_ctch_svd.py`**  
  Preprocesses raw Biosemi BDF recordings, decodes complex hardware trigger sequences (isolating standard RSVP trials and catch trials), applies band-pass/notch filtering and common average referencing across 64 scalp channels, and aligns continuous data with behavioral logs into structured participant `.npy` matrices.

### 2. Behavioral & Psychometric Analysis
* **`1_Plotting_behavioural Results.py`**  
  Parses subjective behavioral "faceness" ratings across the 21-level object-to-face morph continuum, calculating grand-mean psychometric functions, standard errors (SEM), and instantaneous rates of change ($\Delta\text{Faceness} / \Delta\text{Morph}$).

### 3. Representational Similarity Analysis (RSA) & Model Fitting
* **`2_neural_psychm_functions_fldr_3fits_topos_activity_svd_2.py`**  
  Fits candidate computational psychometric models (**Linear**, **Sigmoid**, and **Step**) to single-channel EEG data across time windows, evaluating adjusted coefficients of determination ($R_{\text{adj}}^2$) to identify winning representational strategies.
* **`3-Model-based_RSA_topos_folder_based_CV_only_RDMS_plotting.py`**  
  Visualises theoretical representational dissimilarity matrices (RDMs) and psychometric profiles across the 21 morph levels for all candidate hypotheses.
* **`3_Highlighting_electrodes.py`**  
  Generates region-of-interest (ROI) topographical maps to delineate anterior (`Fp`, `AF`, `F`, `FC`) and posterior (`CP`, `P`, `PO`, `O`) electrode clusters.

### 4. Transformation-Based Representational Connectivity Analysis (RCA)
* **`3-Model-based_RCA_trnsfrm_model_based_transformation_CV_scalar_all_stats_Granger_params_svd.py`**  
  Implements the core **Transformation-Based RCA** framework. Incorporates history-stripping (Granger-causality controls) to track directed information flow and computational transformation between posterior linear feature tracking and anterior sigmoidal categorization. Computes group-level Bayes Factors ($BF_{10}$) via JZS priors.
* **`4-Model-based_RCA_trnsfrm_model_based_transformation_CV_scalar_all_stats_Granger_params_ambiguities_svd.py`**  
  Evaluates how categorical uncertainty modulates information flow by partitioning morph levels into subsets (**Object-like**, **Ambiguous Pareidolic**, and **Face-like**), testing whether ambiguous stimuli selectively recruit prolonged feedback.

### 5. Generalisation, Animacy Priors & Behavioral Correlation
* **`5_time_resolved_decodings_distances_from_boundary_blk_excld_elec_interp_21_morph_levels_svd.py`**  
  Trains linear support vector classifiers (LinearSVC) on natural object/face images and projects AI-generated morphs onto the decision boundary over time.
* **`6_neural_psychm_functions_fldr_3fits_fld_vars_Anim_Inanam_stats_svd.py`**  
  Tests whether semantic animacy priors structurally alter category boundaries by comparing fitted Sigmoid threshold ($x_0$) and slope ($k$) parameters between animate and inanimate object bases.
* **`7-Neural_behav_correlation_activity_based_SubjectWise_distance_based_svd.py`**  
  Assesses behavioral relevance by correlating individual participant neural sigmoid slopes with reaction times on probe trials.

---

## Requirements & Dependencies

The analyses require a standard scientific Python environment with the following core libraries:
* `numpy`, `scipy`, `pandas`
* `mne` (v1.0+)
* `scikit-learn`
* `matplotlib`, `seaborn`

---

## Usage Instructions

1. Place raw participant folders (`P01`, `P02`, etc.) containing BDF recordings and behavioral Excel sheets into your designated `BASE_DIR`.
2. Run script **`0`** to generate processed subject arrays (`Processed_Outputs/`).
3. Execute the modeling, RSA, and RCA scripts sequentially to replicate the publication figures and statistical outputs.

---

## License

This project is licensed under the terms of the MIT License.

