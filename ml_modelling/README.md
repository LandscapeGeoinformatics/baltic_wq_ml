# Code supplement

- 01_explore_target_data.ipynb: Calculates the statistics of the input water quality data.
- 02_covariate_set_reduction.ipynb: Performs covariate reduction based on within-group and target correlations.
- 03_baseline_and_top5_models.ipynb: Trains baseline models using the covariate sets obtained from 02_covariate_set_reduction.ipynb, determines the top 5 most important covariates, and trains new models with those top 5 covariates. Assesses the prediction accuracy, calculates and plots SHAP values and partial dependence.
- 04_spatially_adjusted_models.ipynb: Trains coordinate, buffer, and top 5 buffer models. Assesses the prediction accuracy, calculates and plots SHAP values and partial dependence.
- 05_residuals_analysis.ipynb: Calculates the statistics and Moran's I of the residuals of all models.
- model_rf.py: Functions for Random Forest model training.
- plot_xai.py: Functions for plotting SHAP values and partial dependence.