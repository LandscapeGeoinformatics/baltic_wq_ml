import pandas as pd

import shap
from sklearn.inspection import PartialDependenceDisplay

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np

# Set plot color
cmap = sns.diverging_palette(220, 20, as_cmap=True)
plot_tone = cmap(0.0)

## XAI

def calculate_shap(rfReg, X_train):
    
    shap_values = shap.TreeExplainer(rfReg).shap_values(X_train)
    
    return shap_values


def plot_avg_shap(shap_values, X_train, directory, figure_name):
    
    if figure_name != 'no output':

        # Number of covariates
        n_features = X_train.shape[1]

    # Fixed width, scalable height
        width = 8
        height_per_feature = 0.4
        height = n_features * height_per_feature


        # Create figure
        fig, ax = plt.subplots(figsize=(width, height))

        # Summary plot - average impact (bar plot)
        shap.summary_plot(
        shap_values=shap_values,
        features=X_train,
        feature_names=X_train.columns,
        plot_type="bar",
        color='black',
        show=False
    )

    # Adjust font sizes (matplotlib manages axes differently from SHAP)
        plt.gca().tick_params(labelsize=17)
        plt.xlabel("mean SHAP value", fontsize=17)

        plt.tight_layout()


        plt.savefig(f'{directory}\\{figure_name}.pdf', dpi=500)
            
        plt.show()
        
    else:

        print('No figure created')
    

def plot_beeswarm_shap(shap_values, X_train, directory, param, model, figure_name):

    if figure_name != 'no output':
        
        n_features = X_train.shape[1]
        fig_width = 8
        fig_height = 0.4 * n_features

        # Create figure
        fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout = False)
        ax0 = fig.add_subplot(111)

        # beeswarm plot - direction of influence
        shap.summary_plot(shap_values=shap_values, features=X_train, feature_names=X_train.columns, cmap = cmap, show=False)
        
        plt.gca().tick_params(labelsize=17)
        plt.xlabel("SHAP value", fontsize=17)

        fig = plt.gcf()
        for ax in fig.axes:
            if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == 'Feature value':
                ax.tick_params(labelsize=17)  # increase font size of color bar ticks
                ax.set_ylabel('Feature value', fontsize=17)  # increase color bar label


        long_name = select_full_model_name(model)

        plt.title(f"{param.upper()} {long_name}", fontsize=20, pad=20, fontweight='bold')

        plt.tight_layout()
        
        plt.savefig(f'{directory}\\{figure_name}.pdf', dpi=500)
    
        plt.show()
    
    else:

        print('No figure created')



def plot_pdps(rfReg, X_train, directory, figure_name, height_per_plot=1, plot_tone='black'):
    
    if figure_name != 'no output':

        num_features = len(X_train.columns)

        fig_height = num_features * height_per_plot + 0.5

        fig, ax = plt.subplots(figsize=(13, fig_height))

        disp = PartialDependenceDisplay.from_estimator(
            rfReg, X_train, X_train.columns, ax=ax, line_kw={'color':plot_tone}, random_state=1)


        axes = ax if isinstance(ax, (list, np.ndarray)) else [ax]

        for row in disp.axes_:
            for axis in row:
                if axis is not None:
                    axis.set_xlabel(axis.get_xlabel(), fontsize=14)
                    axis.set_ylabel(axis.get_ylabel(), fontsize=14)
                    axis.tick_params(labelsize=14)

        fig.subplots_adjust(hspace=0.4) 

        plt.tight_layout()

        plt.savefig(f'{directory}\\{figure_name}.pdf', dpi=500)

        plt.show()

    else:

        print('No figure created')


def retrieve_top(X_train, shap_values, top, model_name, output_folder):
    
    importance_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": abs(shap_values).mean(axis=0)
    })

    # Get the top 10 most important variables
    top_10_features = importance_df.sort_values("importance", ascending=False).head(top)
    
    top_10_features.to_csv(f'{output_folder}\\top_{top}_{model_name}.csv')
    
    return top_10_features


def select_full_model_name(short_name):

    mapping = {
        "baseline": "Baseline",
        "top5": "Top 5",
        "coordinate": "Coordinate",
        "buffer": "Buffer",
        "top5_buffer": "Top 5 buffer",
    }
    return mapping.get(short_name, short_name) 