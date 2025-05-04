import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import seaborn as sns
import os
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import concurrent.futures
import random
from sklearn.preprocessing import StandardScaler

def save_model_figure_df_train_test(df, df_filtered, out_path, target_column_categorized, drop_col_class, positive_class='active'):
    """
    Save model, metrics figure, and train/test dataframe for MLPClassifier
    
    Parameters:
    df - Original DataFrame with all data
    df_filtered - Filtered rows from results_df with model parameters to save
    out_path - Path to save files
    target_column_categorized - Name of target column
    drop_col_class - List of columns to drop
    positive_class - Name of positive class (default: 'active')
    """
    
    # Create directory if it doesn't exist
    os.makedirs(out_path, exist_ok=True)
    
    for _, row in df_filtered.iterrows():
        # Extract model parameters
        name = row['name']
        l1 = row['l1']
        l2 = row['l2']
        l3 = row['l3']
        l4 = row['l4']
        n_splits = row['kf']
        random_state_kf = row['random_state_kf']
        test_size = row['test_size_split']
        random_state_split = row['random_state_split']
        
        # Prepare filename
        params_str = f"{name}_layers-{l1}-{l2}-{l3}-{l4}_kf-{n_splits}_rs-{random_state_kf}_test-{test_size}_split-{random_state_split}"
        
        # Apply preprocessing
        df_target_col = df[target_column_categorized]  
        zeros_count = (df.drop(columns=[target_column_categorized]) == 0).sum()  
        nan_count = df.drop(columns=[target_column_categorized]).isna().sum()  
        
        # Filter columns
        cols_to_keep = zeros_count[zeros_count <= 20].index.intersection(
                       nan_count[nan_count <= 20].index)
        
        df_processed = df[cols_to_keep].copy()
        df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
        
        # Remove specified columns
        df_processed = df_processed.drop(columns=[col for col in drop_col_class if col in df_processed.columns], errors="ignore")
        
        # Ensure target column is included
        df_processed = pd.concat([df_processed, df_target_col], axis=1)
        
        # Verify target column exists
        if target_column_categorized not in df_processed.columns:
            print(f"Dataset {name} missing column {target_column_categorized}. Skipping.")
            continue
            
        # Verify we have at least two classes
        if len(df_processed[target_column_categorized].unique()) < 2:
            print(f"Dataset {name} has only one class. Skipping.")
            continue
        
        # Separate features and target
        X = df_processed.drop(columns=[target_column_categorized]).select_dtypes(include=["number"])
        y = df_processed[target_column_categorized]
        
        # Standardize data
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state_split, stratify=y
        )
        
        # Get positive class index
        try:
            pos_class_idx = list(y_train.unique()).index(positive_class)
        except ValueError:
            print(f"Positive class '{positive_class}' not found. Using first class as positive.")
            pos_class_idx = 0

        # Train final model
        model = MLPClassifier(
            hidden_layer_sizes=(l1, l2, l3, l4),
            random_state=random_state_split,
            max_iter=10000
        )
        model.fit(X_train, y_train)
        
        # 1. Save model and scaler
        model_path = os.path.join(out_path, f"model_{params_str}.joblib")
        joblib.dump({'model': model, 'scaler': scaler}, model_path)
               
        # 2. Generate and save metrics figure
        plt.figure(figsize=(12, 8))
        
        # Predictions for train and test
        for subset, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, pos_class_idx]
            
            # Calculate metrics
            fpr, tpr, _ = roc_curve(y, y_proba, pos_label=positive_class)
            roc_auc = auc(fpr, tpr)
            
            precision, recall, _ = precision_recall_curve(y, y_proba, pos_label=positive_class)
            pr_auc = auc(recall, precision)
            
            # Plot ROC
            plt.subplot(2, 2, 1)
            plt.plot(fpr, tpr, label=f'{subset} (AUC = {roc_auc:.2f})')
            
            # Plot Precision-Recall
            plt.subplot(2, 2, 2)
            plt.plot(recall, precision, label=f'{subset} (AUC = {pr_auc:.2f})')
            
            # Plot probability distribution
            plt.subplot(2, 2, 3)
            sns.kdeplot(y_proba[y == positive_class], label=f'{subset} {positive_class}')
            other_class = [c for c in y_train.unique() if c != positive_class][0]
            sns.kdeplot(y_proba[y != positive_class], label=f'{subset} {other_class}')
            
        # Configure plots
        plt.subplot(2, 2, 1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        
        plt.subplot(2, 2, 2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.subplot(2, 2, 3)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Probability Distribution')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        test_probs = model.predict_proba(X_test)[:, pos_class_idx]
        test_preds = model.predict(X_test)
        
        metrics = {
            'Accuracy': accuracy_score(y_test, test_preds),
            'Precision': precision_score(y_test, test_preds, pos_label=positive_class),
            'Recall': recall_score(y_test, test_preds, pos_label=positive_class),
            'F1': f1_score(y_test, test_preds, pos_label=positive_class),
            'ROC AUC': roc_auc_score(y_test, test_probs)
        }
        
        plt.table(cellText=[[f"{v:.2f}" for v in metrics.values()]],
                 rowLabels=['Test'],
                 colLabels=list(metrics.keys()),
                 loc='center')
        plt.axis('off')
        
        plt.tight_layout()
        fig_path = os.path.join(out_path, f"metrics_{params_str}.svg")
        plt.savefig(fig_path, format='svg')
        plt.close()
        
        # 3. Save dataframe with train/test results
        df_results = pd.DataFrame({
            'index': X.index,
            'true_label': y,
            'predicted_label': model.predict(X),
            'probability': model.predict_proba(X)[:, pos_class_idx],
            'train_test': ['train' if idx in X_train.index else 'test' for idx in X.index]
        })
        
        df_path = os.path.join(out_path, f"df_results_{params_str}.tsv")
        df_results.to_csv(df_path, sep='\t', index=False)


alpha = 10
thresholds = [round(i / alpha, 1) for i in range(alpha + 1)]

path = f"{pathway_dir}/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/classification/"

categories = ["FRET_class", "fluor_class", "FRET_fluor_SPR_class"]
dimensions = ["2D", "3D"]

datasets = {}

for category in categories:
    for dim in dimensions:
        for threshold in thresholds:
            filename = f"df_{category}_{dim}_threshold_{threshold}_class.tsv"
            file_path = os.path.join(path, filename)
            
            if os.path.exists(file_path):  
                datasets[f"{category}_{dim}_threshold_{threshold}"] = pd.read_csv(file_path, sep="\t")
            else:
                print(f"Aviso: file {filename} was not found.")

target_column_categorized = "pIC50_class"
drop_col_class = ['pIC50_class']


# Example usage:
pathway_dir = "/home/asouza/projects/htvs"
out_path = f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/models_analysis_mlp/"
target_column_categorized = "pIC50_class"
drop_col_class = ['pIC50_class']

# Filter top models (example: top 3 by AUC)
top_models_FRET_fluor_SPR = results_df[results_df.name.str.contains("SPR")].query('acc_nonkfold_cv_train > 0.8 and accuracy_test > 0.7').sort_values(by=["accuracy_test"], ascending=False).drop_duplicates(subset=['name', 'l1', 'l2', 'l3', 'l4', 'kf', 'test_size_split', 'random_state_split'], keep="first").sort_values(by=['roc_auc_test'], ascending=False).head(10)

top_models_FRET = results_df[results_df.name.str.contains("FRET_c")].query('acc_nonkfold_cv_train > 0.8 and accuracy_test > 0.7').sort_values(by=["accuracy_test"], ascending=False).drop_duplicates(subset=['name', 'l1', 'l2', 'l3', 'l4', 'kf', 'test_size_split', 'random_state_split'], keep="first").sort_values(by=['roc_auc_test'], ascending=False).head(10)

top_models_fluor = results_df[results_df.name.str.contains("fluor_c")].query('acc_nonkfold_cv_train > 0.8 and accuracy_test > 0.7').sort_values(by=["accuracy_test"], ascending=False).drop_duplicates(subset=['name', 'l1', 'l2', 'l3', 'l4', 'kf', 'test_size_split', 'random_state_split'], keep="first").sort_values(by=['roc_auc_test'], ascending=False).head(10)

top_models = pd.concat([top_models_FRET, top_models_fluor, top_models_FRET_fluor_SPR])

# Process each model individually
for _, row in top_models.iterrows():
    dataset_name = row['name']
    if dataset_name in datasets:  # Check if dataset exists
        original_df = datasets[dataset_name]
        
        # Create a DataFrame with just the current row
        row_df = pd.DataFrame([row])
        
        save_model_figure_df_train_test(
            original_df,
            row_df,  # Pass only the current row
            out_path,
            target_column_categorized=target_column_categorized,
            drop_col_class=drop_col_class,
            positive_class='active'
        )
    else:
        print(f"Dataset {dataset_name} not found in datasets dictionary.")
