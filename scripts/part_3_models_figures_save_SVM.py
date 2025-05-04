import os
import random
import concurrent.futures
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (precision_recall_curve, roc_curve, auc, confusion_matrix,
                            accuracy_score, recall_score, precision_score, f1_score,
                            roc_auc_score)


results_df = pd.read_csv("/home/asouza/projects/htvs/3-machine_learning_models/supplementary_results/classification/hyperparameters_models_performance/passed/results_conditions_models_SVM.tsv", sep="\t")

output_dir = '/home/asouza/projects/htvs/3-machine_learning_models/supplementary_results/classification/models_analysis_svm'
os.makedirs(output_dir, exist_ok=True)

def save_model_figure_df_train_test(df, df_filtered, out_path, target_column_categorized, drop_col_class, positive_class='active'):
    import matplotlib
    # Set global figure parameters
    plt.rcParams.update({
        'font.size': 14,                # Set font size to 14
        'axes.grid': False,             # Remove grid from all figures
        'figure.facecolor': 'none',    # Transparent figure background
        'axes.facecolor': 'none',       # Transparent axes background
        'savefig.transparent': True     # Save figures with transparent background
    })

    # Define standardized colors
    TRAIN_COLOR = 'gray'
    TEST_COLOR = 'green'
    BORDER_COLOR = 'black'
    LINE_WIDTH = 2

    main_out_path = os.path.join(out_path, "results_svm_charts")
    os.makedirs(main_out_path, exist_ok=True)

    for _, row in df_filtered.iterrows():
        name = row['name']
        kernel = row['kernel']
        n_splits = row['kf']
        random_state_kf = row['random_state_kf']
        test_size = row['test_size_split']
        random_state_split = row['random_state_split']

        model_dir_name = f"{name}_kernel-{kernel}_kf-{n_splits}_rs-{random_state_kf}_test-{test_size}_split-{random_state_split}"
        model_dir_path = os.path.join(main_out_path, model_dir_name)
        os.makedirs(model_dir_path, exist_ok=True)

        df_target_col = df[target_column_categorized]
        zeros_count = (df.drop(columns=[target_column_categorized]) == 0).sum()
        nan_count = df.drop(columns=[target_column_categorized]).isna().sum()

        cols_to_keep = zeros_count[zeros_count <= 20].index.intersection(
            nan_count[nan_count <= 20].index)

        df_processed = df[cols_to_keep].copy()
        df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
        df_processed = df_processed.drop(columns=[col for col in drop_col_class if col in df_processed.columns], errors="ignore")
        df_processed = pd.concat([df_processed, df_target_col], axis=1)

        if target_column_categorized not in df_processed.columns:
            print(f"Dataset {name} sem a coluna {target_column_categorized}. Pulando.")
            continue
        if len(df_processed[target_column_categorized].unique()) < 2:
            print(f"Dataset {name} tem apenas uma classe. Pulando.")
            continue

        # Keep original data before scaling
        X_original = df_processed.drop(columns=[target_column_categorized]).select_dtypes(include=["number"])
        y = df_processed[target_column_categorized]

        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_original), columns=X_original.columns, index=X_original.index)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state_split, stratify=y)

        # Define other_class before using it
        other_class = [c for c in y_train.unique() if c != positive_class][0]

        max_components = min(X_train.shape[0], X_train.shape[1], 20)
        pca = PCA(n_components=max_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # PCA Score Plot with standardized colors
        # PCA Score Plots Side-by-Side: Left = by Class, Right = by Train/Test
        ACTIVE_COLOR = 'orange'
        INACTIVE_COLOR = 'lightblue'

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7), facecolor='none')

        # Left panel: PCA colored by class (active/inactive)
        X_combined_pca = np.vstack([X_train_pca, X_test_pca])
        y_combined = pd.concat([y_train, y_test])

        class_colors = y_combined.map({positive_class: ACTIVE_COLOR, other_class: INACTIVE_COLOR})
        edge_colors = y_combined.map({positive_class: BORDER_COLOR, other_class: BORDER_COLOR})

        ax_left.scatter(X_combined_pca[:, 0], X_combined_pca[:, 1], s=60, c=class_colors,
                        edgecolors=edge_colors, linewidth=LINE_WIDTH, alpha=1.0)
        ax_left.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
        ax_left.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14)
        ax_left.set_title('PCA: Active vs Inactive', fontsize=14)
        class_legend = [plt.Line2D([0], [0], marker='o', color='w', label='Active',
                                   markerfacecolor=ACTIVE_COLOR, markeredgecolor=BORDER_COLOR, markersize=15),
                        plt.Line2D([0], [0], marker='o', color='w', label='Inactive',
                                   markerfacecolor=INACTIVE_COLOR, markeredgecolor=BORDER_COLOR, markersize=15)]
        ax_left.legend(handles=class_legend, fontsize=14)

        # Right panel: Original PCA plot (Train vs Test)
        ax_right.scatter(X_train_pca[:, 0], X_train_pca[:, 1], s=60, alpha=1.0, label='Train',
                         c=TRAIN_COLOR, edgecolor=BORDER_COLOR, linewidth=LINE_WIDTH)
        ax_right.scatter(X_test_pca[:, 0], X_test_pca[:, 1], s=60, alpha=1.0, label='Test',
                         c=TEST_COLOR, edgecolor=BORDER_COLOR, linewidth=LINE_WIDTH)
        ax_right.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
        ax_right.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14)
        ax_right.set_title('PCA: Train vs Test', fontsize=14)
        class_legend = [plt.Line2D([0], [0], marker='o', color='w', label='Train',
                                   markerfacecolor=TRAIN_COLOR, markeredgecolor=BORDER_COLOR, markersize=15),
                        plt.Line2D([0], [0], marker='o', color='w', label='Test',
                                   markerfacecolor=TEST_COLOR, markeredgecolor=BORDER_COLOR, markersize=15)]        
        ax_right.legend(handles=class_legend, fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(model_dir_path, "pca_scores_dual.svg"), format='svg')
        plt.close()


        # PCA Variance
        plt.figure(figsize=(10, 6), facecolor='none')
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', 
                color='black', markerfacecolor='black', markeredgecolor='black')
        plt.xlabel('Number of Components', fontsize=14)
        plt.ylabel('Cumulative Explained Variance', fontsize=14)
        plt.title('Cumulative Explained Variance by PCA Components', fontsize=14)
        plt.grid(False)
        plt.savefig(os.path.join(model_dir_path, "pca_variance.svg"), format='svg')
        plt.close()

        # Bar plot of active/inactive compounds distribution
        plt.figure(figsize=(10, 6), facecolor='none')

        # Create a dataframe with the counts
        count_data = pd.DataFrame({
            'Train': [
                sum(y_train == other_class),  # Inactive count (train)
                sum(y_train == positive_class)   # Active count (train)
            ],
            'Test': [
                sum(y_test == other_class),    # Inactive count (test)
                sum(y_test == positive_class)    # Active count (test)
            ]
        }, index=['Inactive', 'Active'])

        # Plot the bars with standardized colors
        ax = count_data.T.plot(kind='bar', 
                              color={'Inactive': TRAIN_COLOR, 'Active': TEST_COLOR},
                              edgecolor=BORDER_COLOR,
                              linewidth=1,
                              width=0.8,
                              figsize=(10, 6))

        # Customize the plot
        plt.xlabel('Dataset', fontsize=14)
        plt.ylabel('Number of Compounds', fontsize=14)
        plt.title('Distribution of Active vs Inactive Compounds', fontsize=14)
        plt.xticks(rotation=0, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(title='Class', fontsize=12, title_fontsize=12)
        plt.grid(False)

        # Add value labels on top of each bar
        for p in ax.containers:
            ax.bar_label(p, label_type='edge', fontsize=12, padding=3)

        plt.tight_layout()
        plt.savefig(os.path.join(model_dir_path, "distribution_barplot.svg"), format='svg')
        plt.close()

        # Model training
        model = SVC(kernel=kernel, random_state=random_state_split, probability=True)
        model.fit(X_train, y_train)

        try:
            pos_class_idx = list(model.classes_).index(positive_class)
        except ValueError:
            print(f"Positive class '{positive_class}' not found. Using second class.")
            pos_class_idx = 1

        plt.figure(figsize=(12, 8), facecolor='none')

        for subset, X_set, y_set in [('train', X_train, y_train), ('test', X_test, y_test)]:
            y_pred = model.predict(X_set)
            y_proba = model.predict_proba(X_set)[:, pos_class_idx]

            fpr, tpr, _ = roc_curve(y_set, y_proba, pos_label=positive_class)
            precision, recall, _ = precision_recall_curve(y_set, y_proba, pos_label=positive_class)

            plt.subplot(2, 2, 1)
            plt.plot(fpr, tpr, label=f'{subset} (AUC = {auc(fpr, tpr):.2f})')

            plt.subplot(2, 2, 2)
            plt.plot(recall, precision, label=f'{subset} (AUC = {auc(recall, precision):.2f})')

            plt.subplot(2, 2, 3)
            sns.kdeplot(y_proba[y_set == positive_class], label=f'{subset} {positive_class}')
            sns.kdeplot(y_proba[y_set != positive_class], label=f'{subset} {other_class}')

        plt.subplot(2, 2, 1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=14)
        plt.grid(False)

        plt.subplot(2, 2, 2)
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left", fontsize=14)
        plt.grid(False)

        plt.subplot(2, 2, 3)
        plt.xlabel('Predicted Probability', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Probability Distribution', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(False)

        plt.subplot(2, 2, 4)
        test_probs = model.predict_proba(X_test)[:, pos_class_idx]
        test_preds = model.predict(X_test)

        metrics = {
            'Accuracy': accuracy_score(y_test, test_preds),
            'Precision': precision_score(y_test, test_preds, pos_label=positive_class),
            'Recall': recall_score(y_test, test_preds, pos_label=positive_class),
            'F1': f1_score(y_test, test_preds, pos_label=positive_class),
            'ROC AUC': 1 - roc_auc_score(y_test, test_probs)
        }

        plt.table(cellText=[[f"{v:.2f}" for v in metrics.values()]],
                  rowLabels=['Test'],
                  colLabels=list(metrics.keys()),
                  loc='center')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(model_dir_path, "metrics.svg"), format='svg')
        plt.close()

        # Confusion Matrix with standardized colors and borders
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='none')
        
        # Train confusion matrix (gray)
        cm_train = confusion_matrix(y_train, model.predict(X_train))
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greys', ax=ax1,
                    xticklabels=model.classes_, yticklabels=model.classes_, 
                    cbar=False, linewidths=LINE_WIDTH, linecolor=BORDER_COLOR)
        ax1.set_title('Train Confusion Matrix', fontsize=14)
        ax1.set_xlabel('Predicted', fontsize=14)
        ax1.set_ylabel('True', fontsize=14)

        # Test confusion matrix (green)
        cm_test = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=ax2,
                    xticklabels=model.classes_, yticklabels=model.classes_,
                    cbar=False, linewidths=LINE_WIDTH, linecolor=BORDER_COLOR)
        ax2.set_title('Test Confusion Matrix', fontsize=14)
        ax2.set_xlabel('Predicted', fontsize=14)
        ax2.set_ylabel('True', fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(model_dir_path, "confusion_matrix.svg"), format='svg')
        plt.close()

        # Save confusion matrix
        cm_train_df = pd.DataFrame(cm_train,
                                   index=pd.MultiIndex.from_product([['Train'], model.classes_]),
                                   columns=model.classes_)
        cm_test_df = pd.DataFrame(cm_test,
                                  index=pd.MultiIndex.from_product([['Test'], model.classes_]),
                                  columns=model.classes_)
        cm_combined = pd.concat([cm_train_df, cm_test_df])
        cm_combined.to_csv(os.path.join(model_dir_path, "confusion_matrix.tsv"), sep='\t')

        # Save results
        df_results = pd.DataFrame({
            'index': X_scaled.index,
            'true_label': y,
            'predicted_label': model.predict(X_scaled),
            'probability': model.predict_proba(X_scaled)[:, pos_class_idx],
            'train_test': ['train' if idx in X_train.index else 'test' for idx in X_scaled.index]
        })
        df_results.to_csv(os.path.join(model_dir_path, "results.tsv"), sep='\t', index=False)

        df_modificado = df.copy()
        df_modificado['train_test'] = ['train' if idx in X_train.index else 'test' for idx in df.index]
        df_modificado.to_csv(os.path.join(model_dir_path, "modified_data.tsv"), sep='\t', index=False)

        # Create and save summary.txt with detailed model information
        with open(os.path.join(model_dir_path, "summary.txt"), 'w') as f:
            f.write(f"MODEL SUMMARY\n")
            f.write(f"=============\n\n")
            f.write(f"Model Name: {name}\n")
            f.write(f"Kernel: {kernel}\n")
            f.write(f"Number of K-Fold Splits: {n_splits}\n")
            f.write(f"Random State (KF): {random_state_kf}\n")
            f.write(f"Test Size: {test_size}\n")
            f.write(f"Random State (Split): {random_state_split}\n\n")
            
            # Add detailed model parameters
            f.write(f"MODEL PARAMETERS\n")
            f.write(f"================\n\n")
            f.write(f"Positive class: {positive_class}\n")
            f.write(f"Classes: {model.classes_}\n")
            f.write(f"Class weights: {model.class_weight}\n")
            f.write(f"Intercept: {model.intercept_ if hasattr(model, 'intercept_') else 'Not available'}\n")
            f.write(f"Number of support vectors per class: {model.n_support_}\n")
            f.write(f"Total number of support vectors: {model.support_vectors_.shape[0]}\n")
            f.write(f"Kernel type: {model.kernel}\n")
            f.write(f"Degree (for polynomial kernel): {model.degree}\n")
            f.write(f"Gamma (kernel coefficient): {model._gamma}\n")
            f.write(f"Coef0 (independent term in kernel): {model.coef0}\n")
            f.write(f"Probability estimates: {model.probability}\n")
            f.write(f"Shrinking heuristic: {model.shrinking}\n")
            f.write(f"Tolerance for stopping criterion: {model.tol}\n")
            f.write(f"Cache size: {model.cache_size}\n")
            f.write(f"Maximum iterations: {model.max_iter}\n")
            f.write(f"Decision function shape: {model.decision_function_shape}\n")
            f.write(f"Break ties: {model.break_ties}\n\n")
            
            # Add support vector information
            f.write(f"SUPPORT VECTORS\n")
            f.write(f"===============\n\n")
            f.write(f"Number of support vectors: {len(model.support_)}\n")
            f.write(f"Support vector indices: {model.support_}\n")
            f.write(f"Support vectors shape: {model.support_vectors_.shape}\n\n")
            
            # Add dual coefficients (for linear kernel)
            if kernel == 'linear':
                f.write(f"DUAL COEFFICIENTS (TRANSFORMED DATA - SCALED)\n")
                f.write(f"============================================\n\n")
                if len(model.classes_) == 2:
                    f.write(f"Coefficients for class '{positive_class}':\n")
                    for feature, coef in zip(X_scaled.columns, model.coef_[0]):
                        f.write(f"{feature}: {coef:.6f}\n")
                else:
                    for i, class_name in enumerate(model.classes_):
                        f.write(f"Coefficients for class '{class_name}':\n")
                        for feature, coef in zip(X_scaled.columns, model.coef_[i]):
                            f.write(f"{feature}: {coef:.6f}\n")
                        f.write("\n")
                f.write("\n")
                
                # For linear kernel, show coefficients adjusted for original scale
                f.write(f"DUAL COEFFICIENTS (ORIGINAL DATA - UNSCALED)\n")
                f.write(f"==========================================\n\n")
                if len(model.classes_) == 2:
                    f.write(f"Coefficients for class '{positive_class}':\n")
                    for feature, coef in zip(X_original.columns, model.coef_[0] / scaler.scale_):
                        f.write(f"{feature}: {coef:.6f}\n")
                    f.write(f"\nIntercept (adjusted for original scale): {model.intercept_[0] - np.sum(model.coef_[0] * scaler.mean_ / scaler.scale_):.6f}\n")
                else:
                    for i, class_name in enumerate(model.classes_):
                        f.write(f"Coefficients for class '{class_name}':\n")
                        for feature, coef in zip(X_original.columns, model.coef_[i] / scaler.scale_):
                            f.write(f"{feature}: {coef:.6f}\n")
                        f.write(f"\nIntercept (adjusted for original scale): {model.intercept_[i] - np.sum(model.coef_[i] * scaler.mean_ / scaler.scale_):.6f}\n")
                        f.write("\n")
                f.write("\n")
            
            # Add scaler parameters
            f.write(f"SCALER PARAMETERS\n")
            f.write(f"=================\n\n")
            f.write(f"Mean: {scaler.mean_}\n")
            f.write(f"Scale: {scaler.scale_}\n")
            f.write(f"Number of features seen: {scaler.n_features_in_}\n")
            f.write(f"Feature names: {scaler.feature_names_in_}\n\n")
            
            f.write(f"DATA INFORMATION\n")
            f.write(f"===============\n\n")
            f.write(f"Number of Descriptors: {X_scaled.shape[1]}\n")
            f.write(f"Descriptors Used: {', '.join(X_scaled.columns)}\n\n")
            
            f.write(f"SAMPLE SIZES\n")
            f.write(f"============\n\n")
            f.write(f"Train Set Size: {len(X_train)} compounds\n")
            f.write(f"Test Set Size: {len(X_test)} compounds\n")
            f.write(f"Total Size: {len(X_scaled)} compounds\n\n")
            
            f.write(f"CLASS DISTRIBUTION\n")
            f.write(f"=================\n\n")
            f.write(f"Train Set:\n")
            f.write(f"- Active: {sum(y_train == positive_class)} compounds\n")
            f.write(f"- Inactive: {sum(y_train != positive_class)} compounds\n\n")
            f.write(f"Test Set:\n")
            f.write(f"- Active: {sum(y_test == positive_class)} compounds\n")
            f.write(f"- Inactive: {sum(y_test != positive_class)} compounds\n\n")
            
            f.write(f"PERFORMANCE METRICS\n")
            f.write(f"==================\n\n")
            f.write(f"Train Set:\n")
            f.write(f"- Accuracy: {accuracy_score(y_train, model.predict(X_train)):.4f}\n")
            f.write(f"- Precision: {precision_score(y_train, model.predict(X_train), pos_label=positive_class):.4f}\n")
            f.write(f"- Recall: {recall_score(y_train, model.predict(X_train), pos_label=positive_class):.4f}\n")
            f.write(f"- F1 Score: {f1_score(y_train, model.predict(X_train), pos_label=positive_class):.4f}\n")
            f.write(f"- ROC AUC: {1 - roc_auc_score(y_train, model.predict_proba(X_train)[:, pos_class_idx]):.4f}\n\n")
            
            f.write(f"Test Set:\n")
            f.write(f"- Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}\n")
            f.write(f"- Precision: {precision_score(y_test, model.predict(X_test), pos_label=positive_class):.4f}\n")
            f.write(f"- Recall: {recall_score(y_test, model.predict(X_test), pos_label=positive_class):.4f}\n")
            f.write(f"- F1 Score: {f1_score(y_test, model.predict(X_test), pos_label=positive_class):.4f}\n")
            f.write(f"- ROC AUC: {1 - roc_auc_score(y_test, model.predict_proba(X_test)[:, pos_class_idx]):.4f}\n\n")
            
            f.write(f"CONFUSION MATRIX (Train)\n")
            f.write(f"=======================\n")
            f.write(f"{cm_train}\n\n")
            
            f.write(f"CONFUSION MATRIX (Test)\n")
            f.write(f"======================\n")
            f.write(f"{cm_test}\n")


# Rest of the code remains the same
alpha = 10
thresholds = [round(i / alpha, 1) for i in range(alpha + 1)]

pathway_dir = "/home/asouza/projects/htvs"
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
out_path = f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/models_analysis_svm/"
target_column_categorized = "pIC50_class"
drop_col_class = ['pIC50_class']

# Filter top models (example: top 3 by AUC)
top_models_FRET_fluor_SPR = results_df[results_df.name.str.contains("SPR")].query('train_acc_no_cv > 0.7 and test_acc > 0.7').sort_values(by=["test_roc_auc"], ascending=False).drop_duplicates(subset=['name', 'kernel', 'kf','test_size_split', 'random_state_split'], keep="first").head(10)

top_models_FRET = results_df[results_df.name.str.contains("FRET_c")].query('train_acc_no_cv > 0.7 and test_acc > 0.7').sort_values(by=["test_roc_auc"], ascending=False).drop_duplicates(subset=['name', 'kernel', 'kf','test_size_split', 'random_state_split'], keep="first").head(10)

top_models_fluor = results_df[results_df.name.str.contains("fluor_c")].query('train_acc_no_cv > 0.7 and test_acc > 0.7').sort_values(by=["test_roc_auc"], ascending=False).drop_duplicates(subset=['name', 'kernel', 'kf','test_size_split', 'random_state_split'], keep="first").head(10)

top_models = pd.concat(objs=[top_models_FRET, top_models_fluor, top_models_FRET_fluor_SPR])
top_models.to_csv(out_path+"top_models.tsv", sep="\t", index=False)

# Processar apenas as linhas do top_models, uma por uma
for _, row in top_models.iterrows():
    dataset_name = row['name']
    if dataset_name in datasets:  # Verificar se o dataset existe
        original_df = datasets[dataset_name]
        
        # Criar um DataFrame com apenas a linha atual
        row_df = pd.DataFrame([row])
        
        save_model_figure_df_train_test(
            original_df,
            row_df,  # Passar apenas a linha atual
            out_path,
            target_column_categorized=target_column_categorized,
            drop_col_class=drop_col_class,
            positive_class='active'
        )
    else:
        print(f"Dataset {dataset_name} not found in datasets dictionary.")
