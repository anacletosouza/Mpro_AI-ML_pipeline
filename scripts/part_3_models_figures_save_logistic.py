import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
import seaborn as sns
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

output_dir = '/home/asouza/projects/htvs/3-machine_learning_models/supplementary_results/classification/models_analysis_logistic_reg'
os.makedirs(output_dir, exist_ok=True)

results_df = pd.read_csv(
    "/home/asouza/projects/htvs/3-machine_learning_models/"
    "supplementary_results/classification/hyperparameters_models_performance/"
    "passed/results_conditions_models_LogisticRegression.tsv",
    sep="\t"
)

def save_model_figure_df_train_test(df, df_filtered, out_path, target_column_categorized, drop_col_class, positive_class='active'):
    """
    Salva o modelo, figura com métricas e dataframe com informações de train/test
    Adaptado para Regressão Logística
    
    Parâmetros:
    df - DataFrame original com todos os dados
    df_filtered - Linhas filtradas do results_df com os parâmetros do modelo a ser salvo
    out_path - Caminho para salvar os arquivos
    target_column_categorized - Nome da coluna target
    drop_col_class - Lista de colunas a serem removidas
    positive_class - Nome da classe positiva (default: 'active')
    """
    
    # Configurações visuais padrão
    plt.rcParams.update({
        'font.size': 14,
        'axes.grid': False,
        'figure.facecolor': 'none',
        'axes.facecolor': 'none',
        'savefig.transparent': True,
        'figure.max_open_warning': 100
    })
    
    # Cores padronizadas
    TRAIN_COLOR = 'gray'
    TEST_COLOR = 'green'
    BORDER_COLOR = 'black'
    LINE_WIDTH = 2
    
    # Criar diretório principal se não existir
    main_out_path = os.path.join(out_path, "results_logreg_charts")
    os.makedirs(main_out_path, exist_ok=True)
    
    for _, row in df_filtered.iterrows():
        # Extrair parâmetros do modelo
        name = row['name']
        solver = row['solver']
        n_splits = row['kf']
        random_state_kf = row['random_state_kf']
        test_size = row['test_size_split']
        random_state_split = row['random_state_split']
        
        # Criar diretório específico para este modelo
        model_dir_name = f"{name}_solver-{solver}_kf-{n_splits}_rs-{random_state_kf}_test-{test_size}_split-{random_state_split}"
        model_dir_path = os.path.join(main_out_path, model_dir_name)
        os.makedirs(model_dir_path, exist_ok=True)
        
        # Aplicar pré-processamento
        df_target_col = df[target_column_categorized]
        zeros_count = (df.drop(columns=[target_column_categorized]) == 0).sum()
        nan_count = df.drop(columns=[target_column_categorized]).isna().sum()
        
        # Filtrar colunas
        cols_to_keep = zeros_count[zeros_count <= 20].index.intersection(
                       nan_count[nan_count <= 20].index)
        
        df_processed = df[cols_to_keep].copy()
        df_processed = df_processed.fillna(df_processed.mean(numeric_only=True))
        
        # Remover colunas especificadas
        df_processed = df_processed.drop(columns=[col for col in drop_col_class if col in df_processed.columns], errors="ignore")
        
        # Garantir que a coluna target está incluída
        df_processed = pd.concat([df_processed, df_target_col], axis=1)
        
        # Verificações
        if target_column_categorized not in df_processed.columns:
            print(f"Dataset {name} sem a coluna {target_column_categorized}. Pulando.")
            continue
            
        if len(df_processed[target_column_categorized].unique()) < 2:
            print(f"Dataset {name} tem apenas uma classe. Pulando.")
            continue
        
        # Separar features e target (dados originais não transformados)
        X_original = df_processed.drop(columns=[target_column_categorized]).select_dtypes(include=["number"])
        y = df_processed[target_column_categorized]
        
        # Padronizar os dados (dados transformados)
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_original), columns=X_original.columns, index=X_original.index)
        
        # Split train-test (usando dados transformados)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state_split, stratify=y
        )
        
        # PCA para visualização
        max_components = min(X_train.shape[0], X_train.shape[1], 20)
        pca = PCA(n_components=max_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        # Verificar se temos pelo menos 2 componentes para plotar
        n_components_to_plot = min(2, max_components)
        
        # PCA Score Plot com cores padronizadas
        ACTIVE_COLOR = 'orange'
        INACTIVE_COLOR = 'lightblue'
        
        fig1, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7), facecolor='none')
        
        # Painel esquerdo: PCA colorido por classe (active/inactive)
        X_combined_pca = np.vstack([X_train_pca, X_test_pca])
        y_combined = pd.concat([y_train, y_test])
        
        other_class = [c for c in y_combined.unique() if c != positive_class][0]
        class_colors = y_combined.map({positive_class: ACTIVE_COLOR, other_class: INACTIVE_COLOR})
        edge_colors = y_combined.map({positive_class: BORDER_COLOR, other_class: BORDER_COLOR})
        
        if n_components_to_plot >= 2:
            ax_left.scatter(X_combined_pca[:, 0], X_combined_pca[:, 1], s=60, c=class_colors,
                           edgecolors=edge_colors, linewidth=LINE_WIDTH, alpha=1.0)
            ax_left.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
            ax_left.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14)
            ax_left.set_title('PCA: Active vs Inactive', fontsize=14)
        else:
            ax_left.scatter(X_combined_pca[:, 0], np.zeros(len(X_combined_pca)), s=60, c=class_colors,
                           edgecolors=edge_colors, linewidth=LINE_WIDTH, alpha=1.0)
            ax_left.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
            ax_left.set_title('PCA (1 component only)', fontsize=14)
        
        class_legend = [plt.Line2D([0], [0], marker='o', color='w', label='Active',
                                  markerfacecolor=ACTIVE_COLOR, markeredgecolor=BORDER_COLOR, markersize=15),
                       plt.Line2D([0], [0], marker='o', color='w', label='Inactive',
                                  markerfacecolor=INACTIVE_COLOR, markeredgecolor=BORDER_COLOR, markersize=15)]
        ax_left.legend(handles=class_legend, fontsize=14)
        
        # Painel direito: PCA original (Train vs Test)
        if n_components_to_plot >= 2:
            ax_right.scatter(X_train_pca[:, 0], X_train_pca[:, 1], s=60, alpha=1.0, label='Train',
                            c=TRAIN_COLOR, edgecolor=BORDER_COLOR, linewidth=LINE_WIDTH)
            ax_right.scatter(X_test_pca[:, 0], X_test_pca[:, 1], s=60, alpha=1.0, label='Test',
                            c=TEST_COLOR, edgecolor=BORDER_COLOR, linewidth=LINE_WIDTH)
            ax_right.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
            ax_right.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=14)
        else:
            ax_right.scatter(X_train_pca[:, 0], np.zeros(len(X_train_pca)), s=60, alpha=1.0, label='Train',
                            c=TRAIN_COLOR, edgecolor=BORDER_COLOR, linewidth=LINE_WIDTH)
            ax_right.scatter(X_test_pca[:, 0], np.zeros(len(X_test_pca)), s=60, alpha=1.0, label='Test',
                            c=TEST_COLOR, edgecolor=BORDER_COLOR, linewidth=LINE_WIDTH)
            ax_right.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=14)
        
        ax_right.set_title('PCA: Train vs Test', fontsize=14)
        class_legend = [plt.Line2D([0], [0], marker='o', color='w', label='Train',
                       markerfacecolor=TRAIN_COLOR, markeredgecolor=BORDER_COLOR, markersize=15),
                       plt.Line2D([0], [0], marker='o', color='w', label='Test',
                       markerfacecolor=TEST_COLOR, markeredgecolor=BORDER_COLOR, markersize=15)]        
        ax_right.legend(handles=class_legend, fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir_path, "pca_scores_dual.svg"), format='svg')
        plt.close(fig1)
        
        # PCA Variance
        fig2 = plt.figure(figsize=(10, 6), facecolor='none')
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', 
                color='black', markerfacecolor='black', markeredgecolor='black')
        plt.xlabel('Number of Components', fontsize=14)
        plt.ylabel('Cumulative Explained Variance', fontsize=14)
        plt.title('Cumulative Explained Variance by PCA Components', fontsize=14)
        plt.grid(False)
        plt.savefig(os.path.join(model_dir_path, "pca_variance.svg"), format='svg')
        plt.close(fig2)
        
        # Bar plot of active/inactive compounds distribution
        fig3 = plt.figure(figsize=(10, 6), facecolor='none')
        
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
        plt.close(fig3)
        
        # Treinar modelo final com dados transformados
        model = LogisticRegression(
            solver=solver,
            random_state=random_state_split,
            max_iter=1000
        )
        model.fit(X_train, y_train)
        
        # Obter índice da classe positiva
        try:
            pos_class_idx = list(model.classes_).index(positive_class)
        except ValueError:
            print(f"Positive class '{positive_class}' not found. Using second class.")
            pos_class_idx = 1
        
        # Salvar o modelo e o scaler
        joblib.dump({'model': model, 'scaler': scaler}, os.path.join(model_dir_path, "model.joblib"))
        
        # Gerar figura com métricas
        fig4 = plt.figure(figsize=(12, 8), facecolor='none')
        
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
        plt.close(fig4)
        
        # Matriz de confusão com cores padronizadas
        fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor='none')
        
        # Matriz de confusão para treino (cinza)
        cm_train = confusion_matrix(y_train, model.predict(X_train))
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greys', ax=ax1,
                   xticklabels=model.classes_, yticklabels=model.classes_, 
                   cbar=False, linewidths=LINE_WIDTH, linecolor=BORDER_COLOR)
        ax1.set_title('Train Confusion Matrix', fontsize=14)
        ax1.set_xlabel('Predicted', fontsize=14)
        ax1.set_ylabel('True', fontsize=14)
        
        # Matriz de confusão para teste (verde)
        cm_test = confusion_matrix(y_test, model.predict(X_test))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=ax2,
                   xticklabels=model.classes_, yticklabels=model.classes_,
                   cbar=False, linewidths=LINE_WIDTH, linecolor=BORDER_COLOR)
        ax2.set_title('Test Confusion Matrix', fontsize=14)
        ax2.set_xlabel('Predicted', fontsize=14)
        ax2.set_ylabel('True', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir_path, "confusion_matrix.svg"), format='svg')
        plt.close(fig5)
        
        # Salvar matriz de confusão
        cm_train_df = pd.DataFrame(cm_train,
                                 index=pd.MultiIndex.from_product([['Train'], model.classes_]),
                                 columns=model.classes_)
        cm_test_df = pd.DataFrame(cm_test,
                                index=pd.MultiIndex.from_product([['Test'], model.classes_]),
                                columns=model.classes_)
        cm_combined = pd.concat([cm_train_df, cm_test_df])
        cm_combined.to_csv(os.path.join(model_dir_path, "confusion_matrix.tsv"), sep='\t')
        
        # Salvar resultados
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
        
        # Criar e salvar arquivo summary.txt com informações detalhadas
        with open(os.path.join(model_dir_path, "summary.txt"), 'w') as f:
            f.write(f"MODEL SUMMARY\n")
            f.write(f"=============\n\n")
            f.write(f"Model Name: {name}\n")
            f.write(f"Solver: {solver}\n")
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
            f.write(f"Intercept: {model.intercept_}\n")
            f.write(f"Maximum iterations: {model.max_iter}\n")
            f.write(f"Actual iterations: {model.n_iter_[0] if hasattr(model, 'n_iter_') else 'Not available'}\n")
            f.write(f"Penalty: {model.penalty}\n")
            f.write(f"Tolerance: {model.tol}\n")
            f.write(f"Multi-class: {model.multi_class}\n")
            f.write(f"Warm start: {model.warm_start}\n")
            f.write(f"Fit intercept: {model.fit_intercept}\n")
            f.write(f"Dual formulation: {model.dual}\n")
            f.write(f"Regularization strength (C): {model.C}\n")
            f.write(f"Intercept scaling: {model.intercept_scaling}\n")
            f.write(f"Verbose: {model.verbose}\n\n")
            
            # Add coefficients for transformed data (scaled)
            f.write(f"MODEL COEFFICIENTS (TRANSFORMED DATA - SCALED)\n")
            f.write(f"==============================================\n\n")
            if len(model.classes_) == 2:
                # Binary classification - single set of coefficients
                f.write(f"Coefficients for class '{positive_class}':\n")
                for feature, coef in zip(X_scaled.columns, model.coef_[0]):
                    f.write(f"{feature}: {coef:.6f}\n")
            else:
                # Multiclass - coefficients for each class
                for i, class_name in enumerate(model.classes_):
                    f.write(f"Coefficients for class '{class_name}':\n")
                    for feature, coef in zip(X_scaled.columns, model.coef_[i]):
                        f.write(f"{feature}: {coef:.6f}\n")
                    f.write("\n")
            f.write("\n")
            
            # Add coefficients for original (untransformed) data
            # We need to adjust coefficients for the original scale
            f.write(f"MODEL COEFFICIENTS (ORIGINAL DATA - UNSCALED)\n")
            f.write(f"============================================\n\n")
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

# Configurações iniciais
alpha = 10
thresholds = [round(i / alpha, 1) for i in range(alpha + 1)]

pathway_dir = "/home/asouza/projects/htvs"
path = f"{pathway_dir}/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/classification/"
out_path = f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/models_analysis_logistic_reg/"

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

# Filtrar os melhores modelos
top_models_FRET_fluor_SPR = results_df[results_df.name.str.contains("SPR")].query('train_acc_no_cv > 0.7 and test_acc > 0.7').sort_values(by=["test_roc_auc"], ascending=False).drop_duplicates(subset=['name', 'solver', 'kf','test_size_split', 'random_state_split'], keep="first").head(10)

top_models_FRET = results_df[results_df.name.str.contains("FRET_c")].query('train_acc_no_cv > 0.7 and test_acc > 0.7').sort_values(by=["test_roc_auc"], ascending=False).drop_duplicates(subset=['name', 'solver', 'kf','test_size_split', 'random_state_split'], keep="first").head(10)

top_models_fluor = results_df[results_df.name.str.contains("fluor_c")].query('train_acc_no_cv > 0.7 and test_acc > 0.7').sort_values(by=["test_roc_auc"], ascending=False).drop_duplicates(subset=['name', 'solver', 'kf','test_size_split', 'random_state_split'], keep="first").head(10)

top_models = pd.concat([top_models_FRET, top_models_fluor, top_models_FRET_fluor_SPR])
top_models.to_csv(out_path+"top_models.tsv", sep="\t", index=False)

# Processar cada modelo individualmente
for _, row in top_models.iterrows():
    dataset_name = row['name']
    if dataset_name in datasets:
        # Criar um DataFrame com apenas a linha atual
        row_df = pd.DataFrame([row])
        
        save_model_figure_df_train_test(
            datasets[dataset_name],
            row_df,  # Passar apenas a linha atual
            out_path,
            target_column_categorized=target_column_categorized,
            drop_col_class=drop_col_class,
            positive_class='active'
        )
    else:
        print(f"Dataset {dataset_name} not found in datasets dictionary.")
