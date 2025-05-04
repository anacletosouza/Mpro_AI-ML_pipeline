import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import seaborn as sns
import os
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

def save_model_figure_df_train_test(df, df_filtered, out_path, target_column_categorized, drop_col_class, positive_class='active'):
    """
    Salva o modelo, figura com métricas e dataframe com informações de train/test
    Adaptado para GaussianNB
    
    Parâmetros:
    df - DataFrame original com todos os dados
    df_filtered - Linhas filtradas do results_df com os parâmetros do modelo a ser salvo
    out_path - Caminho para salvar os arquivos
    target_column_categorized - Nome da coluna target
    drop_col_class - Lista de colunas a serem removidas
    positive_class - Nome da classe positiva (default: 'active')
    """
    
    # Criar diretório se não existir
    os.makedirs(out_path, exist_ok=True)
    
    for _, row in df_filtered.iterrows():
        # Extrair parâmetros do modelo
        name = row['name']
        n_splits = row['kf']
        random_state_kf = row['random_state_kf']
        test_size = row['test_size_split']
        random_state_split = row['random_state_split']
        
        # Preparar nome do arquivo
        params_str = f"{name}_kf-{n_splits}_rs-{random_state_kf}_test-{test_size}_split-{random_state_split}"
        
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
        
        # Separar features e target
        X = df_processed.drop(columns=[target_column_categorized]).select_dtypes(include=["number"])
        y = df_processed[target_column_categorized]
        
        # Padronizar os dados
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        # Split train-test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state_split, stratify=y
        )
        
        # Treinar modelo final
        model = GaussianNB()
        model.fit(X_train, y_train)
        
        # Obter índice da classe positiva
        try:
            pos_class_idx = list(model.classes_).index(positive_class)
        except ValueError:
            print(f"Positive class '{positive_class}' not found. Using second class.")
            pos_class_idx = 1

        # 1. Salvar o modelo e o scaler
        model_path = os.path.join(out_path, f"model_{params_str}.joblib")
        joblib.dump({'model': model, 'scaler': scaler}, model_path)
               
        # 2. Gerar e salvar figura com métricas
        plt.figure(figsize=(12, 8))
        
        # Previsões para train e test
        for subset, X, y in [('train', X_train, y_train), ('test', X_test, y_test)]:
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, pos_class_idx]
            
            # Calcular métricas
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
            
            # Plot distribuição das probabilidades
            plt.subplot(2, 2, 3)
            sns.kdeplot(y_proba[y == positive_class], label=f'{subset} {positive_class}')
            other_class = [c for c in model.classes_ if c != positive_class][0]
            sns.kdeplot(y_proba[y != positive_class], label=f'{subset} {other_class}')
            
        # Configurar plots
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
        
        # 3. Salvar dataframe com train/test
        df_results = pd.DataFrame({
            'index': X.index,
            'true_label': y,
            'predicted_label': model.predict(X),
            'probability': model.predict_proba(X)[:, pos_class_idx],
            'train_test': ['train' if idx in X_train.index else 'test' for idx in X.index]
        })
        
        df_path = os.path.join(out_path, f"df_results_{params_str}.tsv")
        df_results.to_csv(df_path, sep='\t', index=False)

# Configurações iniciais
pathway_dir = "/home/asouza/projects/htvs"
out_path = f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/models_analysis_gaussiannb/"
target_column_categorized = "pIC50_class"
drop_col_class = ['pIC50_class']

# Carregar datasets
alpha = 10
thresholds = [round(i / alpha, 1) for i in range(alpha + 1)]
categories = ["FRET_class", "fluor_class", "FRET_fluor_SPR_class"]
dimensions = ["2D", "3D"]

datasets = {}
path = f"{pathway_dir}/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/classification/"

for category in categories:
    for dim in dimensions:
        for threshold in thresholds:
            filename = f"df_{category}_{dim}_threshold_{threshold}_class.tsv"
            file_path = os.path.join(path, filename)
            
            if os.path.exists(file_path):  
                datasets[f"{category}_{dim}_threshold_{threshold}"] = pd.read_csv(file_path, sep='\t')
            else:
                print(f"Aviso: file {filename} was not found.")

# Processar os modelos
# Filtrar os modelos desejados (exemplo: top 3 por AUC)
top_models_FRET_fluor_SPR = results_df[results_df.name.str.contains("SPR")].query('train_acc_no_cv > 0.6 and test_acc > 0.5').sort_values(by=["test_roc_auc"], ascending=False).drop_duplicates(subset=['name',  'kf','test_size_split', 'random_state_split'], keep="first").head(10)

top_models_FRET = results_df[results_df.name.str.contains("FRET_c")].query('train_acc_no_cv > 0.6 and test_acc > 0.5').sort_values(by=["test_roc_auc"], ascending=False).drop_duplicates(subset=['name',  'kf','test_size_split', 'random_state_split'], keep="first").head(10)

top_models_fluor = results_df[results_df.name.str.contains("fluor_c")].query('train_acc_no_cv > 0.6 and test_acc > 0.5').sort_values(by=["test_roc_auc"], ascending=False).drop_duplicates(subset=['name',  'kf','test_size_split', 'random_state_split'], keep="first").head(10)

top_models = pd.concat(objs=[top_models_FRET, top_models_fluor, top_models_FRET_fluor_SPR])

# Processar cada modelo individualmente
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


