from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
import concurrent.futures
import random
import os

pathway_dir = "/home/asouza/projects/htvs"
path = f"{pathway_dir}/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/"

def generate_tasks():
    tasks = []
    for name, df in datasets.items():
        for i in n_splits_list:
            for j in random_state_kf_list:
                for k in random_state_list:
                    for m in test_size_split_list:
                        for kernel in kernel_list:
                            tasks.append((name, df, i, j, m, k, kernel))
    return tasks

def train_model(args):
    name, df, n_splits, random_state_kf, test_size_split, random_state_split, kernel = args

    if target_column_categorized not in df.columns:
        print(f"Dataset {name} without column {target_column_categorized}. Jumpping.")
        return None

    X = df.drop(columns=[target_column_categorized]).select_dtypes(include=["number"]).dropna(axis=1)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    y = df[target_column_categorized]

    # Verificar se temos pelo menos duas classes
    if len(y.unique()) < 2:
        print(f"Dataset {name} has only one class. Skipping.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_split, random_state=random_state_split, stratify=y
    )

    # Metrics calculation without CV
    model_no_cv = SVC(kernel=kernel, random_state=random_state_split, probability=True)
    model_no_cv.fit(X_train, y_train)
    
    y_train_pred_no_cv = model_no_cv.predict(X_train)
    y_train_proba_no_cv = model_no_cv.predict_proba(X_train)
    
    # Get the positive class (assuming binary classification)
    positive_class = y_train.unique()[1] if y_train.unique()[0] == 'inactive' else y_train.unique()[0]
    
    train_metrics_no_cv = {
        'accuracy': accuracy_score(y_train, y_train_pred_no_cv),
        'recall': recall_score(y_train, y_train_pred_no_cv, pos_label=positive_class),
        'precision': precision_score(y_train, y_train_pred_no_cv, pos_label=positive_class),
        'f1': f1_score(y_train, y_train_pred_no_cv, pos_label=positive_class),
        'roc_auc': roc_auc_score(y_train, y_train_proba_no_cv[:, 1] if positive_class == y_train.unique()[1] else y_train_proba_no_cv[:, 0])
    }

    # Metrics calculation with CV
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state_kf)
    
    cv_metrics = {
        'train_accuracy': [],
        'val_accuracy': [],
        'val_recall': [],
        'val_precision': [],
        'val_f1': [],
        'val_roc_auc': []
    }

    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = SVC(kernel=kernel, random_state=random_state_kf, probability=True)
        model.fit(X_fold_train, y_fold_train)

        # Train metrics
        y_fold_train_pred = model.predict(X_fold_train)
        cv_metrics['train_accuracy'].append(accuracy_score(y_fold_train, y_fold_train_pred))

        # Validation metrics
        y_fold_val_pred = model.predict(X_fold_val)
        y_fold_val_proba = model.predict_proba(X_fold_val)
        
        cv_metrics['val_accuracy'].append(accuracy_score(y_fold_val, y_fold_val_pred))
        cv_metrics['val_recall'].append(recall_score(y_fold_val, y_fold_val_pred, pos_label=positive_class))
        cv_metrics['val_precision'].append(precision_score(y_fold_val, y_fold_val_pred, pos_label=positive_class))
        cv_metrics['val_f1'].append(f1_score(y_fold_val, y_fold_val_pred, pos_label=positive_class))
        cv_metrics['val_roc_auc'].append(roc_auc_score(y_fold_val, y_fold_val_proba[:, 1] if positive_class == y_train.unique()[1] else y_fold_val_proba[:, 0]))

    # Final model and test metrics
    model_final = SVC(kernel=kernel, random_state=random_state_split, probability=True)
    model_final.fit(X_train, y_train)
    
    y_test_pred = model_final.predict(X_test)
    y_test_proba = model_final.predict_proba(X_test)
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred, pos_label=positive_class),
        'precision': precision_score(y_test, y_test_pred, pos_label=positive_class),
        'f1': f1_score(y_test, y_test_pred, pos_label=positive_class),
        'roc_auc': roc_auc_score(y_test, y_test_proba[:, 1] if positive_class == y_train.unique()[1] else y_test_proba[:, 0])
    }

    # Compile all results
    results = {
        "name": name, 
        "kernel": kernel,
        "kf": n_splits,
        "random_state_kf": random_state_kf,
        "test_size_split": test_size_split,
        "random_state_split": random_state_split,
        
        # No CV metrics
        "train_acc_no_cv": round(train_metrics_no_cv['accuracy'], 2),
        "train_recall_no_cv": round(train_metrics_no_cv['recall'], 2),
        "train_precision_no_cv": round(train_metrics_no_cv['precision'], 2),
        "train_f1_no_cv": round(train_metrics_no_cv['f1'], 2),
        "train_roc_auc_no_cv": round(train_metrics_no_cv['roc_auc'], 2),
        
        # CV metrics (mean across folds)
        "mean_train_acc_cv": round(np.mean(cv_metrics['train_accuracy']), 2),
        "mean_val_acc_cv": round(np.mean(cv_metrics['val_accuracy']), 2),
        "mean_val_recall_cv": round(np.mean(cv_metrics['val_recall']), 2),
        "mean_val_precision_cv": round(np.mean(cv_metrics['val_precision']), 2),
        "mean_val_f1_cv": round(np.mean(cv_metrics['val_f1']), 2),
        "mean_val_roc_auc_cv": round(np.mean(cv_metrics['val_roc_auc']), 2),
        
        # Test metrics
        "test_acc": round(test_metrics['accuracy'], 2),
        "test_recall": round(test_metrics['recall'], 2),
        "test_precision": round(test_metrics['precision'], 2),
        "test_f1": round(test_metrics['f1'], 2),
        "test_roc_auc": round(test_metrics['roc_auc'], 2)
    }
    
    if (
        round(train_metrics_no_cv['accuracy'], 2) >= threshold_no_cv
        and round(np.mean(cv_metrics['val_accuracy']), 2) >= threshold_cv
        and round(test_metrics['accuracy'], 2) >= threshold_test
    ):
        print(results)
        return results
    else:
        return None


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

n_splits_list = [5, 10]
random_state_kf_list = random.sample(range(101), 1)
random_state_list = random.sample(range(101), 1)
test_size_split_list = [0.20, 0.25, 0.30]
kernel_list = ["rbf", "linear", "poly",  "sigmoid"]
threshold_no_cv = 0.60
threshold_cv = 0.60
threshold_test = 0.50
final_results = []

for name, df in datasets.items():
    df_target_col = df[target_column_categorized]  
    zeros_count = (df == 0).sum()  
    nan_count = df.isna().sum()  
    
    df = df.loc[:, zeros_count <= 20]
    df = df.loc[:, nan_count <= 20]
    df = df.fillna(df.mean(numeric_only=True))
    
    df = df.drop(columns=[col for col in drop_col_class if col in df.columns], errors="ignore")
    
    datasets[name] = pd.concat([df, df_target_col], axis=1)

count = 0
while datasets:
    tasks = generate_tasks()
    with concurrent.futures.ProcessPoolExecutor(max_workers=250) as executor:  
        results = list(filter(None, executor.map(train_model, tasks)))
        final_results.extend(results)
    datasets = {name: df for name, df in datasets.items() if name not in {res["name"] for res in results}}
    count += 1
    if count > 2:
        results_df = pd.DataFrame(final_results)
        results_df.to_csv(f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_SVM.tsv", sep="\t", index=False)
        print(f"Count: {count}")
        break

results_df = pd.DataFrame(final_results)
results_df.to_csv(f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_SVM.tsv", sep="\t", index=False)
