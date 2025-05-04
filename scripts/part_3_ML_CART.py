from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import os
import concurrent.futures
import random
from sklearn.preprocessing import StandardScaler

pathway_dir = "/home/asouza/projects/htvs"
path = "{pathway_dir}/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/"

if not os.path.exists(f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/hyperparameters_models_performance"):
    os.makedirs(f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/hyperparameters_models_performance")

def train_model_wrapper(args):
    name, df, n_splits, random_state_kf, test_size_split, random_state_split, criterion, min_samples_split, min_samples_leaf = args
    return train_model(
        name, df, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
        n_splits=n_splits, random_state_kf=random_state_kf, test_size_split=test_size_split, random_state_split=random_state_split
    )

def generate_tasks():
    tasks = []
    for name, df in datasets.items():
        for i in n_splits_list:
            for j in random_state_kf_list:
                for k in random_state_list:
                    for m in test_size_split_list:
                        for n in random_state_split_list:
                            for criterion in criterion_list:
                                for min_samples_split in min_samples_split_list:
                                    for min_samples_leaf in min_samples_leaf_list:
                                        tasks.append((name, df, i, j, m, n, criterion, min_samples_split, min_samples_leaf))
    return tasks


def train_model(name, df, max_depth=None, criterion="gini", min_samples_split=2, min_samples_leaf=1, n_splits=5, 
                random_state_kf=90, test_size_split=0.3, random_state_split=69, 
                threshold_no_cv=0.6, threshold_cv=0.6, threshold_test=0.5,
                min_precision=0.5, min_recall=0.5, min_f1=0.5, min_auc=0.5):

    if target_column_categorized not in df.columns:
        print(f"Dataset {name} has not column {target_column_categorized}. Jumpping.")
        return None

    X = df.drop(columns=[target_column_categorized]).select_dtypes(include=["number"]).dropna(axis=1)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)    
    y = df[target_column_categorized]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=random_state_split, stratify=y)

    # Training without CV
    model_no_cv = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, 
                                         min_samples_leaf=min_samples_leaf, random_state=random_state_split)
    model_no_cv.fit(X_train, y_train)
    
    # Metrics without CV
    y_pred_train_no_cv = model_no_cv.predict(X_train)
    y_proba_train_no_cv = model_no_cv.predict_proba(X_train)[:, 1] if hasattr(model_no_cv, "predict_proba") else None
    
    train_metrics_no_cv = {
        'accuracy': accuracy_score(y_train, y_pred_train_no_cv),
        'recall': recall_score(y_train, y_pred_train_no_cv, pos_label="active", zero_division=0),
        'precision': precision_score(y_train, y_pred_train_no_cv, pos_label="active", zero_division=0),
        'f1': f1_score(y_train, y_pred_train_no_cv, pos_label="active", zero_division=0),
        'roc_auc': roc_auc_score((y_train == "active").astype(int), y_proba_train_no_cv) if y_proba_train_no_cv is not None else 0.0
    }

    # Training with CV
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

        model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, 
                                     min_samples_split=min_samples_split, 
                                     min_samples_leaf=min_samples_leaf, 
                                     random_state=random_state_kf)
        model.fit(X_fold_train, y_fold_train)
        
        # Train metrics for this fold
        y_pred_fold_train = model.predict(X_fold_train)
        cv_metrics['train_accuracy'].append(accuracy_score(y_fold_train, y_pred_fold_train))
        
        # Validation metrics for this fold
        y_pred_fold_val = model.predict(X_fold_val)
        y_proba_fold_val = model.predict_proba(X_fold_val)[:, 1] if hasattr(model, "predict_proba") else None
        
        cv_metrics['val_accuracy'].append(accuracy_score(y_fold_val, y_pred_fold_val))
        cv_metrics['val_recall'].append(recall_score(y_fold_val, y_pred_fold_val, pos_label="active", zero_division=0))
        cv_metrics['val_precision'].append(precision_score(y_fold_val, y_pred_fold_val, pos_label="active", zero_division=0))
        cv_metrics['val_f1'].append(f1_score(y_fold_val, y_pred_fold_val, pos_label="active", zero_division=0))
        cv_metrics['val_roc_auc'].append(roc_auc_score((y_fold_val == "active").astype(int), y_proba_fold_val) if y_proba_fold_val is not None else 0.0)

    # Final model training and test evaluation
    model_final = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, 
                                       min_samples_split=min_samples_split, 
                                       min_samples_leaf=min_samples_leaf, 
                                       random_state=random_state_split)
    model_final.fit(X_train, y_train)
    
    # Test metrics
    y_pred_test = model_final.predict(X_test)
    y_proba_test = model_final.predict_proba(X_test)[:, 1] if hasattr(model_final, "predict_proba") else None
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test, pos_label="active", zero_division=0),
        'precision': precision_score(y_test, y_pred_test, pos_label="active", zero_division=0),
        'f1': f1_score(y_test, y_pred_test, pos_label="active", zero_division=0),
        'roc_auc': roc_auc_score((y_test == "active").astype(int), y_proba_test) if y_proba_test is not None else 0.0
    }

    # Compile all results
    results = {
        "name": name, 
        "max_depth": max_depth,
        "criterion": criterion,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
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

    if all([
        results["train_acc_no_cv"] >= threshold_no_cv,
        results["mean_val_acc_cv"] >= threshold_cv,
        results["test_acc"] >= threshold_test,
        results["test_precision"] >= min_precision,
        results["test_recall"] >= min_recall,
        results["test_f1"] >= min_f1,
        results["test_roc_auc"] >= min_auc
    ]):
        print(results)
        return results
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

n_splits_list, criterion_list, min_samples_split_list, min_samples_leaf_list = [5, 10], ["gini", "entropy", "log_loss"], range(2, 10), range(1, 10)
random_state_kf_list, random_state_list = random.sample(range(101), 1), random.sample(range(101), 1)
test_size_split_list, random_state_split_list = [0.20, 0.25, 0.30], random.sample(range(101), 1)
final_results, count = [], 0


for name, df in datasets.items():
    df_target_col = df[target_column_categorized]  
    zeros_count = (df == 0).sum()  
    nan_count = df.isna().sum()  
    
    df = df.loc[:, zeros_count <= 20]
    df = df.loc[:, nan_count <= 20]
    df = df.fillna(df.mean(numeric_only=True))
    
    df = df.drop(columns=[col for col in drop_col_class if col in df.columns], errors="ignore")
    
    datasets[name] = pd.concat([df, df_target_col], axis=1)


while datasets:
    tasks = generate_tasks()
    with concurrent.futures.ProcessPoolExecutor(max_workers=250) as executor:
        final_results.extend(filter(None, executor.map(train_model_wrapper, tasks)))
    datasets = {name: df for name, df in datasets.items() if name not in {res["name"] for res in final_results}}
    count += 1
    print(f"Interations: {count}")
    
    if count > 2:
        results_df = pd.DataFrame(final_results)
        results_df.to_csv(f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_decision_tree_classifier.tsv", sep="\t", index=False)
        print(f"interation completed: {count} interactions")
        break

results_df_decision_tree_models = pd.DataFrame(final_results)
print(f"Number of iterations: {count}")
results_df_decision_tree_models.to_csv(f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_decision_tree_classifier.tsv", sep="\t", index=False)
