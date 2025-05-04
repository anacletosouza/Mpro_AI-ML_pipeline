from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import concurrent.futures
import random
import os
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress UndefinedMetricWarning
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

pathway_dir = "/home/asouza/projects/htvs"
path = f"{pathway_dir}/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/classification/"

if not os.path.exists(f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/hyperparameters_models_performance"):
    os.makedirs(f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/hyperparameters_models_performance")

def train_model_wrapper(args):
    name, df, n_splits, random_state_kf, test_size_split, random_state_split = args
    return train_model(name, df, n_splits=n_splits, random_state_kf=random_state_kf, 
                      test_size_split=test_size_split, random_state_split=random_state_split)

def generate_tasks():
    tasks = []
    for name, df in datasets.items():
        for i in n_splits_list:
            for j in random_state_kf_list:
                for k in random_state_list:
                    for m in test_size_split_list:
                        for n in random_state_split_list:
                            tasks.append((name, df, i, j, m, n))
    return tasks

def safe_metrics(y_true, y_pred, y_proba, positive_label='active'):
    """Calculate metrics with error handling for undefined cases"""
    metrics = {}
    
    # Accuracy is always defined
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Handle recall, precision, f1 with zero_division parameter
    try:
        metrics['recall'] = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        metrics['precision'] = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    except:
        metrics['recall'] = 0
        metrics['precision'] = 0
        metrics['f1'] = 0
    
    # Handle ROC AUC - requires at least one positive and one negative class
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    except:
        metrics['roc_auc'] = 0.5  # Neutral value when undefined
    
    return metrics

def train_model(name, 
               df, 
               n_splits=5,
               random_state_kf=90, 
               test_size_split=0.3, 
               random_state_split=69,
               threshold_no_cv=0.6, 
               threshold_cv=0.6, 
               threshold_test=0.5,
               min_recall=0.5,
               min_precision=0.5,
               min_f1=0.5,
               min_roc_auc=0.5,
               positive_label='active'):

    if target_column_categorized not in df.columns:
        print(f"Dataset {name} without column {target_column_categorized}. Skipping.")
        return None

    X = df.drop(columns=[target_column_categorized]).select_dtypes(include=["number"]).dropna(axis=1)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)    
    y = df[target_column_categorized]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, 
                                                       random_state=random_state_split, stratify=y)

    # Model without CV
    model_no_cv = GaussianNB()
    model_no_cv.fit(X_train, y_train)
    y_train_pred_no_cv = model_no_cv.predict(X_train)
    y_train_proba_no_cv = model_no_cv.predict_proba(X_train)[:, list(model_no_cv.classes_).index(positive_label)]
    
    train_metrics_no_cv = safe_metrics(y_train, y_train_pred_no_cv, y_train_proba_no_cv, positive_label)

    # K-Fold Cross Validation
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

        model = GaussianNB()
        model.fit(X_fold_train, y_fold_train)

        # Train metrics
        y_fold_train_pred = model.predict(X_fold_train)
        cv_metrics['train_accuracy'].append(accuracy_score(y_fold_train, y_fold_train_pred))

        # Validation metrics
        y_fold_val_pred = model.predict(X_fold_val)
        y_fold_val_proba = model.predict_proba(X_fold_val)[:, list(model.classes_).index(positive_label)]
        val_metrics = safe_metrics(y_fold_val, y_fold_val_pred, y_fold_val_proba, positive_label)
        
        cv_metrics['val_accuracy'].append(val_metrics['accuracy'])
        cv_metrics['val_recall'].append(val_metrics['recall'])
        cv_metrics['val_precision'].append(val_metrics['precision'])
        cv_metrics['val_f1'].append(val_metrics['f1'])
        cv_metrics['val_roc_auc'].append(val_metrics['roc_auc'])

    # Final model evaluation on test set
    model_final = GaussianNB()
    model_final.fit(X_train, y_train)
    y_test_pred = model_final.predict(X_test)
    y_test_proba = model_final.predict_proba(X_test)[:, list(model_final.classes_).index(positive_label)]
    test_metrics = safe_metrics(y_test, y_test_pred, y_test_proba, positive_label)

    # Compile all results
    results = {
        "name": name, 
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

    # Model selection criteria
    if (results["train_acc_no_cv"] >= threshold_no_cv and
        results["mean_val_acc_cv"] >= threshold_cv and
        results["test_acc"] >= threshold_test and
        results["test_recall"] >= min_recall and
        results["test_precision"] >= min_precision and
        results["test_f1"] >= min_f1 and
        results["test_roc_auc"] >= min_roc_auc):
        
        print(f"Selected model: {results}")
        return results
    else:
        return None

# Rest of your code remains the same...


alpha = 10
thresholds = [round(i / alpha, 1) for i in range(alpha + 1)]



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
                print(f"file {filename} not found.")

target_column_categorized = "pIC50_class"
drop_col_class = ['pIC50_class']

n_splits_list = [5, 10]
random_state_kf_list = random.sample(range(101), 10)
random_state_list = random.sample(range(101), 10)
test_size_split_list = [0.20, 0.25, 0.30]
random_state_split_list = random.sample(range(101), 10)
results = []

tasks = generate_tasks()
final_results = []
count = 0

for name, df in datasets.items():
    df_target_col = df[target_column_categorized]
    zeros_count = (df == 0).sum()
    nan_count = df.isna().sum()

    df = df.loc[:, zeros_count <= 20]
    df = df.loc[:, nan_count <= 20]
    df = df.fillna(df.mean(numeric_only=True))

    df = df.drop(columns=[col for col in drop_col_class if col in df.columns], errors="ignore")

    datasets[name] = pd.concat([df, df_target_col], axis=1)


def get_unprocessed_datasets(results, datasets):
    processed = {r["name"] for r in results}
    return {name: df for name, df in datasets.items() if name not in processed}

while datasets:
    tasks = generate_tasks()
    partial_results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=120) as executor:
        for result in executor.map(train_model_wrapper, tasks):
            if result:
                partial_results.append(result)

    final_results.extend(partial_results)
    count = count + 1
    print(f"Iteration {count}")
    
    if count > 5:
        results_df = pd.DataFrame(final_results)
        results_df.to_csv(f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_gaussian_naivebayes.tsv", 
                         sep="\t", index=False)
        print(f"Count: {count}!")
        break

    datasets = get_unprocessed_datasets(final_results, datasets)

results_df = pd.DataFrame(final_results)
results_df.to_csv(f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_gaussian_naivebayes.tsv", 
                 sep="\t", index=False)
