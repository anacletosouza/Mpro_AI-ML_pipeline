from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import concurrent.futures
import random
import os
from sklearn.preprocessing import StandardScaler

pathway_dir = "/home/asouza/projects/htvs"
path = f"{pathway_dir}/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/classification/"


def train_model_wrapper(args):
    name, df, n_splits, random_state_kf, test_size_split, random_state_split, l1, l2, l3, l4 = args
    return train_model(
        name, df, l1=l1, l2=l2, l3=l3, l4=l4, n_splits=n_splits,
        random_state_kf=random_state_kf, test_size_split=test_size_split, random_state_split=random_state_split
    )

def generate_tasks():
    tasks = []
    for name, df in datasets.items():
        for i in n_splits_list:
            for j in random_state_kf_list:
                for k in random_state_list:
                    for m in test_size_split_list:
                        for n in random_state_split_list:
                            for l1 in l1_list:
                                for l2 in l2_list:
                                    for l3 in l3_list:
                                        for l4 in l4_list:
                                            tasks.append((name, df, i, j, m, n, l1, l2, l3, l4))
    return tasks

def train_model(
    name, df, l1=3, l2=2, l3=1, l4=3, n_splits=5, random_state_kf=90,
    test_size_split=0.3, random_state_split=69, 
    threshold_no_cv=0.6, threshold_cv=0.6, threshold_test=0.5,
    min_recall=0.5, min_precision=0.5, min_f1=0.5, min_roc_auc=0.5
):
    if target_column_categorized not in df.columns:
        print(f"Dataset {name} has no column {target_column_categorized}. Skipping.")
        return None

    X = df.drop(columns=[target_column_categorized]).select_dtypes(include=["number"]).dropna(axis=1)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    y = df[target_column_categorized]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_split, random_state=random_state_split, stratify=y
    )

    # Determine which class should be considered positive (assuming 'active' is the positive class)
    positive_class = 'active' if 'active' in y.unique() else y.unique()[0]

    model_no_cv = MLPClassifier(
        hidden_layer_sizes=(l1, l2, l3, l4), random_state=random_state_split, max_iter=10000
    )
    model_no_cv.fit(X_train, y_train)
    y_train_pred = model_no_cv.predict(X_train)
    train_accuracy_no_cv = accuracy_score(y_train, y_train_pred)
    train_recall_no_cv = recall_score(y_train, y_train_pred, pos_label=positive_class)
    train_precision_no_cv = precision_score(y_train, y_train_pred, pos_label=positive_class)
    train_f1_no_cv = f1_score(y_train, y_train_pred, pos_label=positive_class)
    train_roc_auc_no_cv = roc_auc_score(y_train, model_no_cv.predict_proba(X_train)[:, list(model_no_cv.classes_).index(positive_class)])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state_kf)
    train_accuracies, val_accuracies = [], []
    train_recalls, val_recalls = [], []
    train_precisions, val_precisions = [], []
    train_f1s, val_f1s = [], []
    train_roc_aucs, val_roc_aucs = [], []

    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = MLPClassifier(hidden_layer_sizes=(l1, l2, l3, l4), random_state=random_state_kf, max_iter=10000)
        model.fit(X_fold_train, y_fold_train)
        
        # Training fold metrics
        y_fold_train_pred = model.predict(X_fold_train)
        train_accuracies.append(accuracy_score(y_fold_train, y_fold_train_pred))
        train_recalls.append(recall_score(y_fold_train, y_fold_train_pred, pos_label=positive_class))
        train_precisions.append(precision_score(y_fold_train, y_fold_train_pred, pos_label=positive_class))
        train_f1s.append(f1_score(y_fold_train, y_fold_train_pred, pos_label=positive_class))
        train_roc_aucs.append(roc_auc_score(y_fold_train, model.predict_proba(X_fold_train)[:, list(model.classes_).index(positive_class)]))
        
        # Validation fold metrics
        y_fold_val_pred = model.predict(X_fold_val)
        val_accuracies.append(accuracy_score(y_fold_val, y_fold_val_pred))
        val_recalls.append(recall_score(y_fold_val, y_fold_val_pred, pos_label=positive_class))
        val_precisions.append(precision_score(y_fold_val, y_fold_val_pred, pos_label=positive_class))
        val_f1s.append(f1_score(y_fold_val, y_fold_val_pred, pos_label=positive_class))
        val_roc_aucs.append(roc_auc_score(y_fold_val, model.predict_proba(X_fold_val)[:, list(model.classes_).index(positive_class)]))

    model_final = MLPClassifier(
        hidden_layer_sizes=(l1, l2, l3, l4), random_state=random_state_split, max_iter=10000
    )
    model_final.fit(X_train, y_train)
    y_test_pred = model_final.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred, pos_label=positive_class)
    test_precision = precision_score(y_test, y_test_pred, pos_label=positive_class)
    test_f1 = f1_score(y_test, y_test_pred, pos_label=positive_class)
    test_roc_auc = roc_auc_score(y_test, model_final.predict_proba(X_test)[:, list(model_final.classes_).index(positive_class)])

    results = {
        "name": name, "l1": l1, "l2": l2, "l3": l3, "l4": l4, 
        "kf": n_splits, "random_state_kf": random_state_kf,
        "test_size_split": test_size_split, "random_state_split": random_state_split,
        "acc_nonkfold_cv_train": round(train_accuracy_no_cv, 2),
        "recall_nonkfold_cv_train": round(train_recall_no_cv, 2),
        "precision_nonkfold_cv_train": round(train_precision_no_cv, 2),
        "f1_nonkfold_cv_train": round(train_f1_no_cv, 2),
        "roc_auc_nonkfold_cv_train": round(train_roc_auc_no_cv, 2),
        "mean_acc_kfold_cv_train": round(np.mean(train_accuracies), 2),
        "mean_recall_kfold_cv_train": round(np.mean(train_recalls), 2),
        "mean_precision_kfold_cv_train": round(np.mean(train_precisions), 2),
        "mean_f1_kfold_cv_train": round(np.mean(train_f1s), 2),
        "mean_roc_auc_kfold_cv_train": round(np.mean(train_roc_aucs), 2),
        "mean_acc_kfold_cv_val": round(np.mean(val_accuracies), 2),
        "mean_recall_kfold_cv_val": round(np.mean(val_recalls), 2),
        "mean_precision_kfold_cv_val": round(np.mean(val_precisions), 2),
        "mean_f1_kfold_cv_val": round(np.mean(val_f1s), 2),
        "mean_roc_auc_kfold_cv_val": round(np.mean(val_roc_aucs), 2),
        "accuracy_test": round(test_accuracy, 2),
        "recall_test": round(test_recall, 2),
        "precision_test": round(test_precision, 2),
        "f1_test": round(test_f1, 2),
        "roc_auc_test": round(test_roc_auc, 2)
    }

    # Check if all metrics meet their respective thresholds
    if all([
        round(results["acc_nonkfold_cv_train"], 2) >= threshold_no_cv,
        round(results["mean_acc_kfold_cv_train"], 2) >= threshold_cv,
        round(results["accuracy_test"], 2) >= threshold_test,
        round(results["recall_test"], 2) >= min_recall,
        round(results["precision_test"], 2) >= min_precision,
        round(results["f1_test"], 2) >= min_f1,
        round(results["roc_auc_test"], 2) >= min_roc_auc
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
                print(f"file {filename} was not found.")

target_column_categorized = "pIC50_class"
drop_col_class = ['pIC50_class']

n_splits_list, l1_list, l2_list, l3_list, l4_list = [5, 10], range(2, 8), range(2, 8), range(2, 8), range(2, 8)
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
    print(f"Iteração {count}")
    if count > 1:
        results_df = pd.DataFrame(final_results)
        results_df.to_csv(
            f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_MLP_ANNs_l1-l4.tsv",
            sep="\t", 
            index=False
        )
        print(f"Chegou ao máximo de {count} interações")
        break

results_df = pd.DataFrame(final_results)
print(f"Number of iterations: {count}")
results_df.to_csv(
    f"{pathway_dir}/3-machine_learning_models/supplementary_results/results_conditions_models_MLP_ANNs_l1-l4.tsv",
    sep="\t", 
    index=False
)
