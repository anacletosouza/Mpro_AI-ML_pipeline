from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
import concurrent.futures
import random
import os

pathway_dir = "your_pathway"
path = f"{pathway_dir}/ML_data_sciences/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/"


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


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_split, random_state=random_state_split, stratify=y
    )


    model_no_cv = SVC(kernel=kernel, random_state=random_state_split)
    model_no_cv.fit(X_train, y_train)
    y_train_pred_no_cv = model_no_cv.predict(X_train)
    train_accuracy_no_cv = accuracy_score(y_train, y_train_pred_no_cv)


    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state_kf)
    train_accuracies = []
    val_accuracies = []

    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = SVC(kernel=kernel, random_state=random_state_kf)
        model.fit(X_fold_train, y_fold_train)

        train_accuracies.append(accuracy_score(y_fold_train, model.predict(X_fold_train)))
        val_accuracies.append(accuracy_score(y_fold_val, model.predict(X_fold_val)))


    model_final = SVC(kernel=kernel, random_state=random_state_split)
    model_final.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model_final.predict(X_test))

    results = {
        "name": name,
        "kernel": kernel,
        "kf": n_splits,
        "random_state_kf": random_state_kf,
        "test_size_split": test_size_split,
        "random_state_split": random_state_split,
        "acc_nonkfold_cv_train": round(train_accuracy_no_cv, 2),
        "mean_acc_kfold_cv_train": round(np.mean(train_accuracies), 2),
        "accuracy_test": round(test_accuracy, 2),
    }
    if (
        round(train_accuracy_no_cv, 2) >= threshold_no_cv
        and round(np.mean(train_accuracies), 2) >= threshold_cv
        and round(test_accuracy, 2) >= threshold_test
    ):
        print(results)
        return results
    else:
        return None


alpha = 10
thresholds = [round(i / alpha, 1) for i in range(alpha + 1)]


path = f"{pathway_dir}/ML_data_sciences/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/classification/"


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
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:  
        results = list(filter(None, executor.map(train_model, tasks)))
        final_results.extend(results)
    datasets = {name: df for name, df in datasets.items() if name not in {res["name"] for res in results}}
    count += 1
    if count > 2:
        results_df = pd.DataFrame(final_results)
        results_df.to_csv(f"{pathway_dir}/ML_data_sciences/3-machine_learning_models/ML_conditions_2_cat/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_SVM.tsv", sep="\t", index=False)
        print(f"Count: {count}")
        break

results_df = pd.DataFrame(final_results)
results_df.to_csv(f"{pathway_dir}/ML_data_sciences/3-machine_learning_models/ML_conditions_2_cat/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_SVM.tsv", sep="\t", index=False)


