from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import concurrent.futures
import random
import os
from sklearn.preprocessing import StandardScaler

pathway_dir = "your_pathway"
path = "f{pathway_dir}/ML_data_sciences/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/classification/"


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
    test_size_split=0.3, random_state_split=69, threshold_no_cv=0.6, threshold_cv=0.6, threshold_test=0.5
):
    if target_column_categorized not in df.columns:
        print(f"Dataset {name} has no column {target_column_categorized}. Jumpping.")
        return None

    X = df.drop(columns=[target_column_categorized]).select_dtypes(include=["number"]).dropna(axis=1)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    y = df[target_column_categorized]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_split, random_state=random_state_split, stratify=y
    )

    model_no_cv = MLPClassifier(
        hidden_layer_sizes=(l1, l2, l3, l4), random_state=random_state_split, max_iter=10000
    )
    model_no_cv.fit(X_train, y_train)
    train_accuracy_no_cv = accuracy_score(y_train, model_no_cv.predict(X_train))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state_kf)
    train_accuracies, val_accuracies = [], []

    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = MLPClassifier(hidden_layer_sizes=(l1, l2, l3, l4), random_state=random_state_kf, max_iter=10000)
        model.fit(X_fold_train, y_fold_train)
        train_accuracies.append(accuracy_score(y_fold_train, model.predict(X_fold_train)))
        val_accuracies.append(accuracy_score(y_fold_val, model.predict(X_fold_val)))

    model_final = MLPClassifier(
        hidden_layer_sizes=(l1, l2, l3, l4), random_state=random_state_split, max_iter=10000
    )
    model_final.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model_final.predict(X_test))

    results = {
        "name": name, "l1": l1, "l2": l2, "l3": l3, "l4": l4, "kf": n_splits, "random_state_kf": random_state_kf,
        "test_size_split": test_size_split, "random_state_split": random_state_split,
        "acc_nonkfold_cv_train": round(train_accuracy_no_cv, 2),
        "mean_acc_kfold_cv_train": round(np.mean(train_accuracies), 2),
        "accuracy_test": round(test_accuracy, 2)
    }

    if all(
        round(results[key], 2) >= threshold
        for key, threshold in zip(
            ["acc_nonkfold_cv_train", "mean_acc_kfold_cv_train", "accuracy_test"],
            [threshold_no_cv, threshold_cv, threshold_test]
        )
    ):
        print(results)
        return results
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        final_results.extend(filter(None, executor.map(train_model_wrapper, tasks)))
    datasets = {name: df for name, df in datasets.items() if name not in {res["name"] for res in final_results}}
    count += 1
    print(f"Iteração {count}")
    if count >= 2:
        results_df = pd.DataFrame(final_results)
        results_df.to_csv(
                         f"{pathway_dir}/ML_data_sciences/3-machine_learning_models/ML_conditions_2_cat/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_MLP_ANNs_l1-l4.tsv",
                         sep="\t", 
                         index=False
                         )
        print("Chegou ao máximo de 3 interações")
        break

results_df = pd.DataFrame(final_results)
print(f"Number of iterations: {count}")
results_df.to_csv(
                 f"{pathway_dir}/ML_data_sciences/3-machine_learning_models/ML_conditions_2_cat/supplementary_results/results_conditions_models_MLP_ANNs_l1-l4.tsv",
                 sep="\t", 
                 index=False
                 )





