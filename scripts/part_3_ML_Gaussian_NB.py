from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import concurrent.futures
import random
import os
from sklearn.preprocessing import StandardScaler



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

def train_model(name, 
                df, 
                n_splits=5,
                random_state_kf=90, 
                test_size_split=0.3, 
                random_state_split=69,
                threshold_no_cv = 0.6, 
                threshold_cv = 0.6, 
                threshold_test = 0.5):

    if target_column_categorized not in df.columns:
        print(f"Dataset {name} without column {target_column_categorized}. Jumpping.")
        return None


    X = df.drop(columns=[target_column_categorized]).select_dtypes(include=["number"]).dropna(axis=1)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)    
    y = df[target_column_categorized]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=random_state_split, stratify=y)


    model_no_cv = GaussianNB()
    model_no_cv.fit(X_train, y_train)
    y_train_pred_no_cv = model_no_cv.predict(X_train)
    train_accuracy_no_cv = accuracy_score(y_train, y_train_pred_no_cv)


    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state_kf)
    train_accuracies = []
    val_accuracies = []

    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = GaussianNB()
        model.fit(X_fold_train, y_fold_train)


        y_fold_train_pred = model.predict(X_fold_train)
        train_accuracies.append(accuracy_score(y_fold_train, y_fold_train_pred))


        y_fold_val_pred = model.predict(X_fold_val)
        val_accuracies.append(accuracy_score(y_fold_val, y_fold_val_pred))


    model_final = GaussianNB()
    model_final.fit(X_train, y_train)
    y_test_pred = model_final.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
        
    results = {
        "name": name, 
        "kf": n_splits,
        "random_state_kf": random_state_kf,
        "test_size_split": test_size_split,
        "random_state_split": random_state_split,
        "acc_nonkfold_cv_train": round(train_accuracy_no_cv, 2),
        "mean_acc_kfold_cv_train": round(np.mean(train_accuracies), 2), 
        "accuracy_test": round(test_accuracy, 2)
    }
    if round(train_accuracy_no_cv, 2) >= threshold_no_cv and round(np.mean(train_accuracies), 2) >= threshold_cv and round(test_accuracy, 2) >= threshold_test:
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
                print(f"file {filename} did not find.")


target_column_categorized = "pIC50_class"

drop_col_class = ['pIC50_class']


n_splits_list = [5, 10]
random_state_kf_list = random.sample(range(101), 10)
random_state_list = random.sample(range(101), 10)
test_size_split_list = [0.20, 0.25, 0.30]
random_state_split_list = random.sample(range(101), 10)
results = []


for name, df in datasets.items():
    
    df_target_col = df[target_column_categorized]  
    zeros_count = (df == 0).sum()  
    nan_count = df.isna().sum()  
    

    df = df.loc[:, zeros_count <= 20]
    df = df.loc[:, nan_count <= 20]
    df = df.fillna(df.mean(numeric_only=True))
    

    df = df.drop(columns=[col for col in drop_col_class if col in df.columns], errors="ignore")
    

    datasets[name] = pd.concat([df, df_target_col], axis=1)


tasks = generate_tasks()


final_results = []
count = 0


def get_unprocessed_datasets(results, datasets):

    processed = {r["name"] for r in results}

    return {name: df for name, df in datasets.items() if name not in processed}


while datasets:

    tasks = generate_tasks()


    partial_results = []


    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        for result in executor.map(train_model_wrapper, tasks):
            if result:
                partial_results.append(result)


    final_results.extend(partial_results)
    count = count + 1
    print(f"Iteração {count}")
    
    if count > 2:
        results_df = pd.DataFrame(final_results)
        results_df.to_csv(f"{pathway_dir}/ML_data_sciences/3-machine_learning_models/ML_conditions_2_cat/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_gaussian_naivebayes.tsv", sep = "\t", index=False)
        print(f"Count: {count}!")
        break
    	

    datasets = get_unprocessed_datasets(final_results, datasets)


results_df = pd.DataFrame(final_results)
results_df.to_csv(f"{pathway_dir}/ML_data_sciences/3-machine_learning_models/ML_conditions_2_cat/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_gaussian_naivebayes.tsv", sep = "\t", index=False)

