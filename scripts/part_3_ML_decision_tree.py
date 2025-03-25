from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import concurrent.futures
import random
from sklearn.preprocessing import StandardScaler

pathway_dir = "your_pathway"
path = "{pathway_dir}/ML_data_sciences/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/"


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
                random_state_kf=90, test_size_split=0.3, random_state_split=69, threshold_no_cv=0.6, threshold_cv=0.6, 
                threshold_test=0.5):
    if target_column_categorized not in df.columns:
        print(f"Dataset {name} has not column {target_column_categorized}. Jumpping.")
        return None

    X = df.drop(columns=[target_column_categorized]).select_dtypes(include=["number"]).dropna(axis=1)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)    
    y = df[target_column_categorized]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=random_state_split, stratify=y)

    model_no_cv = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, 
                                         min_samples_leaf=min_samples_leaf, random_state=random_state_split)
    model_no_cv.fit(X_train, y_train)
    train_accuracy_no_cv = accuracy_score(y_train, model_no_cv.predict(X_train))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state_kf)
    train_accuracies, val_accuracies = [], []

    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_fold_train, y_fold_val = y_train.iloc[train_index], y_train.iloc[val_index]

        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state_kf)
        model.fit(X_fold_train, y_fold_train)
        train_accuracies.append(accuracy_score(y_fold_train, model.predict(X_fold_train)))
        val_accuracies.append(accuracy_score(y_fold_val, model.predict(X_fold_val)))

    model_final = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, 
                                         min_samples_leaf=min_samples_leaf, random_state=random_state_split)
    model_final.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model_final.predict(X_test))

    results = {
        "name": name, "max_depth": max_depth, "criterion" : criterion, "min_samples_split" : min_samples_split, "min_samples_leaf" : min_samples_leaf, "min_samples_split": min_samples_split, "kf": n_splits, 
        "random_state_kf": random_state_kf, "test_size_split": test_size_split, "random_state_split": random_state_split,
        "acc_nonkfold_cv_train": round(train_accuracy_no_cv, 2),
        "mean_acc_kfold_cv_train": round(np.mean(train_accuracies), 2), 
        "accuracy_test": round(test_accuracy, 2)
    }
    
    if all(round(results[key], 2) >= threshold for key, threshold in 
           zip(["acc_nonkfold_cv_train", "mean_acc_kfold_cv_train", "accuracy_test"], 
               [threshold_no_cv, threshold_cv, threshold_test])):
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
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
        final_results.extend(filter(None, executor.map(train_model_wrapper, tasks)))
    datasets = {name: df for name, df in datasets.items() if name not in {res["name"] for res in final_results}}
    count += 1
    print(f"Interations: {count}")
    
    if count > 10:
        results_df = pd.DataFrame(final_results)
        results_df.to_csv(f"{pathway_dir}/ML_data_sciences/3-machine_learning_models/ML_conditions_2_cat/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_decision_tree_classifier.tsv", sep="\t", index=False)
        print(f"interation completed: {count} interactions")
        break

results_df_decision_tree_models = pd.DataFrame(final_results)
print(f"Number of iterations: {count}")
results_df_decision_tree_models.to_csv(f"{pathway_dir}/ML_data_sciences/3-machine_learning_models/ML_conditions_2_cat/supplementary_results/classification/hyperparameters_models_performance/results_conditions_models_decision_tree_classifier.tsv", sep="\t", index=False)


#########################################
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
import concurrent.futures
import random
from sklearn.preprocessing import StandardScaler


target_column_categorized = "pIC50_class"

test_size_split = 0.25
random_state_split = 46
max_depth = None
criterion="entropy"
min_samples_split = 5
min_samples_leaf = 1


X_nonnormalized = df.drop(columns=[target_column_categorized]).select_dtypes(include=["number"]).dropna(axis=1)
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X_nonnormalized), columns=X_nonnormalized.columns, index=X_nonnormalized.index)    
y = df[target_column_categorized]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=random_state_split, stratify=y)

model_no_cv = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_state_split)
model_no_cv.fit(X_train, y_train)
train_accuracy_no_cv = accuracy_score(y_train, model_no_cv.predict(X_train))

model_final = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_state_split)
model_final.fit(X_train, y_train)
test_accuracy = accuracy_score(y_test, model_final.predict(X_test))


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib

#joblib.dump(model_final, "model_cart_model1_condition_22.pkl")
# model = joblib.load("model_cart_model1_condition_22.pkl")
model = model_final
plt.figure(figsize=(30, 15))  # Aumente o tamanho da figura
plot_tree(model, filled=True, feature_names=X_train.columns, class_names=["active", "inactive"], fontsize=12)
plt.show()

#exportando arvore em txt
from sklearn.tree import DecisionTreeClassifier, export_text
tree_rules = export_text(model_final, feature_names=X_train.columns)
# Salvar as regras em um arquivo de texto
with open("FRET_fluor_SPR_model_10_cart.txt", "w") as file:
    file.write(tree_rules)

# reventendo as regras da árvore de decisão de dados padronizados

import pandas as pd
import numpy as np

df = pd.read_csv('df_FRET_fluor_SPR_class_2D_threshold_1.0_class.tsv', sep="\t", index_col=False)


X = X_nonnormalized
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=random_state_split, stratify=y)
mean_values = X_train.mean()
std_values = X_train.std()

# Suponha que df já esteja definido
df = pd.concat([X_train, y_train], axis=1, join="inner")

molecular_descriptor = 'CalcChi3v'
value = -0.48
value * std_values[molecular_descriptor] + mean_values[molecular_descriptor]


##############################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define parameters
target_column_categorized = "pIC50_class"
test_size_split = 0.25
random_state_split = 46
max_depth = None
criterion = "entropy"
min_samples_split = 5
min_samples_leaf = 1

# Load and prepare the data
X_nonnormalized = df.drop(columns=[target_column_categorized]).select_dtypes(include=["number"]).dropna(axis=1)
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_nonnormalized), columns=X_nonnormalized.columns, index=X_nonnormalized.index)
y = df[target_column_categorized]

# Convert 'active' and 'inactive' to binary values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # 'active' -> 1, 'inactive' -> 0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=test_size_split, random_state=random_state_split, stratify=y_encoded)

# Train the model
model_final = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_state_split)
model_final.fit(X_train, y_train)

# Get the predicted probabilities
y_score = model_final.predict_proba(X_test)

# Compute the ROC curve
fpr, tpr, roc_auc = roc_curve(y_test, y_score[:, 1])  # Considering '1' as the positive class
roc_auc_value = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color="darkorange", lw=lw, label=f"ROC Curve (area = {roc_auc_value:0.2f})")

# Diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")

# Layout settings
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)", fontsize=18)
plt.ylabel("True Positive Rate (TPR)", fontsize=18)
plt.title("ROC Curve for CART model (FRET/fluorescence/SPR)", fontsize=18)

# Customize axis thickness and font size of ticks
plt.tick_params(axis='both', which='major', labelsize=18, width=2)  # Increase font size and axis thickness

# Remove grid
plt.grid(False)

# Remove top and right spines
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')

# Make the x and y axis spines thicker
plt.gca().spines['left'].set_linewidth(2)  # Thicken the left (y) axis
plt.gca().spines['bottom'].set_linewidth(2)  # Thicken the bottom (x) axis

# Display the legend
plt.legend(loc="lower right", fontsize=18)

# Show the plot
plt.show()




##############################################################################################################################





import re
import pandas as pd

pathway_dir = "your_pathway"
df = pd.read_csv('{pathway_dir}/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/original_data/df_FRET_fluor_SPR_class_desc_2D_original_data.tsv', sep="\t", index_col=False)

descriptors_list = ['SMR_VSA10','qed','SlogP_VSA2'] # this is a example

df_join = df[['SMILES', 'name', 'name_article'] + descriptors_list + ['IC50_uM', 'KD_uM', 'pIC50_updated', 'pEC50_updated', 'pIC50_class']]
target_column_categorized = "pIC50_class"

test_size_split = 0.20
random_state_split = 46
max_depth = None
criterion="entropy"
min_samples_split = 6
min_samples_leaf = 1


y = df[target_column_categorized]
X = df[['SMILES', 'name', 'name_article', 'SMR_VSA10','qed','SlogP_VSA2', 'IC50_uM', 'KD_uM', 'pIC50_updated', 'pEC50_updated']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=random_state_split, stratify=y)

df = pd.concat([X_train, y_train], axis=1, join="inner")

def parse_tree(tree_text):
    """Parse a decision tree text and generate corresponding pandas queries."""
    lines = tree_text.split('\n')
    queries = []
    stack = []

    for line in lines:
        match = re.search(r'([| ]*)\|--- (.+)', line)
        if match:
            indent = len(match.group(1)) // 4
            condition = match.group(2)


            stack = stack[:indent]
            if 'class:' not in condition:
                stack.append(condition.replace('<=', '<=').replace('> ', '>'))
            else:
                label = condition.split(':')[-1].strip()
                query = " and ".join(stack)
                queries.append((query, label))

    return queries

def count_classes(df, queries):
    """Apply queries to the dataframe and count active/inactive classes."""
    results = []
    for query, label in queries:
        count = df.query(query).pIC50_class.value_counts().to_dict()
        count['query'] = query
        count['label'] = label
        results.append(count)

    return pd.DataFrame(results)


tree_text = """
Include tree from CART model
"""

queries = parse_tree(tree_text)

result = count_classes(df, queries)
print(result)

########################################################################################

descriptors_list="""qed
HallKierAlpha
Chi4v
SPS
Kappa3
LabuteASA
BertzCT
Chi2v
MinEStateIndex
FpDensityMorgan3
Chi1v
TPSA
FpDensityMorgan1
AvgIpc""".split()

    
##########################################################################################

df_join = df[['SMILES', 'name', 'name_article'] + descriptors_list + ['IC50_uM', 'KD_uM', 'pIC50_updated', 'pEC50_updated', 'pIC50_class']]
target_column_categorized = "pIC50_class"

test_size_split = 0.25
random_state_split = 46
max_depth = None
criterion="entropy"
min_samples_split = 5
min_samples_leaf = 1


#X_nonnormalized = df.drop(columns=[target_column_categorized]).select_dtypes(include=["number"]).dropna(axis=1)
#scaler = StandardScaler()
#X = pd.DataFrame(scaler.fit_transform(X_nonnormalized), columns=X_nonnormalized.columns, index=X_nonnormalized.index)
y = df_join[target_column_categorized]
X = df_join[['SMILES', 'name', 'name_article'] + descriptors_list + ['IC50_uM', 'KD_uM', 'pIC50_updated', 'pEC50_updated']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_split, random_state=random_state_split, stratify=y)

df_join = pd.concat([X_train, y_train], axis=1, join="inner")

df_join = df_join.loc[:, ~df_join.columns.duplicated()]


########################################################################################

import pandas as pd
import scipy.stats as stats

# Aplicação do filtro no DataFrame
df_join_1 = df_join.query('(qed <= 0.78535) & (HallKierAlpha > -4.333921917723973) & (Chi4v > 3.651208251502653) & (qed > 0.1882573899090756) & (SPS <= 11.140608843078752) & (Kappa3 <= 4.399656689137886)')

descriptor = "qed"

n = len(df_join_1[descriptor])  # Número de observações

mean = df_join_1[descriptor].mean()  # Média
std = df_join_1[descriptor].std(ddof=1)  # Desvio padrão amostral

t_value = stats.t.ppf(0.975, df=n - 1)

margin_of_error = t_value * (std / (n ** 0.5))
lower_bound = mean - margin_of_error
upper_bound = mean + margin_of_error

print(f"IC[95%; {descriptor}] = ({lower_bound:.4f}, {upper_bound:.4f})")


