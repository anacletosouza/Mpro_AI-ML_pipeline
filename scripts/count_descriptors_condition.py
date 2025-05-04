import re

def parse_condition(condition):
    # Expressão regular para capturar os componentes
    match = re.match(r"(.+?)_([23]D)_threshold_(\d+(?:\.\d+)?)", condition)
    if match:
        measurement_raw, desc, threshold = match.groups()
        # Substitui "_" por "/" apenas no measurement
        measurement = measurement_raw.replace("_class", "").replace("_", "/")
        return pd.Series([measurement, desc, threshold])
    else:
        return pd.Series([None, None, None])  # Em caso de falha

alpha = 10
thresholds = [round(i / alpha, 1) for i in range(alpha + 1)]
pathway_dir = "/home/anacleto/davinci/projects/htvs/"
path_in = f"{pathway_dir}/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/"
path = f"{pathway_dir}/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/classification/"

out_path = f"{pathway_dir}/3-machine_learning_models/supplementary_results/classification/"

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

for name, df in datasets.items():
    df_target_col = df[target_column_categorized]
    zeros_count = (df == 0).sum()
    nan_count = df.isna().sum()

    df = df.loc[:, zeros_count <= 20]
    df = df.loc[:, nan_count <= 20]
    df = df.fillna(df.mean(numeric_only=True))

    df = df.drop(columns=[col for col in drop_col_class if col in df.columns], errors="ignore")

    datasets[name] = pd.concat([df, df_target_col], axis=1)


results_df = []

for name, df in datasets.items():
    X = df.drop("pIC50_class", axis=1)
    descriptors_str = ", ".join(X.columns)
    results_df.append({
        "condition": name,
        "selected_descriptors": descriptors_str,
        "number_descriptors": len(X.columns)
    })

df_descriptors = pd.DataFrame(results_df)

# Aplica ao DataFrame
df_descriptors[["measurement", "desc", "threshold"]] = df_descriptors["condition"].apply(parse_condition)

df_descriptors.drop("condition", axis=1, inplace=True)

df_descriptors.columns = ['selected_descriptors', 'number_descriptors', 'measurement', 'desc', 'threshold']

df_descriptors = df_descriptors[['measurement', 'desc', 'threshold', 'selected_descriptors', 'number_descriptors']]

df_descriptors_drop_dupl = df_descriptors.drop_duplicates(subset=["measurement", "selected_descriptors", "number_descriptors"], keep="last").sort_values(by=["measurement", "threshold"], ascending=True)

df_descriptors_drop_dupl.to_csv(out_path+"dataset_desc_number_desc_condition.tsv", sep="\t", index=False)


