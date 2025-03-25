import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from sklearn.decomposition import PCA
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings

pathway_dir = "your_pathway"
warnings.filterwarnings(action="ignore", message="'Series.swapaxes' is deprecated and will be removed in a future version. Please use 'Series.transpose' instead.", category=FutureWarning)

packages_dir = f'{pathway_dir}/ML_data_sciences/packages/'
output_dir = f'{pathway_dir}/ML_data_sciences/2-computation_molecular_descriptors/'
datasets = f'{pathway_dir}/ML_data_sciences/1-preprocessing/'
path_datasets = f'{pathway_dir}/ML_data_sciences/1-preprocessing/1-datasets/'
path_class = f"{pathway_dir}/ML_data_sciences/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/classification"
path_reg = f"{pathway_dir}/ML_data_sciences/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/regression"

os.makedirs(packages_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(datasets, exist_ok=True)
os.makedirs(path_class, exist_ok=True)
os.makedirs(path_reg, exist_ok=True)

dimorphite_repo = os.path.join(packages_dir, 'dimorphite_dl')
if not os.path.exists(dimorphite_repo):
    get_ipython().system('git clone https://github.com/durrantlab/dimorphite_dl.git $dimorphite_repo')


os.chdir(output_dir)
print(f"Current directory: {os.getcwd()}")

def execute_script(smiles: str, 
                   packages_dir: str, 
                   min_ph: float = 6.8, 
                   max_ph: float = 7.2, 
                   max_variants: int = 1) -> str:
    """
    Executes the dimorphite_dl.py script to calculate the protonation state of a SMILES.
    """
    import numpy as np
    import pandas as pd
    import subprocess
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
   
    try:
        result = subprocess.run(
            [
                'python',
                f'{packages_dir}dimorphite_dl/dimorphite_dl.py',
                '--smiles', smiles,
                '--min_ph', str(min_ph),
                '--max_ph', str(max_ph),
                '--max_variants', str(max_variants)
            ],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Error executing script for SMILES {smiles}: {result.stderr.strip()}")
        output = result.stdout.strip()
        return output.splitlines()[-1] if output else None
    except Exception as e:
        print(f"Failed to process SMILES {smiles}: {e}")
        return None

def process_chunk(chunk, 
                  packages_dir: str, 
                  min_ph: float, 
                  max_ph: float, 
                  max_variants: int):
    """
    Processes a chunk of SMILES using the dimorphite_dl script.
    """
    return [execute_script(smiles, packages_dir, min_ph, max_ph, max_variants) for smiles in chunk]

def process_protonation(df: pd.DataFrame, 
                        number_threads=27,  
                        packages_dir: str = '', 
                        min_ph: float = 6.8, 
                        max_ph: float = 7.2, 
                        max_variants: int = 1) -> pd.DataFrame:
    """
    Processes a DataFrame of SMILES to calculate protonation states using dimorphite_dl,
    with multithreading support.
    """

    # Parallel execution of subprocesses with threads (for multithreading)
    n_threads = min(len(df), number_threads)  # Limits the number of threads
    chunks = np.array_split(df["SMILES"], n_threads)
    index_chunks = np.array_split(df.index, n_threads)

    results = []
    indices = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {
            executor.submit(
                process_chunk, chunk, packages_dir, min_ph, max_ph, max_variants
            ): idx_chunk
            for chunk, idx_chunk in zip(chunks, index_chunks)
        }

        for future in as_completed(futures):
            try:
                processed_result = future.result()
                corresponding_indices = futures[future]
                results.extend(processed_result)
                indices.extend(corresponding_indices)
            except Exception as e:
                print(f"Error during parallel execution: {e}")


    df_result = pd.Series(results, index=indices).reindex(df.index)
    df["SMILES"] = df_result

    return df

def desc_calc_2d(smiles):
    """
    Calculates all 2D descriptors for a molecule represented in SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:

        return {desc[0]: desc[1](mol) for desc in Descriptors.descList}
    else:
        return {desc[0]: None for desc in Descriptors.descList}

def calculate_2d_descriptors(df, number_threads=30):
    """
    Calculates 2D descriptors for SMILES in the DataFrame using multithreading.
    """

    n_threads = min(len(df), number_threads)  # Limits the number of threads to the size of the DataFrame
    smiles_chunks = np.array_split(df['SMILES'], n_threads)  # Splits SMILES into chunks for parallelization
    index_chunks = np.array_split(df.index, n_threads)

    results = []
    indices = []

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(desc_calc_2d, smiles): idx 
                   for chunk, idx_chunk in zip(smiles_chunks, index_chunks)
                   for smiles, idx in zip(chunk, idx_chunk)}

        for future in as_completed(futures):
            try:
                results.append(future.result())
                indices.append(futures[future])
            except Exception as e:
                print(f"Error calculating descriptors for a SMILES: {e}")
                results.append({desc[0]: None for desc in Descriptors.descList})
                indices.append(futures[future])


    descriptors_df = pd.DataFrame(results, index=indices).reindex(df.index)

    df = pd.concat([df, descriptors_df], axis=1)

    return df

def desc_calc_3d(smiles, N_threads=22):


    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol_3d = Chem.AddHs(mol)
        params = AllChem.ETKDGv2()
        params.randomSeed = 56
        params.maxAttempts = 1
        params.useRandomCoords = True
        params.enforceChirality = True
        params.numThreads = N_threads
        embedding_status = AllChem.EmbedMolecule(mol_3d, params)
        if embedding_status != 0:
            print(f"Embedding failed for SMILES: {smiles}")  
            return {}
        mol_desc = {}
        for name, func in Descriptors.descList:
            try:
                mol_desc[name] = func(mol)
            except Exception as e:
                mol_desc[name] = None
        for func_name in dir(rdMolDescriptors):
            if func_name.startswith('Calc'):
                func = getattr(rdMolDescriptors, func_name)
                try:
                    mol_desc[func_name] = func(mol_3d)
                except Exception as e:
                    mol_desc[func_name] = None
        return mol_desc
    else:
        print(f"Failed to convert SMILES to molecule: {smiles}")  
        return {}

def calculate_3d_descriptors(df, number_threads=30):
        
    n_threads = min(len(df), number_threads)
    smiles_chunks = np.array_split(df['SMILES'], n_threads)
    index_chunks = np.array_split(df.index, n_threads)
    results = []
    indices = []
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(desc_calc_3d, smiles): idx 
                   for chunk, idx_chunk in zip(smiles_chunks, index_chunks)
                   for smiles, idx in zip(chunk, idx_chunk)}

        for future in as_completed(futures):
            try:
                results.append(future.result())
                indices.append(futures[future])
            except Exception as e:
                print(f"Error calculating descriptors for a SMILES: {e}")
                results.append({})
                indices.append(futures[future])
    
    descriptors_df = pd.DataFrame(results, index=indices).reindex(df.index)
    df = pd.concat([df, descriptors_df], axis=1)
    
    return df


def process_dataset(name, filepath):
    df = pd.read_csv(filepath, sep = "\t", index_col=False)
    start_time = time.time()
    df = process_protonation(df, 
                             packages_dir=f'{pathway_dir}/ML_data_sciences/packages/',
                             number_threads=27,
                             min_ph=6.8, 
                             max_ph=7.2, 
                             max_variants=1)
    end_time = time.time()
    print(f"Execution time ({name}): {end_time - start_time:.2f} seconds")
    return df


def calculate_descriptors(df, name):
    output_file = f"{pathway_dir}/ML_data_sciences/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/original_data/"
    start_time = time.time()
    print(f"Starting calculation of 2D and 3D descriptors for {name}...")
    df_2D = calculate_2d_descriptors(df)
    df_3D = calculate_3d_descriptors(df)
    end_time = time.time()
    print(f"Execution time for 2D and 3D descriptors ({name}): {end_time - start_time:.2f} seconds")
    output_file_2D = f"{output_file}df_{name}_desc_2D_original_data.tsv"
    df_2D.to_csv(output_file_2D, sep="\t", index=False)
    output_file_3D = f"{output_file}df_{name}_desc_3D_original_data.tsv"
    df_3D.to_csv(output_file_3D, sep="\t", index=False)

    return df_2D, df_3D
    
def rm_bad_col(df, drop_col, target_col="pIC50_updated"):
    df_target = df[target_col]
    df_filtered = df.drop(columns=[col for col in drop_col if col in df.columns], errors="ignore")
    df_filtered = df_filtered.select_dtypes(include=["float64"])
    return pd.concat([df_filtered, df_target], axis=1)

def filter_col_correl(df_r, threshold=0.4, target_col="pIC50_updated"):

    df_target = df_r[[target_col]]  
    df_r = df_r.drop(columns=[target_col])  
    

    correlation_matrix = df_r.corr()
    

    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    

    to_drop = [column for column in upper_triangle.columns if any(abs(upper_triangle[column]) >= threshold)]


    df_reduced = df_r.drop(columns=to_drop)


    df_reduced = df_reduced.loc[:, ~(df_reduced.isin([0.0]).all())]

    zeros_count = (df_reduced == 0).sum()  
    nan_count = df_reduced.isna().sum()  


    df_reduced = df_reduced.loc[:, zeros_count <= 20]
    df_reduced = df_reduced.loc[:, nan_count <= 20]
    df_reduced = df_reduced.fillna(df_reduced.mean(numeric_only=True))
    

    return pd.concat([df_reduced, df_target], axis=1)
    


datasets_reg = {
    "fluor_reg": "fluor_dropduplic_regression.tsv",
    "FRET_reg": "FRET_dropduplic_regression.tsv",
    "FRET_fluor_SPR_reg": "FRET_fluor_SPR_dropduplic_regression.tsv"
}

datasets_class = {
    "fluor_class": "fluor_dropduplic_class.tsv",
    "FRET_class": "FRET_dropduplic_class.tsv",
    "FRET_fluor_SPR_class": "FRET_fluor_SPR_dropduplic_class.tsv"
}

drop_col_reg = [
    'SMILES', 'name', 'name_article', 'IC50_uM', 'KD_uM', 'Ki_uM', 'hit_selection',
    'conc_Mpro_nM', 'conc_subst_uM', 'detergent', 'doi_article', 'assay_type',
    'cell_type', 'EC50_uM', 'KM_pred_uM', 'Ki_KD_mean', 'IC50_conv', 'pIC50_value',
    'EC50_conv', 'pEC50', 'pIC50_mean', 'pEC50_mean', 'pEC50_updated', 'pIC50_updated', 'pIC50_class'
]

drop_col_class = [
    'SMILES', 'name', 'name_article', 'IC50_uM', 'KD_uM', 'Ki_uM', 'hit_selection',
    'conc_Mpro_nM', 'conc_subst_uM', 'detergent', 'doi_article', 'assay_type',
    'cell_type', 'EC50_uM', 'KM_pred_uM', 'Ki_KD_mean', 'IC50_conv', 'pIC50_value',
    'EC50_conv', 'pEC50', 'pIC50_mean', 'pEC50_mean', 'pIC50_updated', 'pEC50_updated', 'pIC50_class'
]

processed_data_reg = {}
descriptors_reg = {}
processed_data_class = {}
descriptors_class = {}


for name, filename in datasets_reg.items():
    filepath = path_datasets + filename
    processed_data_reg[name] = process_dataset(name, filepath)


for name, df in processed_data_reg.items():
    descriptors_reg[name] = calculate_descriptors(df, name)


categories_reg = ["FRET_reg", "fluor_reg", "FRET_fluor_SPR_reg"]
processed_descriptors_reg = {}

categories_class = ["FRET_class", "fluor_class", "FRET_fluor_SPR_class"]
processed_descriptors_class = {}

for category in categories_reg:
    processed_descriptors_reg[category] = {
        "2D": descriptors_reg[category][0],
        "3D": descriptors_reg[category][1]
    }


for name, filename in datasets_class.items():
    filepath = path_datasets + filename
    processed_data_class[name] = process_dataset(name, filepath)


for name, df in processed_data_class.items():
    descriptors_class[name] = calculate_descriptors(df, name)


categories_reg = ["FRET_reg", "fluor_reg", "FRET_fluor_SPR_reg"]
processed_descriptors_reg = {}

categories_class = ["FRET_class", "fluor_class", "FRET_fluor_SPR_class"]
processed_descriptors_class = {}

for category in categories_reg:
    processed_descriptors_reg[category] = {
        "2D": descriptors_reg[category][0],
        "3D": descriptors_reg[category][1]
    }


for category in categories_class:
    processed_descriptors_class[category] = {
        "2D": descriptors_class[category][0],
        "3D": descriptors_class[category][1]
    }


target_reg = "pIC50_updated"
target_class = "pIC50_class"


alpha = 10
thresholds = [round(i / alpha, 1) for i in range(alpha + 1)]


for category, data in processed_descriptors_reg.items():
    for dim, df in data.items():
        for threshold in thresholds:
            df_r = rm_bad_col(df, drop_col_reg)
            df_rf = filter_col_correl(df_r, threshold, target_col=target_reg)
            df_processed = df_rf 

            output_file = f"{pathway_dir}/ML_data_sciences/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/regression/df_{category}_{dim}_threshold_{threshold}_reg.tsv"
            df_processed.to_csv(output_file, sep="\t", index=False)



for category, data in processed_descriptors_class.items():
    for dim, df in data.items():
        for threshold in thresholds:
            df_r = rm_bad_col(df, drop_col_class, target_col=target_class)
            df_rf = filter_col_correl(df_r, threshold, target_col=target_class)
            df_processed = df_rf 

            output_file = f"{pathway_dir}/ML_data_sciences/2-computation_molecular_descriptors/dataset_non-correlation_desc_reduced_dim/classification/df_{category}_{dim}_threshold_{threshold}_class.tsv"
            df_processed.to_csv(output_file, sep="\t", index=False)
