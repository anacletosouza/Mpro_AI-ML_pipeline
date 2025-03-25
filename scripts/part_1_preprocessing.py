#!/usr/bin/env python

import pandas as pd  
import numpy as np  
import os  
import matplotlib.pyplot as plt  
import plotly.express as px 
import re

pathway_dir = "your_pathway" # example: pathway_dir = "/home"
output_dir = f"{pathway_dir}/ML_data_sciences/1-preprocessing/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

os.chdir(output_dir)

if not os.path.exists("1-datasets"):
    os.makedirs("1-datasets")

if not os.path.exists("2-figures"):
    os.makedirs("2-figures")

df = pd.read_csv(f'{output_dir}dataset.csv', sep=",", index_col = False)

df = df[df['SMILES'].notna()] 

df['name'] = df['name'].fillna(df['name_article'])
df['name_article'] = df['name_article'].fillna(df['name'])

def extrair_doi(url):
    if not isinstance(url, (str, bytes)):  
        return None
    match = re.search(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', url)
    return match.group(0) if match else None


df['doi_article'] = df['doi_article'].apply(extrair_doi)

df["KM_pred_uM"] = df.apply(
    lambda row: 24 if row["conc_Mpro_nM"] > 50 and row["conc_subst_uM"] > 15
    else 75.41 if row["conc_subst_uM"] > 15
    else 40.87,
    axis=1
)

df["Ki_KD_mean"] = df.apply(
    lambda row: (row['KD_uM'] + row['Ki_uM']) / 2 if pd.notna(row['Ki_uM']) and pd.notna(row['KD_uM']) else (row['Ki_uM'] if pd.notna(row['Ki_uM']) else (row['KD_uM'] if pd.notna(row['KD_uM']) else np.nan)), axis=1)

def convert_ic50(value):
     value = str(value).strip() 
     
     if value.startswith(">") or value.startswith("<") or value.replace(".", "", 1).isdigit() or (value.count('.') == 1 and value.replace(".", "", 1).isdigit()):
         return value
     else:
         return np.nan

df["IC50_conv"] = df["IC50_uM"].apply(convert_ic50)

def calculate_pIC50(row):
    try:
        ic50_value = float(row["IC50_conv"])
        if ic50_value > 0 and ic50_value < 90:
            return -np.log10(ic50_value/10**6)
    except (ValueError, TypeError):
        pass

    adjustment_factor = 1 + ((row["conc_Mpro_nM"] / 10**9) / (row["KM_pred_uM"] / 10**6)) + ((row["conc_Mpro_nM"] / 10**9) / 2)

    if isinstance(row["IC50_conv"], str) and (">" in row["IC50_conv"] or "<" in row["IC50_conv"]):
        if -np.log10(row["Ki_KD_mean"] * adjustment_factor/10**6) > 0:
            return -np.log10(row["Ki_KD_mean"] * adjustment_factor/10**6)

    if isinstance(row["Ki_KD_mean"], (int, float)) and row["Ki_KD_mean"] > 0:
        return -np.log10(row["Ki_KD_mean"] * adjustment_factor/10**6)

    return np.nan


df["pIC50_value"] = df.apply(calculate_pIC50, axis=1)


df['EC50_conv'] = pd.to_numeric(df['EC50_uM'], errors='coerce')
df['EC50_conv'] = df['EC50_conv'].where(df['EC50_conv'] > 0, np.nan)


df['pEC50'] = -np.log10(df['EC50_conv'] / 1e6)
df['pEC50'] = df['pEC50'].where(df['EC50_conv'].notna(), np.nan)

df["name"] = df["name"].str.upper()
df["name_article"] = df["name_article"].str.upper()

df['pIC50_mean'] = df.groupby(['SMILES', 'name_article', 'assay_type', 'detergent'])['pIC50_value'].transform('mean')
df["pEC50_mean"] = df.groupby(['SMILES', 'name_article', 'assay_type', 'detergent'])['pEC50'].transform('mean')

df.loc[:, 'pIC50_updated'] = df['pIC50_mean'].fillna(df['pIC50_value'])

df.loc[:, 'pEC50_updated'] = df['pEC50_mean'].fillna(df['pEC50'])

df_FRET = df[df["assay_type"] == "FRET"]
df_fluor = df[df["assay_type"] == 'FLUOR']
df_SPR = df[df["assay_type"] == "SPR"]
df_FRET_fluor_SPR = df

os.makedirs(f"{pathway_dir}/ML_data_sciences/1-preprocessing/2-figures/", exist_ok=True)

path = f"{pathway_dir}/ML_data_sciences/1-preprocessing/2-figures/"

df_FRET_dropduplic_reg = df_FRET.dropna(subset=["pIC50_updated"]).sort_values(by=["pIC50_updated"], ascending=False).drop_duplicates(subset=["SMILES"], keep="first")
df_fluor_dropduplic_reg = df_fluor.dropna(subset=["pIC50_updated"]).sort_values(by=["pIC50_updated"], ascending=False).drop_duplicates(subset=["SMILES"], keep="first")
df_SPR_dropduplic_reg = df_SPR.dropna(subset=["pIC50_updated"]).sort_values(by=["pIC50_updated"], ascending=False).drop_duplicates(subset=["SMILES"], keep="first")
df_FRET_fluor_SPR_dropduplic_reg = df_FRET_fluor_SPR.dropna(subset=["pIC50_updated"]).sort_values(by=["pIC50_updated"], ascending=False).drop_duplicates(subset=["SMILES"], keep="first")

os.makedirs(f"{pathway_dir}/ML_data_sciences/1-preprocessing/1-datasets/", exist_ok=True)

path = f"{pathway_dir}/ML_data_sciences/1-preprocessing/1-datasets/"

df_FRET_dropduplic_reg.to_csv(f"{path}FRET_dropduplic_regression.tsv", sep="\t", index=False)
df_fluor_dropduplic_reg.to_csv(f"{path}fluor_dropduplic_regression.tsv", sep="\t", index=False)
df_FRET_fluor_SPR_dropduplic_reg.to_csv(f"{path}FRET_fluor_SPR_dropduplic_regression.tsv", sep="\t", index=False)

def classify_pIC50(row):
    if '>' in str(row['IC50_conv']):
        return 'inactive'
    elif pd.notna(row['pIC50_updated']):
        if row['pIC50_updated'] < 5:
            return 'inactive'
       else:
            return 'active'
    else:
        return None  

df["pIC50_class"] = df.apply(classify_pIC50, axis=1)

df_FRET = df[df["assay_type"] == "FRET"]
df_fluor = df[df["assay_type"] == 'FLUOR']
df_SPR = df[df["assay_type"] == "SPR"]
df_FRET_fluor_SPR = df

df_FRET_dropduplic_class = df_FRET.dropna(subset=["pIC50_class"]).sort_values(by=["pIC50_updated"], ascending=False).drop_duplicates(subset=["SMILES"], keep="first")
df_fluor_dropduplic_class = df_fluor.dropna(subset=["pIC50_class"]).sort_values(by=["pIC50_updated"], ascending=False).drop_duplicates(subset=["SMILES"], keep="first")
df_SPR_dropduplic_class = df_SPR.dropna(subset=["pIC50_class"]).sort_values(by=["pIC50_updated"], ascending=False).drop_duplicates(subset=["SMILES"], keep="first")
df_FRET_fluor_SPR_dropduplic_class = df_FRET_fluor_SPR.dropna(subset=["pIC50_class"]).sort_values(by=["pIC50_updated"], ascending=False).drop_duplicates(subset=["SMILES"], keep="first")

df_FRET_dropduplic_class.to_csv(f"{path}FRET_dropduplic_class.tsv", sep="\t", index=False)
df_fluor_dropduplic_class.to_csv(f"{path}fluor_dropduplic_class.tsv", sep="\t", index=False)
df_FRET_fluor_SPR_dropduplic_class.to_csv(f"{path}FRET_fluor_SPR_dropduplic_class.tsv", sep="\t", index=False)
