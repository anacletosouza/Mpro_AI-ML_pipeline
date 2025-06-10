import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.BRICS import FindBRICSBonds, BreakBRICSBonds
from rdkit.Chem.rdmolops import GetMolFrags
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from itertools import combinations

df = pd.read_csv("/home/anacleto/Desktop/manuscript_Mpro/naturecomm/supplementary_file_S2/6-chemical-library/1-final_dataset.csv", sep=",")
df = df[df["SMILES"].notnull()]
df["SMILES"] = df["SMILES"].astype(str)

def is_valid_fragment(smiles, min_atoms=5):
    """Check if fragment has at least min_atoms atoms and meets other criteria"""
    if "*" in smiles or "." in smiles:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.GetNumAtoms() >= min_atoms

def fragment_row(row, min_atoms=5):
    smiles = row["SMILES"]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    try:
        bonds = list(FindBRICSBonds(mol))
        fragments = set()
        
        # Strategy 1: BRICS fragmentation
        if bonds:
            for i in range(1, min(3, len(bonds)) + 1):
                for bond_comb in combinations(bonds, i):
                    try:
                        frag_mol = BreakBRICSBonds(mol, bond_comb)
                        frags = GetMolFrags(frag_mol, asMols=True)
                        for frag in frags:
                            frag_smi = Chem.MolToSmiles(frag, isomericSmiles=True)
                            if is_valid_fragment(frag_smi, min_atoms):
                                fragments.add(frag_smi)
                    except:
                        continue
        
        # Strategy 2: Single bond cleavage
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                try:
                    em = Chem.EditableMol(mol)
                    em.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                    modified_mol = em.GetMol()
                    frags = GetMolFrags(modified_mol, asMols=True)
                    if len(frags) > 1:
                        for frag in frags:
                            frag_smi = Chem.MolToSmiles(frag, isomericSmiles=True)
                            if is_valid_fragment(frag_smi, min_atoms) and frag_smi != smiles:
                                fragments.add(frag_smi)
                except:
                    continue
        
        return [{
            "frag_smiles": frag,  # Will be renamed to "SMILES" later
            "SMILES": smiles,      # Will be renamed to "origin_SMILES" later
            **row.drop(labels=["SMILES"]).to_dict()
        } for frag in fragments]
    
    except Exception as e:
        print(f"Error processing {smiles}: {str(e)}")
        return []

rows = [row for _, row in df.iterrows()]
fragment_data = []

with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    futures = [executor.submit(fragment_row, row) for row in rows]
    for future in as_completed(futures):
        fragment_data.extend(future.result())

df_frag = pd.DataFrame(fragment_data)
df_frag = df_frag.drop_duplicates(subset=["frag_smiles"])

# Final filter to ensure minimum size (redundant but safe)
df_frag = df_frag[df_frag["frag_smiles"].apply(lambda x: is_valid_fragment(x, min_atoms=5))]

# Column renaming and new sequential numbering
df_frag = df_frag.rename(columns={
    "frag_smiles": "SMILES",
    "SMILES": "origin_SMILES",
    "name": "origin_mol_name"
})

# Add new sequential 'name' column
df_frag.insert(0, "name", range(1, len(df_frag) + 1))

# Reorder columns
cols = df_frag.columns.tolist()
new_order = ["SMILES","name", "origin_mol_name", "origin_SMILES"] + \
           [col for col in cols if col not in ["SMILES","name", "origin_mol_name", "origin_SMILES"]]
df_frag = df_frag[new_order]

print(f"Total unique fragments generated (≥5 atoms): {len(df_frag)}")
print(df_frag.head())

