import pandas as pd
import numpy as np
from scipy import stats

def calcular_estatisticas(df):
    # Copia para evitar alterações no original
    df_copy = df.copy()

    # Extrai nomes das colunas que contêm os valores numéricos (exclui a coluna 'mol_desc')
    colunas_valores = df_copy.columns.drop('mol_desc')

    # Lista para armazenar os resultados
    resultados = []

    # Itera sobre cada linha (cada descritor)
    for _, row in df_copy.iterrows():
        desc = row['mol_desc']
        valores = row[colunas_valores].astype(float).values

        mean = np.mean(valores)
        sd = np.std(valores, ddof=1)  # desvio padrão amostral
        n = len(valores)
        
        # Intervalo de confiança de 95%
        t_crit = stats.t.ppf(0.975, df=n-1)  # t de Student
        margin_error = t_crit * sd / np.sqrt(n)
        IC95_inf = mean - margin_error
        IC95_sup = mean + margin_error

        resultados.append({
            'mol_desc': desc,
            'IC95_inf': IC95_inf,
            'IC95_sup': IC95_sup,
            'mean': mean,
            'sd': sd
        })

    # Converte a lista de dicionários para DataFrame
    df_resultado = pd.DataFrame(resultados)
    return df_resultado
