import pandas as pd
import numpy as np
from scipy import stats

def compute_descriptor_statistics(df, decimals=2):
    df_copy = df.copy()
    value_columns = df_copy.columns.drop('mol_desc')

    results = []

    for _, row in df_copy.iterrows():
        descriptor = row['mol_desc']
        values = row[value_columns].astype(float).values

        mean = np.mean(values)
        sd = np.std(values, ddof=1)
        n = len(values)

        t_crit = stats.t.ppf(0.975, df=n-1)
        margin_error = t_crit * sd / np.sqrt(n)
        CI95_lower = mean - margin_error
        CI95_upper = mean + margin_error

        # Format as strings with fixed-point notation (no scientific notation)
        results.append({
            'mol_desc': descriptor,
            'IC95_inf': f"{CI95_lower:.{decimals}f}",
            'IC95_sup': f"{CI95_upper:.{decimals}f}",
            'mean': f"{mean:.{decimals}f}",
            'sd': f"{sd:.{decimals}f}"
        })

    result_df = pd.DataFrame(results)
    return result_df
