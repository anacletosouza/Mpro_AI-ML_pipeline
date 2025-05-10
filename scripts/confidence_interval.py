import pandas as pd
import numpy as np
from scipy import stats

def compute_descriptor_statistics(df, decimals=6):
    # Copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Identify value columns (exclude 'mol_desc')
    value_columns = df_copy.columns.drop('mol_desc')

    results = []

    # Iterate over each descriptor (row)
    for _, row in df_copy.iterrows():
        descriptor = row['mol_desc']
        values = row[value_columns].astype(float).values

        mean = np.mean(values)
        sd = np.std(values, ddof=1)  # sample standard deviation
        n = len(values)

        # Compute 95% confidence interval
        t_crit = stats.t.ppf(0.975, df=n-1)
        margin_error = t_crit * sd / np.sqrt(n)
        CI95_lower = mean - margin_error
        CI95_upper = mean + margin_error

        # Append rounded results (no scientific notation)
        results.append({
            'mol_desc': descriptor,
            'IC95_inf': round(CI95_lower, decimals),
            'IC95_sup': round(CI95_upper, decimals),
            'mean': round(mean, decimals),
            'sd': round(sd, decimals)
        })

    # Convert results into a new DataFrame
    result_df = pd.DataFrame(results)
    return result_df
