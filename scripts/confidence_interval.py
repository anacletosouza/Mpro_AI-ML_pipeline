import pandas as pd
import numpy as np
from scipy import stats

def compute_descriptor_statistics(df):
    # Copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Extract only the compound columns (exclude 'mol_desc')
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
        t_crit = stats.t.ppf(0.975, df=n-1)  # Student's t critical value
        margin_error = t_crit * sd / np.sqrt(n)
        CI95_lower = mean - margin_error
        CI95_upper = mean + margin_error

        results.append({
            'mol_desc': descriptor,
            'IC95_inf': CI95_lower,
            'IC95_sup': CI95_upper,
            'mean': mean,
            'sd': sd
        })

    # Convert results into a new DataFrame
    result_df = pd.DataFrame(results)
    return result_df
