import numpy as np
from scipy.stats import t

def t_student_interval(df, n):
    df = df.copy()
    t_critical = t.ppf(0.975, df=n - 1)
    df["t_critical"] = t_critical
    df["IC95_inf"] = df["mean"] - t_critical * df["sd"] / np.sqrt(df["n"])
    df["IC95_sup"] = df["mean"] + t_critical * df["sd"] / np.sqrt(df["n"])
    df["interval_length"] = df["IC95_sup"] - df["IC95_inf"]
    return df

