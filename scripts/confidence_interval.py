from scipy.stats import t

def t_student_interval(df):
  df["t_critical"] = df["n"].apply(lambda n: t.ppf(0.975, df=n-1))
  df["IC95_inf"] = df["mean"] - df["t_critical"] * df["sd"] / np.sqrt(df["n"])
  df["IC95_sup"] = df["mean"] + df["t_critical"] * df["sd"] / np.sqrt(df["n"])
  df["interval_length"] = df["IC95_sup"] - df["IC95_inf"]
  return df
