import pandas as pd

df = pd.read_csv(
    "data/Allstorms.ibtracs_wmo.v03r05.csv",
    delim_whitespace=False)

print(df.head())
