import pandas as pd

df1 = pd.read_csv("../data/fold0.csv")
df2 = pd.read_csv("../data/fold1.csv")

df_merged = pd.concat([df1, df2]).groupby(list(df1.columns), as_index=False).sum()

# Save to CSV file
df_merged.to_csv("../data/merged.csv", index=False)
print("merged", df_merged.describe)