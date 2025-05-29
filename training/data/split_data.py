"""Python script to split the stroke dataset into two parts for training and batch predicting."""

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Split the data randomly into two equal parts
df1, df2 = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
df2, df3 = train_test_split(df2, test_size=0.5, random_state=42, shuffle=True)

# Save the splits
df1.to_csv("stroke-data-1.csv", index=False)
df2.to_csv("stroke-data-2.csv", index=False)
df3.to_csv("stroke-data-3.csv", index=False)
