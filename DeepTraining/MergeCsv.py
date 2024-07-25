import numpy as np
import pandas as pd

csv1=pd.read_csv('/homes/nva01/DeepTraining/EmbeddingsNaive.csv')
csv2=pd.read_csv('/homes/nva01/DeepTraining/EmbeddingsNaiveT.csv')
df = pd.concat([csv1, csv2])
print(csv1.shape, csv2.shape, df.shape)
print(len(df))
df.to_csv('/homes/nva01/DeepTraining/TotalEmbeddingsNaive.csv', index=False)