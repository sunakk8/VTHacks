import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")

#drop useless metrics
drop_cols = [0,1,2,3,4,5,6,8,12,24,25,30]
df.drop(df.columns[drop_cols],axis=1,inplace=True)

df.to_csv('cleaned_data.csv', index=False)  

