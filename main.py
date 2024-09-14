import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")





#drop useless metrics
drop_cols = [0,1,2,3,4,5,6,8,12,24,25,30]
df.drop(df.columns[drop_cols],axis=1,inplace=True)

column_list = df.columns.to_list()

for i in range(len(column_list)):
    print(i, ": ", column_list[i])


pd.set_option('display.max_columns', 500)

df.to_csv('cleaned_data.csv', index=False)  

