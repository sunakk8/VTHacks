import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('cleaned_data.csv')

# Select all columns except the last one for X
X = df.iloc[:, :-1]

# Target variable (Profession)
Y = df['Profession']

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, train_size=0.8, shuffle=True)


print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"Y_test shape: {Y_test.shape}")


