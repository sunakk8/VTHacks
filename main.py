import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv('cleaned_data.csv')


# Specify which columns to one-hot encode
categorical_columns = ['CategoricalColumn1', 'CategoricalColumn2']
target_column = 'Profession'

# Separate the categorical columns and the rest of the DataFrame
df_categorical = df[categorical_columns]
df_numerical = df.drop(columns=categorical_columns + [target_column])

# Apply one-hot encoding to the categorical columns
df_categorical_encoded = pd.get_dummies(df_categorical, drop_first=True)

# Concatenate the encoded categorical columns with the numerical columns
df_encoded = pd.concat([df_numerical, df_categorical_encoded], axis=1)

# Split the features and target
Y = df[target_column]

# Select all columns except the last one for X
X = df.iloc[:, :-1]

# Target variable (Profession)

# Splitting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, train_size=0.8, shuffle=True)


print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"Y_test shape: {Y_test.shape}")




