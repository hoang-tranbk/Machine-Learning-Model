import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

#  Tải dữ liệu
# X_train, X_test, y_train, y_test, load
df = sns.load_dataset("titanic")

# Tiền xử lý
df = df.drop(columns=["deck"])
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

# Mã hóa nhãn
label_cols = ["sex", "embarked", "class", "who", "adult_male", "alone"]
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Chuẩn hóa dữ liệu số
scaler = StandardScaler()
df[["age", "fare"]] = scaler.fit_transform(df[["age", "fare"]])

# Chia dữ liệu train/test
X = df.drop(columns=["survived"])
y = df["survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

