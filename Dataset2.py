# Cài đặt thư viện
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Tải dataset
df = sns.load_dataset("titanic")

# Xem vài dữ liệu đầu
df.head()
# Kích thước dữ liệu
print("Kích thước:", df.shape)

# Thông tin dữ liệu
print(df.info())

# Thống kê mô tả
print(df.describe(include="all"))

# Kiểm tra missing values
print(df.isnull().sum())
# Biểu đồ
plt.figure(figsize=(6,4))
sns.histplot(df["age"].dropna(), bins=30, kde=True)
plt.title("Phân phối tuổi")
plt.show()

# Tỷ lệ
sns.countplot(x="survived", data=df)
plt.title("Tỷ lệ sống/chết")
plt.show()

# Tương quan giữa các biến số
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Ma trận tương quan")
plt.show()
# Điền giá trị thiếu bằng median
df["age"] = df["age"].fillna(df["age"].median())

# Điền giá trị thiếu bằng mode
df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])
# Label Encoding cho các cột phân loại
label_cols = ["sex", "embarked", "class", "who", "adult_male", "alone"]
for col in label_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Chuẩn hóa dữ liệu số
scaler = StandardScaler()
df[["age", "fare"]] = scaler.fit_transform(df[["age", "fare"]])
# Tách dữ liệu
X = df.drop(columns=["survived"])
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Kích thước train:", X_train.shape)
print("Kích thước test:", X_test.shape)

