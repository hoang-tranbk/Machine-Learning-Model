import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# ======================================================
# TẢI DỮ LIỆU VÀ TIỀN XỬ LÝ GIỐNG HỆT TUẦN 5
# ======================================================
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Giữ nguyên tất cả các cột trong lúc fit scaler
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Loại bỏ các cột không dùng cho mô hình
df = df.drop(columns=["Name", "Ticket", "Cabin"])
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=["Survived"])
y = df["Survived"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================================
# TẢI LẠI MÔ HÌNH TUẦN 5
# ======================================================
model = keras.models.load_model("titanic_deep_learning_model.keras")
print("✅ Mô hình đã được tải thành công!")

# ======================================================
# HÀM DỰ ĐOÁN
# ======================================================
def predict_survival(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked_C, Embarked_Q):
    """
    Hàm dự đoán khả năng sống sót (0 hoặc 1)
    """
    # Phải đảm bảo tất cả cột giống X.columns khi fit scaler
    cols = X.columns

    # Tạo hàng dữ liệu giả lập đủ cột
    data = pd.DataFrame([[0]*len(cols)], columns=cols)

    # Gán các giá trị thật
    data.loc[0, "PassengerId"] = 9999  # giả lập ID
    data.loc[0, "Pclass"] = Pclass
    data.loc[0, "Age"] = Age
    data.loc[0, "SibSp"] = SibSp
    data.loc[0, "Parch"] = Parch
    data.loc[0, "Fare"] = Fare
    data.loc[0, "Sex_male"] = 1 if Sex == "male" else 0
    data.loc[0, "Embarked_Q"] = Embarked_Q
    data.loc[0, "Embarked_S"] = Embarked_C

    # Chuẩn hóa theo scaler đã fit
    data_scaled = scaler.transform(data)

    # Dự đoán
    prob = model.predict(data_scaled, verbose=0)[0][0]
    return f"{prob:.2%} khả năng sống sót" if prob >= 0.5 else f"{(1 - prob):.2%} khả năng không sống sót"


# ======================================================
# KIỂM TRA VỚI HÀNH KHÁCH MỚI
# ======================================================
print("\n🔍 Ví dụ dự đoán cho một hành khách mới:")
print("----------------------------------------------------")
print(predict_survival(
    Pclass=1,
    Sex="female",
    Age=25,
    SibSp=0,
    Parch=0,
    Fare=120,
    Embarked_C=0,
    Embarked_Q=1
))

