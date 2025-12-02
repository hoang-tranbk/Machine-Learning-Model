# ===============================================================
# DEEP LEARNING PIPELINE - MLP CHO DỮ LIỆU DẠNG BẢNG
# ===============================================================

# Import thư viện
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ===============================================================
# Tải và xử lý dữ liệu Titanic
# ===============================================================
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Xóa cột không cần thiết
df = df.drop(columns=["Name", "Ticket", "Cabin"])

# Điền giá trị khuyết thiếu
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Mã hóa dữ liệu dạng chuỗi
df = pd.get_dummies(df, drop_first=True)

# Chia dữ liệu
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu đầu vào
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================================================
# Xây dựng mô hình MLP (Multilayer Perceptron)
# ===============================================================
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ===============================================================
# Huấn luyện mô hình
# ===============================================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# ===============================================================
# Đánh giá kết quả
# ===============================================================
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Độ chính xác trên tập test: {acc:.4f}")

# Dự đoán
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).ravel()

# Báo cáo chi tiết
print("\n📋 Báo cáo đánh giá mô hình Deep Learning:")
print(classification_report(y_test, y_pred))

# Ma trận nhầm lẫn
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - MLP (Deep Learning)")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.show()

# ===============================================================
# Biểu đồ Loss và Accuracy
# ===============================================================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Biểu đồ Loss qua các Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Biểu đồ Accuracy qua các Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ===============================================================
# Bước 7. Lưu mô hình (tùy chọn)
# ===============================================================
model.save("titanic_deep_learning_model.keras")
print("\n✅ Đã lưu mô hình thành công: titanic_deep_learning_model.keras")

