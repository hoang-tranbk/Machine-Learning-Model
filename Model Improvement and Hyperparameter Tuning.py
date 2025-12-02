# =====================================
# NÂNG CAO MÔ HÌNH & TỐI ƯU THAM SỐ
# =====================================

# Import thư viện
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Tải dữ liệu Dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Xử lý dữ liệu
df = df.drop(columns=["Name", "Ticket", "Cabin"])
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Mã hóa dữ liệu dạng chuỗi
df = pd.get_dummies(df, drop_first=True)

# Tách biến đầu vào & đầu ra
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================================
# Huấn luyện các mô hình nâng cao
# ==========================================================

# 1️⃣ Logistic Regression (baseline)
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_log = logreg.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)

# 2️⃣ Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# 3️⃣ Gradient Boosting
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)

# 4️⃣ XGBoost
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

# ==========================================================
# So sánh kết quả
# ==========================================================
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost'],
    'Accuracy': [acc_log, acc_rf, acc_gb, acc_xgb]
})

print("\n📊 KẾT QUẢ SO SÁNH CÁC MÔ HÌNH:")
print(results)

# Biểu đồ so sánh
plt.figure(figsize=(7,4))
sns.barplot(x='Model', y='Accuracy', data=results)
plt.title('So sánh độ chính xác các mô hình - Tuần 4')
plt.ylim(0.7, 1)
plt.xticks(rotation=20)
plt.show()

# ==========================================================
# Tối ưu tham số (ví dụ với RandomForest)
# ==========================================================
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("\n✅ Tham số tốt nhất (Random Forest):", grid.best_params_)
print("🎯 Accuracy (tốt nhất sau tuning):", round(grid.best_score_, 4))

# ==========================================================
# Đánh giá chi tiết mô hình tốt nhất
# ==========================================================
best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test)

print("\n📋 Báo cáo chi tiết mô hình tốt nhất:")
print(classification_report(y_test, y_pred_best))

# Confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Mô hình tốt nhất (Random Forest Tuning)")
plt.xlabel("Dự đoán")
plt.ylabel("Thực tế")
plt.show()

