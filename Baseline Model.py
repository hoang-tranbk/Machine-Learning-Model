# ==============================
# Import thư viện
# ==============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Xem 5 dòng đầu
print("📊 Dữ liệu ban đầu:")
print(df.head())

# ==============================
#Xử lý dữ liệu
# ==============================

# Xóa các cột không cần thiết
df = df.drop(columns=["Name", "Ticket", "Cabin"])

# Xử lý giá trị khuyết (nếu có)
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Mã hóa các cột dạng chuỗi (object)
df = pd.get_dummies(df, drop_first=True)

# ==============================
#Tách dữ liệu train/test
# ==============================
X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# Huấn luyện Logistic Regression
# ==============================
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# ==============================
# Dự đoán và đánh giá
# ==============================
y_pred = logreg.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\n🎯 Độ chính xác (Accuracy):", round(acc * 100, 2), "%")

print("\n📉 Ma trận nhầm lẫn (Confusion Matrix):")
print(confusion_matrix(y_test, y_pred))

print("\n📋 Báo cáo chi tiết:")
print(classification_report(y_test, y_pred))

