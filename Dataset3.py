# Sao lưu dữ liệu
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
print("Đã lưu dữ liệu train/test sau tiền xử lý.")
