# Tạo bảng so sánh kết quả
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [acc_lr, acc_rf]
})
results.sort_values(by="Accuracy", ascending=False, inplace=True)

# Hiển thị bảng kết quả
print(results)

# Biểu đồ so sánh
plt.figure(figsize=(6,4))
sns.barplot(x="Model", y="Accuracy", data=results, palette="viridis")
plt.title("So sánh độ chính xác giữa các mô hình Baseline")
plt.ylim(0,1)
plt.show()
