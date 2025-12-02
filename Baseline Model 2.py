# Khởi tạo và huấn luyện
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Dự đoán
y_pred_lr = logreg.predict(X_test)

# Đánh giá
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"🎯 Logistic Regression Accuracy: {acc_lr:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# Ma trận nhầm lẫn
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
