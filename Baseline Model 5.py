best_model = results.iloc[0]
print(f"✅ Mô hình tốt nhất: {best_model['Model']} (Accuracy = {best_model['Accuracy']:.4f})")

# Lưu mô hình để dùng lại tuần 4
import joblib
joblib.dump(rf, "baseline_model.pkl")
print("Đã lưu mô hình baseline (Random Forest).")
