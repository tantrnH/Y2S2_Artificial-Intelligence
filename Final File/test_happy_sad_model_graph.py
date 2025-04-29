import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

print("\n[INFO] Loading features and labels...")

# 加载全部特征
X = np.load('X.npy')
y = np.load('y.npy')
print(f"Data shape: X = {X.shape}, y = {y.shape}")

# 只保留 happy 和 sad
happy_sad_mask = np.isin(y, ['happy', 'sad'])
X = X[happy_sad_mask]
y = y[happy_sad_mask]

# 将标签映射为数字
label_mapping = {"happy": 0, "sad": 1}
y_numeric = np.array([label_mapping[label] for label in y])

print("\n[INFO] Splitting train/test set...")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_numeric, test_size=0.2, random_state=42)

print("\n[INFO] Loading the saved model...")

# 加载保存的模型
model = joblib.load('svm_happy_vs_sad.pkl')

print("\n[INFO] Predicting...")

# 预测
y_pred = model.predict(X_test)

# 输出准确率
print("\n✨ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# 输出详细分类报告
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["happy", "sad"]))

# 画出混淆矩阵
print("\n[INFO] Plotting Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["happy", "sad"], yticklabels=["happy", "sad"])
plt.title("Confusion Matrix for Happy vs Sad Emotion Recognition")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

print("\n[INFO] Done!")