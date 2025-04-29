import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.utils import shuffle

# 1. 载入特征和标签
X = np.load('X.npy')
y = np.load('y.npy')


# 2. 选择需要的两种情绪（比如 "happy" 和 "sad"）
# 假设你的标签是字符串，例如 ["happy", "sad", "angry", ...]
# 如果你的标签是数字，再告诉我，我帮你改！

# 选定你要的情绪类别
target_classes = ['happy', 'sad']

# 筛选出 happy 和 sad 的数据
selected_indices = np.where(np.isin(y, target_classes))[0]
X_selected = X[selected_indices]
y_selected = y[selected_indices]

# 3. 将类别转为数字编码（happy -> 0, sad -> 1）
label_mapping = {'happy': 0, 'sad': 1}
y_encoded = np.array([label_mapping[label] for label in y_selected])

# 4. 平衡样本数量
# 分开 happy 和 sad
happy_indices = np.where(y_encoded == 0)[0]
sad_indices = np.where(y_encoded == 1)[0]

# 取最小数量
min_samples = min(len(happy_indices), len(sad_indices))

# 重新组合平衡的数据
happy_indices = np.random.choice(happy_indices, min_samples, replace=False)
sad_indices = np.random.choice(sad_indices, min_samples, replace=False)

balanced_indices = np.concatenate((happy_indices, sad_indices))
X_balanced = X_selected[balanced_indices]
y_balanced = y_encoded[balanced_indices]

# 打乱数据
X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=42)

# 5. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# 6. 训练分类器（用 SVM 作为例子）
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# 7. 保存训练好的模型
model_filename = 'svm_happy_vs_sad.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved as {model_filename}")