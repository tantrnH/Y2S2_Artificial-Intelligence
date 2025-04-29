import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 加载特征和标签
X = np.load('X.npy')
y = np.load('y.npy')

# 只保留 happy 和 sad
happy_sad_mask = np.isin(y, ['happy', 'sad'])
X = X[happy_sad_mask]
y = y[happy_sad_mask]

# 标签数值化
label_mapping = {"happy": 0, "sad": 1}
y_numeric = np.array([label_mapping[label] for label in y])

# 特征归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 切分数据
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_numeric, test_size=0.2, random_state=42)

# 模型列表
models = {
    "Logistic Regression": LogisticRegression(),
    "SVM (Support Vector Machine)": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# 比较各模型Accuracy
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

# 打印
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")