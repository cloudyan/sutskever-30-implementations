# 阶段 3：机器学习基础

_从数据中学习模式与预测_

---

## 📖 学习指南

**前置知识**：
- ✅ Python 编程基础
- ✅ NumPy、Pandas
- ✅ 线性代数、概率论基础

**学习目标**：
- ✅ 理解机器学习的基本概念
- ✅ 掌握监督学习核心算法
- ✅ 掌握无监督学习核心算法
- ✅ 理解特征工程与模型选择
- ✅ 能独立完成机器学习项目

**预计时间**：45 天

---

## 3.1 机器学习概览

### 什么是机器学习？

<div class="formula-box">

```
定义：
机器学习 = 从数据中学习模式，用于预测或决策

传统编程：
规则 + 数据 → 答案

机器学习：
数据 + 答案 → 规则（模型）
```

</div>

### 机器学习类型

<div class="formula-box">

```
1. 监督学习（Supervised Learning）
   输入：带标签的数据 (X, y)
   任务：分类、回归
   算法：线性回归、逻辑回归、决策树、SVM...

2. 无监督学习（Unsupervised Learning）
   输入：无标签的数据 X
   任务：聚类、降维
   算法：K-Means、PCA、自编码器...

3. 半监督学习
   输入：少量标签 + 大量无标签
   任务：利用无标签数据提升性能

4. 强化学习
   输入：环境反馈（奖励/惩罚）
   任务：学习最优策略
   算法：Q-Learning、Policy Gradient...
```

</div>

### 机器学习工作流程

<div class="formula-box">

```
1. 问题定义
   ↓
2. 数据收集
   ↓
3. 数据探索与可视化
   ↓
4. 数据预处理
   ↓
5. 特征工程
   ↓
6. 模型选择
   ↓
7. 模型训练
   ↓
8. 模型评估
   ↓
9. 超参数调优
   ↓
10. 部署与监控
```

</div>

---

## 3.2 监督学习

### 线性回归

<div class="formula-box">

```python
# 模型
y = w·X + b

# 损失函数（MSE）
L(w, b) = (1/n) Σ(yᵢ - ŷᵢ)²

# 梯度下降
w = w - η·∂L/∂w
b = b - η·∂L/∂b

# scikit-learn 实现
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

</div>

### 逻辑回归（分类）

<div class="formula-box">

```python
# Sigmoid 函数
σ(z) = 1 / (1 + exp(-z))

# 模型
P(y=1|X) = σ(w·X + b)

# 损失函数（交叉熵）
L(w, b) = -Σ[yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]

# scikit-learn 实现
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

</div>

### 决策树

<div class="formula-box">

```
决策树结构：
if 年龄 > 30?
  ├─ 是：if 收入 > 50k?
  │       ├─ 是：购买
  │       └─ 否：不购买
  └─ 否：不购买

分裂标准：
- 分类：信息增益、基尼系数
- 回归：方差减少
```

</div>

<div class="formula-box">

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5,           # 最大深度
    min_samples_split=10,  # 最小分裂样本数
    criterion='gini'       # 分裂标准
)
model.fit(X_train, y_train)
```

</div>

### 随机森林

<div class="formula-box">

```
随机森林 = 多棵决策树的集成

Bagging 思想：
1. Bootstrap 采样（有放回抽样）
2. 每棵树用不同的样本子集训练
3. 随机选择特征子集
4. 投票（分类）或平均（回归）

优势：
- 减少过拟合
- 提高泛化能力
- 可并行训练
```

</div>

<div class="formula-box">

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,      # 树的数量
    max_depth=10,
    n_jobs=-1              # 并行训练
)
model.fit(X_train, y_train)
```

</div>

### 支持向量机（SVM）

<div class="formula-box">

```
核心思想：
找到最大间隔的超平面

优化问题：
min ||w||²
s.t. yᵢ(w·xᵢ + b) ≥ 1

核技巧：
线性不可分 → 映射到高维 → 线性可分

常用核函数：
- 线性核：K(x, x') = x·x'
- 多项式核：K(x, x') = (γx·x' + r)^d
- RBF 核：K(x, x') = exp(-γ||x-x'||²)
```

</div>

<div class="formula-box">

```python
from sklearn.svm import SVC

model = SVC(
    kernel='rbf',          # 核函数
    C=1.0,                 # 正则化参数
    gamma='scale'          # 核系数
)
model.fit(X_train, y_train)
```

</div>

### K 近邻（KNN）

<div class="formula-box">

```
核心思想：
近朱者赤，近墨者黑

算法：
1. 计算测试点与所有训练点的距离
2. 选择最近的 K 个邻居
3. 投票（分类）或平均（回归）

距离度量：
- 欧氏距离：√Σ(xᵢ-yᵢ)²
- 曼哈顿距离：Σ|xᵢ-yᵢ|
- 闵可夫斯基距离：(Σ|xᵢ-yᵢ|^p)^(1/p)
```

</div>

<div class="formula-box">

```python
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=5,         # K 值
    metric='euclidean'     # 距离度量
)
model.fit(X_train, y_train)
```

</div>

### 朴素贝叶斯

<div class="formula-box">

```
贝叶斯定理：
P(y|X) = P(X|y)P(y) / P(X)

朴素假设：
特征之间条件独立

P(X|y) = P(x₁|y) × P(x₂|y) × ... × P(xₙ|y)

变体：
- 高斯朴素贝叶斯：连续特征
- 多项式朴素贝叶斯：离散特征（文本分类）
- 伯努利朴素贝叶斯：二元特征
```

</div>

<div class="formula-box">

```python
from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
```

</div>

---

## 3.3 无监督学习

### K-Means 聚类

<div class="formula-box">

```
算法流程：
1. 随机初始化 K 个聚类中心
2. 分配每个点到最近的聚类中心
3. 更新聚类中心为均值
4. 重复 2-3 直到收敛

优化目标：
min Σ||xᵢ - μⱼ||²

K 值选择：
- 肘部法则（Elbow Method）
- 轮廓系数（Silhouette Score）
```

</div>

<div class="formula-box">

```python
from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=5,          # K 值
    init='k-means++',      # 初始化方法
    n_init=10              # 运行次数
)
model.fit(X)
labels = model.labels_
```

</div>

### 层次聚类

<div class="formula-box">

```
两种方法：
1. 凝聚（自底向上）
   每个点是一个聚类 → 逐步合并

2. 分裂（自顶向下）
   所有点是一个聚类 → 逐步分裂

距离度量：
- 单链接：最近邻距离
- 全链接：最远邻距离
- 平均链接：平均距离
- Ward 链接：方差最小化
```

</div>

### DBSCAN

<div class="formula-box">

```
基于密度的聚类

核心概念：
- 核心点：ε半径内至少有 min_samples 个点
- 边界点：在核心点的ε半径内
- 噪声点：既不是核心点也不是边界点

优势：
- 不需要指定聚类数
- 能发现任意形状的聚类
- 能识别噪声点

参数：
- eps (ε): 邻域半径
- min_samples: 最小样本数
```

</div>

<div class="formula-box">

```python
from sklearn.cluster import DBSCAN

model = DBSCAN(
    eps=0.5,               # ε半径
    min_samples=5          # 最小样本数
)
labels = model.fit_predict(X)
```

</div>

### 主成分分析（PCA）

<div class="formula-box">

```
目标：降维，保留最大方差

步骤：
1. 数据中心化
2. 计算协方差矩阵
3. 特征分解
4. 选择前 k 个特征向量
5. 投影到低维空间

数学：
最大化方差：max Var(XW)
等价于：求协方差矩阵的特征向量
```

</div>

<div class="formula-box">

```python
from sklearn.decomposition import PCA

pca = PCA(
    n_components=2,        # 降维到 2 维
    svd_solver='full'
)
X_reduced = pca.fit_transform(X)

# 解释方差比
print(pca.explained_variance_ratio_)
```

</div>

### 自编码器（Autoencoder）

<div class="formula-box">

```
结构：
输入 → 编码器 → 潜在表示 → 解码器 → 输出

目标：
最小化重构误差：min ||X - X'||²

应用：
- 降维
- 去噪
- 异常检测
- 生成模型（VAE）
```

</div>

---

## 3.4 特征工程

### 特征选择

<div class="formula-box">

```python
from sklearn.feature_selection import SelectKBest, f_classif

# 方差选择
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.1)

# 单变量选择
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 递归特征消除
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=10)
X_selected = rfe.fit_transform(X, y)
```

</div>

### 特征提取

<div class="formula-box">

```python
# 文本特征：TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)

# 图像特征：HOG
from skimage.feature import hog

features = hog(image, orientations=9, pixels_per_cell=(8, 8))

# 类别特征：One-Hot 编码
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical)
```

</div>

### 特征缩放

<div class="formula-box">

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 标准化（Z-score）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# 结果：均值=0，方差=1

# 归一化（Min-Max）
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
# 结果：范围 [0, 1]
```

</div>

### 处理缺失值

<div class="formula-box">

```python
from sklearn.impute import SimpleImputer

# 均值填充
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 中位数填充
imputer = SimpleImputer(strategy='median')

# 众数填充
imputer = SimpleImputer(strategy='most_frequent')

# 常数填充
imputer = SimpleImputer(strategy='constant', fill_value=0)
```

</div>

---

## 3.5 模型评估

### 分类指标

<div class="formula-box">

```
混淆矩阵：
              预测
           正例  负例
实际 正例   TP   FN
     负例   FP   TN

准确率：Accuracy = (TP+TN) / (TP+TN+FP+FN)
精确率：Precision = TP / (TP+FP)
召回率：Recall = TP / (TP+FN)
F1 分数：F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

</div>

<div class="formula-box">

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print(f"准确率：{accuracy_score(y_true, y_pred)}")
print(f"精确率：{precision_score(y_true, y_pred)}")
print(f"召回率：{recall_score(y_true, y_pred)}")
print(f"F1 分数：{f1_score(y_true, y_pred)}")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
```

</div>

### ROC 曲线与 AUC

<div class="formula-box">

```
ROC 曲线：
- X 轴：假阳性率（FPR）
- Y 轴：真阳性率（TPR）
- 曲线下的面积 = AUC

AUC 含义：
- 0.5：随机猜测
- 0.7-0.8：可接受
- 0.8-0.9：良好
- 0.9+：优秀
```

</div>

<div class="formula-box">

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# 绘图
import matplotlib.pyplot as plt

plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
```

</div>

### 回归指标

<div class="formula-box">

```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# 均方误差（MSE）
mse = mean_squared_error(y_true, y_pred)

# 均方根误差（RMSE）
rmse = np.sqrt(mse)

# 平均绝对误差（MAE）
mae = mean_absolute_error(y_true, y_pred)

# R²分数
r2 = r2_score(y_true, y_pred)
```

</div>

---

## 3.6 模型选择与调优

### 交叉验证

<div class="formula-box">

```
K 折交叉验证：
1. 数据分成 K 份
2. 轮流用 K-1 份训练，1 份验证
3. 平均 K 次的结果

优点：
- 更可靠的性能估计
- 充分利用数据
- 减少方差
```

</div>

<div class="formula-box">

```python
from sklearn.model_selection import cross_val_score, KFold

# K 折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

print(f"平均准确率：{scores.mean():.3f} (+/- {scores.std():.3f})")
```

</div>

### 网格搜索

<div class="formula-box">

```python
from sklearn.model_selection import GridSearchCV

# 参数网格
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# 网格搜索
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X, y)

# 最佳参数
print(f"最佳参数：{grid_search.best_params_}")
print(f"最佳分数：{grid_search.best_score_:.3f}")
```

</div>

### 随机搜索

<div class="formula-box">

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform

# 参数分布
param_dist = {
    'C': loguniform(0.1, 10),
    'gamma': loguniform(0.01, 1),
    'kernel': ['linear', 'rbf']
}

# 随机搜索
random_search = RandomizedSearchCV(
    SVC(),
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X, y)
```

</div>

---

## 3.7 实战项目

### 项目 1：房价预测

<div class="formula-box">

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. 加载数据
df = pd.read_csv('housing.csv')

# 2. 特征工程
X = df[['size', 'bedrooms', 'bathrooms', 'age']]
y = df['price']

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 预测与评估
y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# 6. 特征重要性
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
```

</div>

### 项目 2：垃圾邮件分类

<div class="formula-box">

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# 1. 加载数据
emails = [...]  # 邮件文本
labels = [...]  # 0=正常，1=垃圾

# 2. 构建管道
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', MultinomialNB())
])

# 3. 交叉验证
scores = cross_val_score(model, emails, labels, cv=5, scoring='f1')
print(f"平均 F1 分数：{scores.mean():.3f}")

# 4. 训练与预测
model.fit(emails, labels)
predictions = model.predict(new_emails)
```

</div>

### 项目 3：客户分群

<div class="formula-box">

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. 加载数据
df = pd.read_csv('customers.csv')
X = df[['age', 'income', 'spending_score']]

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 确定 K 值（肘部法则）
inertias = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertias.append(model.inertia_)

plt.plot(range(1, 11), inertias, 'bo-')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# 4. 聚类
model = KMeans(n_clusters=5, random_state=42)
clusters = model.fit_predict(X_scaled)

# 5. 可视化
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis')
plt.title('Customer Segments')
plt.show()

# 6. 分析每个聚类
df['cluster'] = clusters
print(df.groupby('cluster').mean())
```

</div>

---

## 📚 学习资源

### 课程

- [吴恩达机器学习](https://www.coursera.org/learn/machine-learning)
- [李宏毅机器学习](https://www.youtube.com/playlist?list=PLJV_el3uVTsODxQFgzMzPLa16h6B8kWM_)

### 书籍

- 《机器学习》周志华（西瓜书）
- 《统计学习方法》李航
- 《Hands-On Machine Learning》Aurélien Géron

### 实践平台

- [Kaggle](https://www.kaggle.com/) - 数据科学竞赛
- [UCI ML Repository](https://archive.ics.uci.edu/) - 数据集

---

## ✅ 学习检查清单

- [ ] 理解机器学习基本概念
- [ ] 掌握线性回归与逻辑回归
- [ ] 掌握决策树与随机森林
- [ ] 掌握 SVM 原理
- [ ] 掌握 K-Means 聚类
- [ ] 掌握 PCA 降维
- [ ] 掌握特征工程方法
- [ ] 掌握模型评估指标
- [ ] 掌握交叉验证与网格搜索
- [ ] 完成至少 2 个实战项目

---

*最后更新：2026-04-22*
