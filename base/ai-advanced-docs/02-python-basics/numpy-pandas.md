# 2.2 科学计算栈：NumPy 与 Pandas

> 数据科学的基石 —— 高效处理数值和表格数据
> 
> _预计时间：3-5 天 | 难度：⭐⭐⭐ | 前置知识：Python 基础_

---

## 📖 内容概览

```
科学计算栈学习路径：

Day 1-2: NumPy 数组操作
Day 3: NumPy 线性代数
Day 4-5: Pandas 数据处理
Day 6: 数据可视化
```

---

## 第一部分：NumPy

### 1. 数组创建

```python
import numpy as np

# 从列表创建
arr = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])

# 特殊数组
zeros = np.zeros((3, 3))      # 全 0
ones = np.ones((2, 4))        # 全 1
full = np.full((2, 2), 7)     # 填充指定值
empty = np.empty((3, 3))      # 未初始化（更快）

# 序列
range_arr = np.arange(0, 10, 2)      # [0, 2, 4, 6, 8]
linspace = np.linspace(0, 1, 5)      # [0, 0.25, 0.5, 0.75, 1]

# 随机数组
rand = np.random.rand(3, 3)          # [0,1) 均匀分布
randn = np.random.randn(3, 3)        # 标准正态分布
randint = np.random.randint(0, 10, (3, 3))  # 随机整数

# 单位矩阵
eye = np.eye(4)
identity = np.identity(3)
```

### 2. 数组属性

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr.ndim)      # 2（维度数）
print(arr.shape)     # (2, 3)（形状）
print(arr.size)      # 6（元素总数）
print(arr.dtype)     # int64（数据类型）
print(arr.itemsize)  # 8（每个元素的字节数）
print(arr.nbytes)    # 48（总字节数）
```

### 3. 数组操作

```python
# 形状变换
arr = np.arange(12)
reshaped = arr.reshape(3, 4)
flattened = reshaped.flatten()
raveled = reshaped.ravel()  # 视图版本

# 转置
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
transposed = arr_2d.T
transposed2 = arr_2d.transpose()

# 拼接
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

vstack = np.vstack([a, b])  # 垂直堆叠
hstack = np.hstack([a, b])  # 水平堆叠
concat_v = np.concatenate([a, b], axis=0)
concat_h = np.concatenate([a, b], axis=1)

# 分割
arr = np.arange(9).reshape(3, 3)
split_v = np.vsplit(arr, 3)  # 垂直分割
split_h = np.hsplit(arr, 3)  # 水平分割

# 添加/删除维度
arr = np.array([1, 2, 3])
expanded = np.expand_dims(arr, axis=0)  # (1, 3)
squeezed = np.squeeze(expanded)         # (3,)
```

### 4. 索引与切片

```python
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

# 基本索引
arr[0, 0]      # 1
arr[1, :]      # [5, 6, 7, 8]
arr[:, 2]      # [3, 7, 11]
arr[0:2, 1:3]  # [[2, 3], [6, 7]]

# 布尔索引
arr = np.array([1, 2, 3, 4, 5, 6])
mask = arr > 3
print(arr[mask])  # [4, 5, 6]
print(arr[arr % 2 == 0])  # [2, 4, 6]

# 花式索引
arr = np.array([10, 20, 30, 40, 50])
indices = [1, 3, 4]
print(arr[indices])  # [20, 40, 50]

# 修改
arr[arr > 3] = 0  # 条件赋值
```

### 5. 广播机制

```python
"""
广播规则：
1. 如果维度数不同，在较小的数组前面补 1
2. 如果某个维度大小相同或其中一个为 1，则兼容
3. 不兼容的维度会报错
"""

# 示例 1：标量 + 数组
arr = np.array([1, 2, 3])
result = arr + 5  # [6, 7, 8]

# 示例 2：一维 + 二维
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_1d = np.array([10, 20, 30])
result = arr_2d + arr_1d  # [[11, 22, 33], [14, 25, 36]]

# 示例 3：列向量 + 行向量
row = np.array([1, 2, 3])     # (3,)
col = np.array([[10], [20]])  # (2, 1)
result = row + col
# [[11, 12, 13],
#  [21, 22, 23]]

# 显式广播
arr = np.array([1, 2, 3])
broadcasted = np.broadcast_to(arr, (3, 3))
```

### 6. 向量化运算

```python
arr = np.array([1, 2, 3, 4, 5])

# 数学运算
np.sqrt(arr)      # [1, 1.41, 1.73, 2, 2.24]
np.exp(arr)       # e^x
np.log(arr)       # ln(x)
np.sin(arr)       # sin(x)
np.abs(arr)       # 绝对值

# 聚合运算
arr.sum()         # 总和
arr.mean()        # 均值
arr.std()         # 标准差
arr.var()         # 方差
arr.min()         # 最小值
arr.max()         # 最大值
arr.argmax()      # 最大值索引

# 按轴聚合
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
arr_2d.sum(axis=0)  # [5, 7, 9]（列求和）
arr_2d.sum(axis=1)  # [6, 15]（行求和）

# 累积运算
arr.cumsum()      # [1, 3, 6, 10, 15]
arr.cumprod()     # [1, 2, 6, 24, 120]
```

### 7. 线性代数

```python
from numpy import linalg as LA

# 矩阵乘法
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B              # 矩阵乘法
C2 = np.dot(A, B)      # 同上
C3 = np.matmul(A, B)   # 同上

# 向量点积
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
dot = np.dot(v1, v2)   # 32

# 逆矩阵
A_inv = LA.inv(A)
print(A @ A_inv)  # 单位矩阵（近似）

# 行列式
det = LA.det(A)

# 特征值和特征向量
eigenvalues, eigenvectors = LA.eig(A)

# SVD 分解
U, S, Vt = LA.svd(A)

# 解线性方程组
# Ax = b
A = np.array([[3, 1], [1, 2]])
b = np.array([9, 8])
x = LA.solve(A, b)
print(x)  # [2. 3.]
```

---

## 第二部分：Pandas

### 1. Series 创建

```python
import pandas as pd
import numpy as np

# 从列表
s = pd.Series([1, 2, 3, 4, 5])

# 指定索引
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])

# 从字典
d = {'a': 1, 'b': 2, 'c': 3}
s = pd.Series(d)

# 操作
s.values      # 值数组
s.index       # 索引
s['a']        # 访问
s['a'] = 10   # 修改
```

### 2. DataFrame 创建

```python
# 从字典
df = pd.DataFrame({
    'name': ['悟空', '八戒', '沙僧'],
    'age': [1000, 500, 800],
    'power': [9999, 8000, 7000]
})

# 从列表
df = pd.DataFrame([
    ['悟空', 1000, 9999],
    ['八戒', 500, 8000],
    ['沙僧', 800, 7000]
], columns=['name', 'age', 'power'])

# 从 CSV
df = pd.read_csv('data.csv')

# 从 Excel
df = pd.read_excel('data.xlsx')

# 查看
df.head()      # 前 5 行
df.tail()      # 后 5 行
df.info()      # 基本信息
df.describe()  # 统计描述
df.shape       # 形状
df.columns     # 列名
df.index       # 索引
```

### 3. 数据选择

```python
# 列选择
df['name']         # 单列（Series）
df[['name', 'age']]  # 多列（DataFrame）

# 行选择（标签）
df.loc[0]          # 单行
df.loc[0:2]        # 多行（包含结束）
df.loc[0:2, 'name':'age']  # 行列切片

# 行选择（位置）
df.iloc[0]         # 第一行
df.iloc[0:2]       # 前两行（不包含结束）
df.iloc[0:2, 0:2]  # 行列切片

# 布尔选择
df[df['age'] > 600]
df[(df['age'] > 600) & (df['power'] > 7500)]
df[(df['age'] > 600) | (df['power'] > 7500)]

# query 方法
df.query('age > 600 and power > 7500')
```

### 4. 数据清洗

```python
# 缺失值处理
df.isnull()        # 检查缺失
df.notnull()       # 检查非缺失
df.dropna()        # 删除缺失
df.fillna(0)       # 填充为 0
df.fillna(method='ffill')  # 前向填充
df.fillna(method='bfill')  # 后向填充

# 重复值
df.duplicated()    # 检查重复
df.drop_duplicates()  # 删除重复

# 重命名
df.rename(columns={'name': '姓名', 'age': '年龄'}, inplace=True)

# 类型转换
df['age'] = df['age'].astype(float)
df['date'] = pd.to_datetime(df['date'])

# 字符串处理
df['name'].str.upper()
df['name'].str.lower()
df['name'].str.len()
df['name'].str.contains('悟')
```

### 5. 数据操作

```python
# 添加列
df['new_col'] = df['age'] * 2
df['power_level'] = df.apply(lambda row: row['power'] / 1000, axis=1)

# 删除列
df.drop('new_col', axis=1, inplace=True)

# 排序
df.sort_values('age')           # 升序
df.sort_values('age', ascending=False)  # 降序
df.sort_values(['age', 'power'])  # 多列排序

# 排名
df['rank'] = df['power'].rank(ascending=False)

# 分组聚合
df.groupby('category')['value'].mean()
df.groupby('category').agg({
    'value': ['mean', 'sum', 'count'],
    'price': 'median'
})

# 透视表
pd.pivot_table(df, values='value', index='date', 
               columns='category', aggfunc='sum')
```

### 6. 数据合并

```python
# 拼接
pd.concat([df1, df2], axis=0)  # 行拼接
pd.concat([df1, df2], axis=1)  # 列拼接

# 合并（类似 SQL JOIN）
pd.merge(df1, df2, on='key')           # 内连接
pd.merge(df1, df2, on='key', how='left')   # 左连接
pd.merge(df1, df2, on='key', how='right')  # 右连接
pd.merge(df1, df2, on='key', how='outer')  # 外连接

# 连接（基于索引）
df1.join(df2, how='left')
```

### 7. 时间序列

```python
# 创建时间范围
dates = pd.date_range('2024-01-01', periods=10, freq='D')

# 时间索引
df = pd.DataFrame(np.random.randn(10, 4), 
                  index=dates, 
                  columns=['A', 'B', 'C', 'D'])

# 重采样
df.resample('W').mean()  # 按周平均
df.resample('M').sum()   # 按月求和

# 移动窗口
df.rolling(window=3).mean()  # 3 日移动平均
df.expanding().mean()        # 扩展窗口平均
```

---

## 💻 实践项目

### 项目 1：泰坦尼克数据分析

```python
"""
项目：泰坦尼克数据集探索性分析
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
df = pd.read_csv('titanic.csv')

# 基本了解
print(df.info())
print(df.describe())

# 数据质量
print(f"缺失值:\n{df.isnull().sum()}")

# 生存率分析
survival_rate = df['Survived'].mean()
print(f"生存率：{survival_rate:.2%}")

# 性别与生存率
survival_by_sex = df.groupby('Sex')['Survived'].mean()
print(f"性别生存率:\n{survival_by_sex}")

# 舱位与生存率
survival_by_class = df.groupby('Pclass')['Survived'].mean()
print(f"舱位生存率:\n{survival_by_class}")

# 年龄分布
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
df['Age'].hist(bins=30)
plt.title('年龄分布')

plt.subplot(1, 2, 2)
df[df['Survived']==1]['Age'].hist(bins=30, alpha=0.5, label='生存')
df[df['Survived']==0]['Age'].hist(bins=30, alpha=0.5, label='遇难')
plt.legend()
plt.title('生存/遇难年龄分布')

plt.tight_layout()
plt.show()

# 特征工程
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
```

### 项目 2：股票数据分析

```python
"""
项目：股票价格分析与可视化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 生成模拟股票数据
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=252, freq='B')  # 交易日
returns = np.random.randn(252) * 0.02  # 日收益率
price = 100 * np.cumprod(1 + returns)  # 从 100 开始

df = pd.DataFrame({
    'date': dates,
    'price': price
})
df.set_index('date', inplace=True)

# 计算指标
df['returns'] = df['price'].pct_change()
df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
df['ma_20'] = df['price'].rolling(window=20).mean()
df['ma_60'] = df['price'].rolling(window=60).mean()

# 波动率
df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 价格
df['price'].plot(ax=axes[0])
df['ma_20'].plot(ax=axes[0])
df['ma_60'].plot(ax=axes[0])
axes[0].set_title('股票价格与移动平均线')
axes[0].legend(['价格', '20 日均线', '60 日均线'])

# 收益率
df['returns'].plot(ax=axes[1])
axes[1].set_title('日收益率')
axes[1].axhline(0, color='red', linestyle='--')

# 波动率
df['volatility_20'].plot(ax=axes[2])
axes[2].set_title('20 日滚动波动率（年化）')

plt.tight_layout()
plt.show()

# 统计
print(f"总收益率：{(price[-1] / price[0] - 1):.2%}")
print(f"年化收益率：{df['returns'].mean() * 252:.2%}")
print(f"年化波动率：{df['returns'].std() * np.sqrt(252):.2%}")
print(f"最大回撤：{(df['price'] / df['price'].cummax() - 1).min():.2%}")
```

---

## ✅ 检查清单

- [ ] 掌握 NumPy 数组创建和操作
- [ ] 理解广播机制
- [ ] 熟练使用 NumPy 线性代数
- [ ] 掌握 Pandas DataFrame 操作
- [ ] 能够进行数据清洗和转换
- [ ] 熟练使用分组聚合
- [ ] 完成实践项目

---

> _NumPy 和 Pandas 是数据科学的双翼，掌握它们，数据将在你手中起舞。_
> 
> _—— 悟空_
