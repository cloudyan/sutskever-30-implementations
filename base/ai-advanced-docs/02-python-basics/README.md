# 阶段 2：Python 编程基础

_Python 核心语法与数据科学工具_

---

## 📖 学习指南

**前置知识**：无编程基础也可学习

**学习目标**：
- ✅ 掌握 Python 核心语法
- ✅ 熟练使用 NumPy 进行数值计算
- ✅ 使用 Pandas 进行数据处理
- ✅ 使用 Matplotlib/Seaborn 进行数据可视化

**预计时间**：14 天

---

## 2.1 Python 核心语法

### 基础语法

<div class="formula-box">

```python
# 变量与数据类型
x = 10              # 整数
y = 3.14            # 浮点数
name = "Python"     # 字符串
is_valid = True     # 布尔值

# 类型转换
int("123")          # 字符串转整数
float(10)           # 整数转浮点数
str(123)            # 整数转字符串
```

</div>

### 数据结构

<div class="formula-box">

```python
# 列表（List）
fruits = ["apple", "banana", "orange"]
fruits.append("grape")      # 添加元素
fruits[0]                   # 访问元素
fruits[1:3]                 # 切片

# 字典（Dictionary）
person = {"name": "Alice", "age": 25}
person["name"]              # 访问值
person["age"] = 26          # 修改值
person.keys()               # 获取所有键
person.values()             # 获取所有值

# 集合（Set）
unique_numbers = {1, 2, 3, 3, 4}  # {1, 2, 3, 4}
unique_numbers.add(5)             # 添加元素

# 元组（Tuple）
coordinates = (10, 20)      # 不可变
x, y = coordinates          # 解包
```

</div>

### 控制流

<div class="formula-box">

```python
# 条件语句
if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
else:
    grade = "C"

# for 循环
for i in range(5):      # 0, 1, 2, 3, 4
    print(i)

for fruit in fruits:
    print(fruit)

# while 循环
count = 0
while count < 5:
    print(count)
    count += 1
```

</div>

### 函数

<div class="formula-box">

```python
# 定义函数
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# 调用函数
greet("Alice")                  # "Hello, Alice!"
greet("Bob", "Hi")              # "Hi, Bob!"

# Lambda 函数
square = lambda x: x ** 2
square(5)                       # 25

# 可变参数
def sum_all(*args):
    return sum(args)

sum_all(1, 2, 3, 4)             # 10
```

</div>

### 类与对象

<div class="formula-box">

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hi, I'm {self.name}"
    
    def __str__(self):
        return f"{self.name}, {self.age}岁"

# 创建对象
alice = Person("Alice", 25)
alice.greet()                   # "Hi, I'm Alice"
print(alice)                    # "Alice, 25 岁"
```

</div>

### 异常处理

<div class="formula-box">

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("不能除以零！")
except Exception as e:
    print(f"发生错误：{e}")
finally:
    print("执行完毕")
```

</div>

### 模块与包

<div class="formula-box">

```python
# 导入整个模块
import math
math.sqrt(16)                   # 4.0

# 导入特定函数
from math import sqrt, pi
sqrt(16)                        # 4.0

# 导入并重命名
import numpy as np
import pandas as pd
```

</div>

---

## 2.2 NumPy 数值计算

### 数组基础

<div class="formula-box">

```python
import numpy as np

# 创建数组
arr = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2], [3, 4]])

# 特殊数组
np.zeros(5)                     # [0. 0. 0. 0. 0.]
np.ones((2, 3))                 # 2x3 的全 1 数组
np.arange(0, 10, 2)             # [0 2 4 6 8]
np.linspace(0, 1, 5)            # [0. 0.25 0.5 0.75 1.]
np.random.rand(3, 3)            # 3x3 随机数组
```

</div>

### 数组操作

<div class="formula-box">

```python
# 形状操作
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr.shape                       # (2, 3)
arr.reshape(3, 2)               # 重塑形状
arr.flatten()                   # 展平
arr.T                           # 转置

# 索引与切片
arr[0, 1]                       # 2
arr[:, 1]                       # [2, 5] 第二列
arr[arr > 3]                    # [4, 5, 6] 条件索引

# 数组合并
np.concatenate([arr1, arr2])    # 沿现有轴连接
np.vstack([arr1, arr2])         # 垂直堆叠
np.hstack([arr1, arr2])         # 水平堆叠
```

</div>

### 广播机制

<div class="formula-box">

```python
# 广播示例
arr = np.array([[1, 2, 3], [4, 5, 6]])
arr + 10                        # [[11, 12, 13], [14, 15, 16]]

arr1 = np.array([[1], [2], [3]])    # (3, 1)
arr2 = np.array([10, 20, 30])       # (3,)
arr1 + arr2                     # 自动广播
```

</div>

### 线性代数运算

<div class="formula-box">

```python
# 矩阵乘法
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
A @ B                           # 矩阵乘法
np.dot(A, B)                    # 同上

# 特征值与特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)

# 逆矩阵
A_inv = np.linalg.inv(A)

# SVD 分解
U, S, Vt = np.linalg.svd(A)
```

</div>

### 统计运算

<div class="formula-box">

```python
arr = np.array([1, 2, 3, 4, 5])

arr.mean()                      # 3.0
arr.std()                       # 标准差
arr.var()                       # 方差
arr.sum()                       # 总和
arr.min(), arr.max()            # 最小值，最大值
np.median(arr)                  # 中位数
np.percentile(arr, 75)          # 75 百分位数
```

</div>

---

## 2.3 Pandas 数据处理

### Series 与 DataFrame

<div class="formula-box">

```python
import pandas as pd

# Series
s = pd.Series([1, 3, 5, 7], index=['a', 'b', 'c', 'd'])
s['b']                          # 3

# DataFrame
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'city': ['NYC', 'LA', 'Chicago']
})

df.head()                       # 前 5 行
df.info()                       # 数据信息
df.describe()                   # 统计摘要
```

</div>

### 数据选择

<div class="formula-box">

```python
# 列选择
df['name']                      # 选择单列
df[['name', 'age']]             # 选择多列

# 行选择
df.loc[0]                       # 按标签选择
df.iloc[0]                      # 按位置选择
df.loc[0:2, 'name':'age']       # 行列切片

# 条件选择
df[df['age'] > 30]              # 年龄大于 30
df[(df['age'] > 30) & (df['city'] == 'LA')]
```

</div>

### 数据清洗

<div class="formula-box">

```python
# 处理缺失值
df.dropna()                     # 删除缺失值
df.fillna(0)                    # 填充为 0
df.fillna(df.mean())            # 用均值填充

# 删除重复值
df.drop_duplicates()

# 数据类型转换
df['age'] = df['age'].astype(float)

# 重命名
df.rename(columns={'name': '姓名', 'age': '年龄'})
```

</div>

### 数据聚合

<div class="formula-box">

```python
# 分组聚合
df.groupby('city')['age'].mean()    # 按城市分组求平均年龄
df.groupby('city').agg({
    'age': ['mean', 'min', 'max'],
    'name': 'count'
})

# 透视表
pd.pivot_table(df, values='age', index='city', aggfunc='mean')

# 交叉表
pd.crosstab(df['city'], df['age'] > 30)
```

</div>

### 数据合并

<div class="formula-box">

```python
# 合并
pd.concat([df1, df2])               # 纵向合并
pd.concat([df1, df2], axis=1)       # 横向合并

# 连接（类似 SQL JOIN）
pd.merge(df1, df2, on='key')        # 内连接
pd.merge(df1, df2, on='key', how='outer')   # 外连接
pd.merge(df1, df2, on='key', how='left')    # 左连接
```

</div>

---

## 2.4 数据可视化

### Matplotlib 基础

<div class="formula-box">

```python
import matplotlib.pyplot as plt

# 基础折线图
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y)
plt.xlabel('X 轴')
plt.ylabel('Y 轴')
plt.title('折线图')
plt.show()

# 散点图
plt.scatter(x, y)
plt.show()

# 柱状图
plt.bar(['A', 'B', 'C'], [10, 20, 30])
plt.show()

# 直方图
plt.hist(data, bins=20)
plt.show()
```

</div>

### Seaborn 高级可视化

<div class="formula-box">

```python
import seaborn as sns

# 设置样式
sns.set_style("whitegrid")

# 热力图
sns.heatmap(corr_matrix, annot=True)

# 箱线图
sns.boxplot(x='category', y='value', data=df)

# 小提琴图
sns.violinplot(x='category', y='value', data=df)

# 成对关系图
sns.pairplot(df)

# 回归图
sns.regplot(x='x', y='y', data=df)
```

</div>

---

## 2.5 实战项目

### 项目 1：数据分析实战

<div class="formula-box">

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 加载数据
df = pd.read_csv('sales_data.csv')

# 2. 数据探索
print(df.info())
print(df.describe())

# 3. 数据清洗
df.dropna(inplace=True)
df['date'] = pd.to_datetime(df['date'])

# 4. 数据分析
monthly_sales = df.groupby(df['date'].dt.to_period('M'))['sales'].sum()

# 5. 可视化
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index.astype(str), monthly_sales.values)
plt.title('月度销售趋势')
plt.xlabel('月份')
plt.ylabel('销售额')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

</div>

### 项目 2：特征工程实战

<div class="formula-box">

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# 1. 处理缺失值
imputer = SimpleImputer(strategy='mean')
df[['age', 'income']] = imputer.fit_transform(df[['age', 'income']])

# 2. 编码分类变量
le = LabelEncoder()
df['gender_encoded'] = le.fit_transform(df['gender'])

# 3. 特征缩放
scaler = StandardScaler()
df[['age_scaled', 'income_scaled']] = scaler.fit_transform(df[['age', 'income']])

# 4. 特征选择
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

</div>

---

## 📚 学习资源

### 官方文档

- [Python 官方文档](https://docs.python.org/zh-cn/3/)
- [NumPy 官方文档](https://numpy.org/doc/)
- [Pandas 官方文档](https://pandas.pydata.org/docs/)
- [Matplotlib 官方文档](https://matplotlib.org/stable/contents.html)

### 练习平台

- [LeetCode](https://leetcode.com/) - 算法练习
- [Kaggle](https://www.kaggle.com/) - 数据科学竞赛
- [HackerRank](https://www.hackerrank.com/) - Python 练习

### 推荐书籍

- 《Python 编程：从入门到实践》
- 《利用 Python 进行数据分析》
- 《Python 数据科学手册》

---

## ✅ 学习检查清单

- [ ] 掌握 Python 基础语法
- [ ] 熟练使用列表、字典、集合
- [ ] 理解函数与类
- [ ] 掌握 NumPy 数组操作
- [ ] 理解广播机制
- [ ] 熟练使用 Pandas DataFrame
- [ ] 掌握数据清洗与聚合
- [ ] 能用 Matplotlib 绘制基础图表
- [ ] 完成至少 1 个实战项目

---

*最后更新：2026-04-22*
