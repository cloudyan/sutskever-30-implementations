# 2.1 Python 核心语法

> 编程基础 —— 掌握 Python 的核心语法和编程范式
> 
> _预计时间：3-5 天 | 难度：⭐⭐ | 前置知识：无_

---

## 📖 内容概览

```
Python 核心语法学习路径：

Day 1: 变量、数据类型、运算符
Day 2: 控制流（条件、循环）
Day 3: 数据结构（列表、字典、集合、元组）
Day 4: 函数与模块
Day 5: 面向对象编程
```

---

## 1. 变量与数据类型

### 基本数据类型

```python
# 整数（int）
age = 25
negative = -10
big_num = 10**100  # Python 支持任意大整数

# 浮点数（float）
price = 19.99
pi = 3.14159
scientific = 1.23e-4  # 科学计数法

# 布尔值（bool）
is_active = True
is_empty = False

# 字符串（str）
name = "悟空"
greeting = '你好，世界'
multi_line = """
这是多行
字符串
"""

# None（空值）
result = None
```

### 类型操作

```python
# 类型查询
type(age)      # <class 'int'>
type(name)     # <class 'str'>

# 类型转换
int("123")      # 123
float("3.14")   # 3.14
str(123)        # "123"
bool(0)         # False
bool(1)         # True
bool("")        # False
bool("abc")     # True

# 格式化字符串
name = "悟空"
age = 1000

# f-string（推荐）
print(f"{name}今年{age}岁")

# format 方法
print("{}今年{}岁".format(name, age))

# % 格式化（旧式）
print("%s今年%d岁" % (name, age))
```

### 运算符

```python
# 算术运算符
a, b = 10, 3
a + b    # 13  加
a - b    # 7   减
a * b    # 30  乘
a / b    # 3.33... 除
a // b   # 3   整除
a % b    # 1   取余
a ** b   # 1000 幂

# 比较运算符
a == b   # 等于
a != b   # 不等于
a > b    # 大于
a < b    # 小于
a >= b   # 大于等于
a <= b   # 小于等于

# 逻辑运算符
True and False   # False
True or False    # True
not True         # False

# 位运算符
a & b    # 按位与
a | b    # 按位或
a ^ b    # 按位异或
~a       # 按位取反
a << 2   # 左移
a >> 2   # 右移

# 赋值运算符
a = 10
a += 5   # a = a + 5
a -= 3   # a = a - 3
a *= 2   # a = a * 2
```

---

## 2. 控制流

### 条件语句

```python
# if-elif-else
score = 85

if score >= 90:
    grade = "A"
    print("优秀！")
elif score >= 80:
    grade = "B"
    print("良好")
elif score >= 60:
    grade = "C"
    print("及格")
else:
    grade = "D"
    print("不及格")

# 三元表达式
status = "通过" if score >= 60 else "不通过"

# 链式比较
if 60 <= score < 90:
    print("中等偏上")

# 成员检查
name = "悟空"
if "悟" in name:
    print("名字包含'悟'")

# 身份检查
a = [1, 2, 3]
b = a
c = [1, 2, 3]
print(a is b)  # True（同一对象）
print(a is c)  # False（不同对象）
print(a == c)  # True（值相等）
```

### 循环语句

```python
# for 循环
for i in range(5):      # 0, 1, 2, 3, 4
    print(i)

for i in range(2, 5):   # 2, 3, 4
    print(i)

for i in range(0, 10, 2):  # 0, 2, 4, 6, 8
    print(i)

# 遍历列表
fruits = ["苹果", "香蕉", "橙子"]
for fruit in fruits:
    print(fruit)

# 遍历带索引
for i, fruit in enumerate(fruits):
    print(f"{i}: {fruit}")

# 遍历字典
person = {"name": "悟空", "age": 1000}
for key, value in person.items():
    print(f"{key}: {value}")

# while 循环
count = 0
while count < 5:
    print(count)
    count += 1

# break 和 continue
for i in range(10):
    if i == 3:
        continue  # 跳过本次
    if i == 7:
        break     # 退出循环
    print(i)

# else 子句（循环正常结束时执行）
for i in range(5):
    print(i)
else:
    print("循环结束")  # 会执行

for i in range(5):
    if i == 3:
        break
else:
    print("不会执行")  # break 后不执行
```

---

## 3. 数据结构

### 列表（List）

```python
# 创建
fruits = ["苹果", "香蕉", "橙子"]
numbers = [1, 2, 3, 4, 5]
mixed = [1, "hello", 3.14, True]
empty = []

# 访问
fruits[0]      # "苹果"
fruits[-1]     # "橙子"（最后一个）
fruits[1:3]    # ["香蕉", "橙子"]（切片）
fruits[::-1]   # 反转列表

# 修改
fruits[0] = "葡萄"
fruits.append("西瓜")      # 末尾添加
fruits.insert(1, "柠檬")   # 指定位置插入
fruits.extend(["桃子", "李子"])  # 扩展

# 删除
fruits.remove("香蕉")      # 删除指定值
del fruits[0]              # 删除指定位置
last = fruits.pop()        # 弹出最后一个
fruits.pop(0)              # 弹出第一个

# 查询
len(fruits)                # 长度
"苹果" in fruits           # 成员检查
fruits.index("橙子")       # 索引位置
fruits.count("苹果")       # 出现次数

# 排序
numbers.sort()             # 原地排序
sorted_numbers = sorted(numbers)  # 返回新列表
numbers.sort(reverse=True) # 降序

# 列表推导式
squares = [x**2 for x in range(10)]
even_squares = [x**2 for x in range(10) if x % 2 == 0]
matrix = [[i*j for j in range(3)] for i in range(3)]
```

### 字典（Dictionary）

```python
# 创建
person = {"name": "悟空", "age": 1000, "power": 9999}
empty_dict = {}
from_keys = dict.fromkeys(["a", "b", "c"], 0)

# 访问
person["name"]           # "悟空"
person.get("name")       # "悟空"
person.get("height", 175)  # 175（默认值）

# 修改
person["age"] = 1001
person["height"] = 175   # 新增键值对

# 删除
del person["power"]
age = person.pop("age")

# 遍历
for key in person:
    print(key)

for value in person.values():
    print(value)

for key, value in person.items():
    print(f"{key}: {value}")

# 字典推导式
squares = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}

# 合并
dict1 = {"a": 1, "b": 2}
dict2 = {"c": 3, "d": 4}
merged = {**dict1, **dict2}  # Python 3.5+
merged = dict1 | dict2       # Python 3.9+
```

### 集合（Set）

```python
# 创建
fruits = {"苹果", "香蕉", "橙子"}
empty_set = set()  # 注意：{}是空字典
from_list = set([1, 2, 2, 3, 3])  # {1, 2, 3}

# 基本操作
fruits.add("西瓜")
fruits.remove("香蕉")  # 不存在会报错
fruits.discard("葡萄")  # 不存在不报错
fruits.pop()  # 随机弹出一个

# 集合运算
A = {1, 2, 3, 4}
B = {3, 4, 5, 6}

A | B   # 并集 {1, 2, 3, 4, 5, 6}
A & B   # 交集 {3, 4}
A - B   # 差集 {1, 2}
A ^ B   # 对称差集 {1, 2, 5, 6}

# 集合推导式
squares = {x**2 for x in range(5)}
```

### 元组（Tuple）

```python
# 创建
point = (3, 4)
single = (1,)  # 单元素元组需要逗号
empty = ()

# 访问
point[0]  # 3
point[1]  # 4

# 不可变（不能修改）
# point[0] = 5  # 会报错

# 解包
x, y = point
first, *rest, last = [1, 2, 3, 4, 5]
# first=1, rest=[2,3,4], last=5

# 命名元组
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(3, 4)
print(p.x, p.y)  # 3 4
```

---

## 4. 函数

### 函数定义

```python
# 基本函数
def greet(name):
    """打招呼函数"""
    return f"你好，{name}！"

# 默认参数
def greet(name, greeting="你好"):
    return f"{greeting}，{name}！"

# 可变参数
def sum_all(*args):
    """接收任意数量的位置参数"""
    return sum(args)

sum_all(1, 2, 3, 4)  # 10

def print_info(**kwargs):
    """接收任意数量的关键字参数"""
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="悟空", age=1000)

# 混合参数
def func(a, b, *args, c=10, **kwargs):
    pass

# Lambda 函数
square = lambda x: x ** 2
add = lambda x, y: x + y

# 在排序中使用
students = [("小明", 85), ("小红", 92), ("小刚", 78)]
students.sort(key=lambda x: x[1], reverse=True)
```

### 作用域

```python
# LEGB 规则
# Local -> Enclosing -> Global -> Built-in

global_var = "全局"

def outer():
    enclosing_var = "外层"
    
    def inner():
        local_var = "内层"
        print(local_var)    # Local
        print(enclosing_var)  # Enclosing
        print(global_var)   # Global
        print(len)          # Built-in
    
    inner()

# global 关键字
count = 0

def increment():
    global count
    count += 1

# nonlocal 关键字
def outer():
    count = 0
    
    def inner():
        nonlocal count
        count += 1
    
    inner()
    print(count)  # 1
```

### 装饰器

```python
# 简单装饰器
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}耗时：{end-start:.4f}秒")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)

slow_function()

# 带参数的装饰器
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("Hello")
```

---

## 5. 模块与包

### 导入模块

```python
# 导入整个模块
import math
print(math.sqrt(16))  # 4.0

# 导入特定函数
from math import sqrt, pi
print(sqrt(16))  # 4.0

# 导入并重命名
import numpy as np
from datetime import datetime as dt

# 导入所有（不推荐）
from math import *

# 自定义模块
# mymodule.py
# def hello():
#     print("Hello")

# main.py
# import mymodule
# mymodule.hello()
```

### 创建包

```
my_package/
├── __init__.py
├── module1.py
├── module2.py
└── subpackage/
    ├── __init__.py
    └── module3.py
```

```python
# __init__.py 可以定义包的公共接口
from .module1 import func1
from .module2 import func2

__all__ = ['func1', 'func2']  # 控制 import * 的行为
```

---

## 💻 实践项目

### 项目 1：数据处理脚本

```python
"""
项目：CSV 数据清洗与统计
要求：
1. 读取 CSV 文件
2. 处理缺失值
3. 计算统计指标
4. 输出结果
"""

import csv
from statistics import mean, median, stdev

def process_csv(input_file, output_file):
    """处理 CSV 数据"""
    data = []
    
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    
    # 统计数值列
    values = [float(row['value']) for row in data if row.get('value')]
    
    stats = {
        'count': len(values),
        'mean': mean(values),
        'median': median(values),
        'std': stdev(values) if len(values) > 1 else 0,
        'min': min(values),
        'max': max(values)
    }
    
    # 输出统计
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("数据统计报告\n")
        f.write("=" * 40 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:.4f}\n")
    
    return stats

# 使用示例
# stats = process_csv('input.csv', 'output.txt')
```

### 项目 2：待办事项管理器

```python
"""
项目：命令行待办事项管理器
功能：添加、查看、完成、删除任务
"""

import json
from datetime import datetime

class TodoManager:
    def __init__(self, filename='todos.json'):
        self.filename = filename
        self.todos = self.load()
    
    def load(self):
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save(self):
        with open(self.filename, 'w', encoding='utf-8') as f:
            json.dump(self.todos, f, ensure_ascii=False, indent=2)
    
    def add(self, title, priority='medium'):
        """添加任务"""
        todo = {
            'id': len(self.todos) + 1,
            'title': title,
            'priority': priority,
            'created_at': datetime.now().isoformat(),
            'completed': False
        }
        self.todos.append(todo)
        self.save()
        print(f"✓ 已添加任务：{title}")
    
    def list(self, show_completed=False):
        """列出任务"""
        for todo in self.todos:
            if not show_completed and todo['completed']:
                continue
            status = "✓" if todo['completed'] else "○"
            print(f"{todo['id']}. [{status}] {todo['title']} ({todo['priority']})")
    
    def complete(self, todo_id):
        """完成任务"""
        for todo in self.todos:
            if todo['id'] == todo_id:
                todo['completed'] = True
                self.save()
                print(f"✓ 已完成任务：{todo['title']}")
                return
        print(f"✗ 未找到任务 {todo_id}")
    
    def delete(self, todo_id):
        """删除任务"""
        for i, todo in enumerate(self.todos):
            if todo['id'] == todo_id:
                removed = self.todos.pop(i)
                self.save()
                print(f"✓ 已删除任务：{removed['title']}")
                return
        print(f"✗ 未找到任务 {todo_id}")

# 使用示例
# manager = TodoManager()
# manager.add("学习 Python", "high")
# manager.add("完成作业", "medium")
# manager.list()
```

---

## ✅ 检查清单

- [ ] 理解变量和数据类型
- [ ] 掌握控制流（if/for/while）
- [ ] 熟练使用列表、字典、集合、元组
- [ ] 能够定义和调用函数
- [ ] 理解作用域规则
- [ ] 了解装饰器基本概念
- [ ] 能够导入和使用模块
- [ ] 完成实践项目

---

> _Python 是 AI 工程师的瑞士军刀，熟练它，你将无所不能。_
> 
> _—— 悟空_
