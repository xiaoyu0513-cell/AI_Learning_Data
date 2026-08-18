这份代码主要是在展示 **Python Google Style Docstring（谷歌风格文档字符串）规范**。

简单来说：

> **Docstring 是写在函数、类、模块里面的三引号字符串，用来描述代码的用途、参数、返回值、异常等，方便生成 API 文档和 IDE 提示。**

---

# 1. 模块（Module）文档

写在文件最顶部：

```python
"""
Example Google style docstrings.

This module demonstrates documentation...
"""
```

作用：

描述整个 Python 文件。

通常包含：

```text
模块作用
示例
全局变量
TODO
引用
```

格式：

```python
"""
模块说明

Example:
    使用示例

Attributes:
    变量名 (类型): 描述

Todo:
    待完成事项
"""
```

---

# 2. 模块变量文档

## 方法1：Attributes 统一描述

```python
module_level_variable1 = 12345
```

在顶部：

```python
Attributes:
    module_level_variable1 (int):
        Module level variable.
```

---

## 方法2：变量下面直接写

```python
module_level_variable2 = 98765
"""int: Module level variable documented inline."""
```

说明：

这个变量类型是：

```python
int
```

---

# 3. 函数 Docstring

基本结构：

```python
def function(param1):
    """
    函数说明

    Args:
        参数说明

    Returns:
        返回值说明

    Raises:
        异常说明
    """
```

---

## Args（参数）

例如：

```python
def add(a, b):
    """
    Args:
        a (int): 第一个数字
        b (int): 第二个数字
    """
```

格式：

```text
参数名 (类型): 描述
```

---

支持：

### 普通参数

```python
param1 (int): 参数说明
```

---

### 可选参数

```python
param2 (str, optional):
    Defaults to None.
```

表示：

```python
param2=None
```

---

### 可变参数

```python
*args: 可变长度参数

**kwargs: 字典参数
```

对应：

```python
def func(*args, **kwargs):
```

---

# 4. Returns（返回值）

描述函数返回什么。

例如：

```python
def login():
    """
    Returns:
        bool:
            True 登录成功
            False 登录失败
    """
```

表示：

返回：

```python
bool
```

---

也可以省略类型：

```python
Returns:
    登录结果
```

---

# 5. Raises（异常）

说明可能抛出的异常：

```python
Raises:
    ValueError:
        参数错误

    AttributeError:
        属性不存在
```

例如：

代码：

```python
if param1 == param2:
    raise ValueError()
```

文档：

```python
Raises:
    ValueError:
        If param1 equals param2.
```

---

# 6. Generator 使用 Yields

普通函数：

```python
return
```

使用：

```text
Returns
```

生成器：

```python
yield
```

使用：

```text
Yields
```

例如：

```python
def numbers():
    """
    Yields:
        int:
            返回数字
    """
    yield 1
```

---

# 7. 类 Class Docstring

格式：

```python
class User:
    """
    用户类说明

    Attributes:
        属性说明
    """
```

例如：

```python
class User:
    """
    用户信息

    Attributes:
        name (str):
            用户名字
    """
```

---

# 8. **init** 方法文档

构造函数：

```python
def __init__(self, name):
```

注意：

**不要写 self**

错误：

```python
Args:
    self:
        用户对象
```

正确：

```python
Args:
    name (str):
        用户名
```

---

# 9. 类属性文档

三种方式：

---

## 方式1：类顶部 Attributes

```python
class User:
    """
    Attributes:
        name (str):
            用户名
    """
```

---

## 方式2：代码旁边注释

```python
self.name = "Tom"  #: 用户名
```

---

## 方式3：下面写字符串

```python
self.age = 18
"""int: 用户年龄"""
```

---

# 10. Property 文档

@property：

```python
@property
def name(self):
    """
    str:
        用户名字
    """
```

文档写在 getter：

```python
@property
def xxx()
```

而不是 setter。

---

# 11. PEP484 类型注解

Python 推荐：

```python
def add(
    a:int,
    b:int
) -> int:
```

代替：

```python
Args:
    a(int)
    b(int)

Returns:
    int
```

因为类型已经写在代码里。

---

例如：

```python
def login(username: str) -> bool:
```

IDE 可以直接识别。

---

# 12. Google Docstring 标准结构总结

## 函数

```python
def func(a, b):
    """
    简短说明

    Args:
        a (int): 参数说明
        b (str): 参数说明

    Returns:
        bool: 返回说明

    Raises:
        ValueError:
            错误说明
    """
```

---

## 类

```python
class User:
    """
    用户类说明

    Attributes:
        name (str):
            用户名
        age (int):
            年龄
    """
```

---

## 模块

```python
"""
模块说明

Example:
    使用例子

Attributes:
    全局变量

Todo:
    待完成
"""
```

---

# 13. 你学习 LangChain 时最常见的部分

LangChain 源码里大量使用这种：

```python
def invoke(
    input: str,
    config: RunnableConfig | None = None
):
    """
    Invoke the runnable.

    Args:
        input:
            Input data.

        config:
            Configuration.

    Returns:
        Output result.
    """
```

重点看：

| 部分         | 作用   |
| ---------- | ---- |
| Args       | 输入参数 |
| Returns    | 返回值  |
| Raises     | 异常   |
| Attributes | 类属性  |
| Example    | 使用示例 |

---

一句话总结：

> Google Style Docstring 是 Python 项目中用于写 API 文档的规范，通过 `Args / Returns / Raises / Attributes / Examples` 等章节，把函数、类、模块的用途和使用方式结构化描述出来。LangChain、FastAPI、Django 等大型项目源码大量采用这种格式。
