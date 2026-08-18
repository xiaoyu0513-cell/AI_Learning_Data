# 第05章：Tools

## 1、Tools概述

### 1.1 工具的重要性

要构建更强大的AI工程应用，只有生成文本这样的“纸上谈兵”能力自然是不够的。

工具是赋予大语言模型与外部世界交互能力的关键组件，从而能让智能体执行搜索、计算、数据库查询、邮件发送或调用第三方API等，进而构建功能强大的AI应用。借助工具，大模型才能从“认识世界”走向“改变世界”。

大模型和智能体的区别，就是是否能够调用工具。大模型能分析出来我们需求中需要用到的工具。但是如何调用工具得依靠智能体。

举例：

![image-20260802150619693](https://i.postimg.cc/CYDKg6bM/image-20260802150619693.png?dl=1)

工具是构建智能体的核心要素之一！

![image-20260802153136497](https://i.postimg.cc/MqLxf77f/image-20260802153136497.png?dl=1)

### 1.2 工具调用的方式

在LangChain中，工具（Tools）实际上是指明确定义了输入和输出的 可调用函数。因此， 工具调用（Tool Calling） 也被称为 函数调用（Function Calling）。mcp 是 fc 的一种

MCP 的全称是 **Model Context Protocol**（模型上下文协议）。

它是由 Anthropic 开源的一项标准协议，旨在为大语言模型（LLM）与外部数据源、本地工具及应用之间提供统一的接口规范。

具体有两种调用方式：

#### **方式1：直接调用**

这种方式，适合测试时使用。

```python
from langchain_core.tools import tool

@tool # 写工具类提示
def get_weather(city: str) -> str:
    """
    获取指定城市的天气信息

    参数:
        city: 城市名称，如"北京"、"上海"

    返回:
        天气信息字符串
    """
    # 你的实现
    return city + "晴天，温度 15°C"
```

```python
# 使用 .invoke() 方法
result = get_weather.invoke({"city": "北京"})
print(result)
```

```text
北京晴天，温度 15°C
```

#### **方式2：绑定到模型（主流）**

这种方式，让AI来调用，开发中使用。

```python
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os


# 从.env文件中加载环境变量
load_dotenv(override=True)

CLOSEAI_API_KEY = os.getenv("CLOSEAI_API_KEY")
CLOSEAI_BASE_URL = os.getenv("CLOSEAI_BASE_URL")

model = init_chat_model(
    model="gpt-5.4-mini",
    model_provider="openai",
    api_key=CLOSEAI_API_KEY,
    base_url=CLOSEAI_BASE_URL
)

from langchain_core.tools import tool

# 定义工具
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气"""
    # 你的实现
    return "晴天，温度 15°C"


# 绑定工具
model_with_tools = model.bind_tools([get_weather])

# AI 可以决定是否调用工具
response = model_with_tools.invoke("北京天气如何？")
# response = model_with_tools.invoke("2 + 3 = ？")

# 检查 AI 是否要调用工具
if response.tool_calls:
    print("AI 想调用工具：", response.tool_calls)
else:
    print("AI 直接回答：", response.content)
```

```text
AI 想调用工具： [{'name': 'get_weather', 'args': {'city': '北京'}, 'id':
'call_fR3LE8Wjqh9lnDosQ61Y892E', 'type': 'tool_call'}]
```

### 1.3 工具调用的整体流程

大模型能根据对话上下文决定何时调用工具以及传递哪些参数。

经典流程如下：

![image-20260802150811010](https://i.postimg.cc/b8QqMSjv/image-20260802150811010.png?dl=1)

大致的意思就是，用户提出询问，模型会判断出使用什么tools，数据返回在`tool_call`字段里，但是模型无法调用tools，我们就得通过ai 调用tools，得到tools输出后的结果，再将结果交给模型整理输出回答。

### 1.4 从Message流转看工具的调用

前提：模型的初始化

```python
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# 从.env文件中加载环境变量
load_dotenv(override=True)

CLOSEAI_API_KEY = os.getenv("CLOSEAI_API_KEY")
CLOSEAI_BASE_URL = os.getenv("CLOSEAI_BASE_URL")

model = init_chat_model(
    model="gpt-5.4-mini",
    model_provider="openai",
    api_key=CLOSEAI_API_KEY,
    base_url=CLOSEAI_BASE_URL
)
```

#### 参考1：不使用@tool修饰

```python
from langchain.messages import HumanMessage, ToolMessage

def get_weather(city: str):
    """获取天气的工具"""
    return f"{city}天气晴朗"

# 将模型和工具绑定
model_with_tools = model.bind_tools([get_weather])

messages = [
    HumanMessage("今天北京天气如何")
]

# 模型生成调用工具请求
response = model_with_tools.invoke(messages)

# 添加AIMessage
messages.append(response)

tool_calls = response.tool_calls

# 手动拼接tools message
for tool_call in tool_calls:
    if tool_call["name"] == "get_weather":
        # 拼接出ToolMessage实例
        tool_response = ToolMessage(
            content=get_weather(**tool_call["args"]),
            tool_call_id=tool_call["id"],
            name=tool_call["name"]
        )
        messages.append(tool_response)
print("=====================> messages <=====================")
for msg in messages:
    msg.pretty_print()
print("=====================> messages <=====================")
final_response = model_with_tools.invoke(messages)
print(f"final_response: \n{final_response}")
```

![image-20260802155953311](https://i.postimg.cc/NYfW8dkp/image-20260802155953311.png?dl=1)

#### 参考2：使用@tool修饰

```python
from langchain.messages import HumanMessage, ToolMessage

@tool
def get_weather(city: str):
    """获取天气的工具"""
    return f"{city}天气晴朗~"


# 将模型和工具绑定
model_with_tools = model.bind_tools([get_weather])

messages = [
    HumanMessage("今天北京天气如何")
]

# 模型生成调用工具请求
response = model_with_tools.invoke(messages)

# 添加AIMessage
messages.append(response)

tool_calls = response.tool_calls

for tool_call in tool_calls:
    if tool_call["name"] == "get_weather":
      	# 我们主动让大模型去调用工具
        # 返回的是ToolMessage类型消息，添加到消息列表中，不需要拼接
        tool_response = get_weather.invoke(tool_call)
        print(type(tool_response))
        messages.append(tool_response)

print("=====================> messages <=====================")
for msg in messages:
    msg.pretty_print()
print("=====================> messages <=====================")
# 再塞给大模型
final_response = model_with_tools.invoke(messages)
print(f"final_response: \n{final_response}")
```

说明：被 @tool 修饰的函数可以调用 invoke 接收模型返回的入参信息执行函数，并返回ToolMessage 实例，我们不再需要手动拼接 ToolMessage。

```python
# <class 'langchain_core.messages.tool.ToolMessage'>
"""
=====================> messages <=====================
================================ Human Message =================================

今天北京天气如何
================================== Ai Message ==================================

好的，我来帮你查一下今天北京的天气情况。
Tool Calls:
  get_weather (call_00_f65kV4JKjBPK0HhURzO99449)
 Call ID: call_00_f65kV4JKjBPK0HhURzO99449
 Args:
  city: 北京
================================= Tool Message =================================
Name: get_weather

北京天气晴朗~
=====================> messages <=====================
"""

final_response = AIMessage(
    content="今天北京天气**晴朗** ☀，是个好天气！如果你要出门的话，可以放心出行哦～",
    additional_kwargs={"refusal": None},
    response_metadata={
        "token_usage": {
            "completion_tokens": 24,
            "prompt_tokens": 343,
            "total_tokens": 367,
            "completion_tokens_details": None,
            "prompt_tokens_details": {
                "audio_tokens": None,
                "cached_tokens": 256,
            },
            "prompt_cache_hit_tokens": 256,
            "prompt_cache_miss_tokens": 87,
        },
        "model_provider": "deepseek",
        "model_name": "deepseek-v4-flash",
        "system_fingerprint": "fp_8b330d02d0_prod0820_fp8_kvcache_20260402",
        "id": "fabbebd6-498e-40f0-b1f6-e9fb43a36297",
        "finish_reason": "stop",
        "logprobs": None,
    },
    id="lc_run--019e5a90-f0ca-7d43-a4e5-3d34379e8208-0",
    tool_calls=[],
    invalid_tool_calls=[],
    usage_metadata={
        "input_tokens": 343,
        "output_tokens": 24,
        "total_tokens": 367,
        "input_token_details": {"cache_read": 256},
        "output_token_details": {},
    },
)
```

#### 核心区别对比

| **维度**                  | **不使用 @tool（纯普通函数）**                               | **使用 @tool 装饰器**                                        |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **工具类型**              | 普通 Python 函数 (`function`)                                | LangChain 工具对象 (`StructuredTool`)                        |
| **参数提取**              | 需手动用 `**tool_call["args"]` 解包                          | 自动解析 `tool_call` 结构                                    |
| **构建 ToolMessage**      | 必须**手动创建** `ToolMessage(...)` 实例并传入 `tool_call_id` 等参数 | **自动打包**为包含 `tool_call_id` 和输出内容的 `ToolMessage` 对象 |
| **错误处理**              | 需要自己写 `try-except` 捕获工具报错                         | 内置错误捕获机制 (可通过 `handle_tool_error` 策略自动兜底)   |
| **Agent / Executor 集成** | **无法**直接被 `create_react_agent` 等标准 Agent 组件直接调用 | **完美兼容** LangChain 所有的工具自动调用框架（如 LangGraph, AgentExecutor） |

对应图示（以参考1为例）：

![image-20260802150649489](https://i.postimg.cc/cdDxhqG1/image-20260802150649489.png?dl=1)

**工具调用流程总结：**

所以如果真正要大模型根据工具调用结果进行回复，完整的调用流程包括如下四个步骤：

步骤1：模型绑定工具：通过model.bind_tools([...])绑定一个或者多个工具。

步骤2：模型生成工具调用请求：用户输入问题，调用模型（比如invoke()）。如果需要调用工具，模型返回包含工具调用信息（如工具名称和参数）的AIMessage。

步骤3：开发者手动执行工具：用户从响应中提取工具调用信息并手动调用对应的工具（比如工具.invoke()）。

步骤4：将工具执行结果ToolMessage传递给模型生成最终结果：将之前用户提问内容和手动执行工具结果ToolMessage返回模型，模型最终生成回复。

特别注意：大模型调用工具是单次推理，直接响应，需要开发者手动执行工具并管理循环，适合简单、确定的任务。

## 2、工具的定义方式1：不使用@tool

### 2.1 模型绑定工具并发送请求

```python
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# 从.env文件中加载环境变量
load_dotenv(override=True)

CLOSEAI_API_KEY = os.getenv("CLOSEAI_API_KEY")
CLOSEAI_BASE_URL = os.getenv("CLOSEAI_BASE_URL")

model = init_chat_model(
    model="gpt-5.4-mini",
    model_provider="openai",
    api_key=CLOSEAI_API_KEY,
    base_url=CLOSEAI_BASE_URL
)
```

看下来就是自己写了一个方法去和大模型绑定

```python
from rich import print as rprint

# 定义工具
def get_weather(city: str):
    return f"{city}天气晴朗"

# 将模型和工具绑定
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke("今天北京天气如何")

rprint(response)
# print(response.tool_calls)
```

输出如下

```json
AIMessage(
    content="",
    additional_kwargs={"refusal": None},
    response_metadata={
        "token_usage": {
            "completion_tokens": 18,
            "prompt_tokens": 121,
            "total_tokens": 139,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            "latency_checkpoint": {
                "engine_tbt_ms": 3,
                "engine_ttft_ms": 41,
                "engine_ttlt_ms": 100,
                "pre_inference_ms": 99,
                "service_tbt_ms": 3,
                "service_ttft_ms": 207,
                "service_ttlt_ms": 260,
                "total_duration_ms": 168,
                "user_visible_ttft_ms": 108,
            },
        },
        "model_provider": "openai",
        "model_name": "gpt-5.4-mini-2026-03-17",
        "system_fingerprint": None,
        "id": "chatcmpl-Dif1tWr37cPmJRNpJuCLJNU9Yrxlk",
        "service_tier": "default",
        "finish_reason": "tool_calls",
        "logprobs": None,
    },
    id="lc_run--019e54a4-6989-71b1-8c7e-6efcfa08d4bb-0",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"city": "北京"},
            "id": "call_ECvZNV7RLTWpKQSjhvdGzKBd",
            "type": "tool_call",
        }
    ],
    invalid_tool_calls=[],
    usage_metadata={
        "input_tokens": 121,
        "output_tokens": 18,
        "total_tokens": 139,
        "input_token_details": {"audio": 0, "cache_read": 0},
        "output_token_details": {"audio": 0, "reasoning": 0},
    },
)
```

### 2.2 工具描述的各部分详解

#### 1. 了解：`convert_to_openai_tool`

执行 `model.bind_tools([get_weather])`，底层最终会调用 `convert_to_openai_tool`生成工具描述。所以我们可以直接调用后者查看解析后的工具描述。

```python
from langchain_core.utils.function_calling import convert_to_openai_tool

from rich import print as rprint

def get_weather(city: str):
    return f"{city}天气晴朗"

rprint(convert_to_openai_tool(get_weather))
```

输出如下

```JSON
{
  'type': 'function',
  'function': {
    'name': 'get_weather',
    'description': '',
    'parameters': {
        'properties': {
            'city': {
                'type': 'string'
            }
        },
        'required': ['city'],
        'type': 'object'
    }
  }
}
```

**结果字段说明：**

- （1） `type`：定义当前数据节点必须是什么数据类型。常见类型有 string, number, integer, boolean,object, array, null。object即是json对象。

- （2） `properties`：用于定义JSON 对象（Object）中可以包含哪些属性（键），以及每个属性对应的值类型和说明。

- （3） `required`：当 type为 "object"时使用，是一个数组，列出了对象中必须存在的属性名。

**问题：为什么不使用`@tool`装饰器修饰的函数，也可以理解为工具呢？**

查看 `convert_to_openai_tool` 底层源码：

```python
elif isinstance(function, langchain_core.tools.base.BaseTool):
    oai_function = cast("dict", _format_tool_to_openai_function(function))
elif callable(function):
    oai_function = cast("dict", _convert_python_function_to_openai_function(function))
```

相当于加了`@tool`修饰的函数走上面的分支，没有加`@tool`修饰的函数走下面的分支，后者会基于函数定义和`docstring`生成`pydantic`模式的描述，然后转换为规范的`tool_schema`。

#### 2. `description`说明

`convert_to_openai_tool` 会从 `docstring`(文档字符串) 加载工具的描述信息，上面的案例中，`docstring`为空，所以抽取的 `description` 为空。

`docstring`，文档字符串，使用三个双引号表示开始和结束。

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from rich import print as rprint

def get_weather(city: str):
    """
    天气查询工具
    """
   return f"{city}天气晴朗"

rprint(convert_to_openai_tool(get_weather))
```

输出

```json
{
 'type': 'function',
 'function': {
     'name': 'get_weather',
     'description': '天气查询工具',
     'parameters': {
         'properties': {
             'city': {
                 'type': 'string'
             }
         },
         'required': ['city'],
         'type': 'object'
     }
 }
}
```

#### 3. 参数说明

`convert_to_openai_tool` 会从 `docstring` 加载参数说明，这里的 `docstring` 必须遵循 `Google` 风格。

- Google 风格 docstring 说明：<https://google.github.io/styleguide/pyguide.html>

- Google 风格 docstring 示例：<https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html>

- Python docstring 通用约定：<https://peps.python.org/pep-0257/>

基础用法不必完整阅读规范，只需要按照下面的示例仿写即可。

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from rich import print as rprint

def get_weather(city: str):
    """
    天气查询工具

    Args:
        city: 城市名称
    """
    return f"{city}天气晴朗"

rprint(convert_to_openai_tool(get_weather))
```

使用 `Args:、 Returns:、 Raises:` 等关键字，这种方式可读性强。Agent通过工具的这些注释来理解工具的用途和调用时机，因此清晰、准确的文档字符串是工具能被正确调用的前提。

输出如下：

```json
{
 'type': 'function',
 'function': {
     'name': 'get_weather',
     'description': '天气查询工具',
     'parameters': {
         'properties': {
             'city': {
                 'description': '城市名称',
                 'type': 'string'
             }
         },
         'required': ['city'],
         'type': 'object'
     }
 }
}
```

AI 依赖 docstring 来理解工具。

```python
# ❌ 不好：太模糊
@tool
def tool1(x: str) -> str:
    """做一些事情"""
    ...
```

```python
# ✅ 好：清晰明确
@tool
def search_products(query: str) -> str:
    """
    在产品数据库中搜索产品

    Args:
        query: 搜索关键词，如"笔记本电脑"、"手机"

    Returns:
        产品列表的 JSON 字符串
    """
    ...
```

#### 4. 参数类型说明

参数类型来源于函数的类型注解。

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from rich import print as rprint

def get_weather(city):
    """
    天气查询工具
    """
    return f"{city}天气晴朗"

rprint(convert_to_openai_tool(get_weather))
```

输出如下

```json
{
 'type': 'function',
 'function': {
     'name': 'get_weather',
     'description': '天气查询工具',
     'parameters': {
         'properties': {
             'city': {}
      },
         'required': ['city'],
         'type': 'object'
     }
 }
}
```

删除了参数类型注解，则工具描述中不包含参数类型说明

注意：如果`docstring`中包含参数说明，则对应的参数必须有类型注解，否则报错

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from rich import print as rprint

def get_weather(city):
    """
    天气查询工具

    Args:
        city: 城市名称
    """
    print("天气晴朗")

rprint(convert_to_openai_tool(get_weather))
```

报错如下：

```text
Traceback...
ValueError: Arg city in docstring not found in function signature.
```

```python
from typing import Any, Dict
from langchain_core.utils.function_calling import convert_to_openai_tool
from rich import print as rprint


def get_weather(city: str) -> None:
    """天气查询工具

    Args:
        city: 城市名称
    """
    print("天气晴朗")


# 转换并输出 OpenAI Tool Schema
tool_schema: Dict[str, Any] = convert_to_openai_tool(get_weather)
rprint(tool_schema)
```

#### 5. 参数默认值说明

如果参数没有默认值，则会包含 在required对应的列表 中。

反之，则参数的描述信息会包含 default 字段，并且 不会出现在required列表 中。

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from rich import print as rprint

def get_weather(city: str="北京"):
    """
    天气查询工具

    Args:
        city: 城市名称
    """
    return f"{city}天气晴朗"

rprint(convert_to_openai_tool(get_weather))
```

输出如下

```json
{
 'type': 'function',
 'function': {
     'name': 'get_weather',
     'description': '天气查询工具',
     'parameters': {
         'properties': {
             'city': {
                 'default': '北京',
                 'description': '城市名称',
                 'type': 'string'
             }
         },
         'type': 'object'
     }
 }
}
```

目前只有一个参数，并且有默认值，所以required字段被移除了。

举例2：

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from rich import print as rprint

def get_weather(dt: str, city: str="北京"):
    """
    天气查询工具

    Args:
        dt: 日期
        city: 城市名称
    """
    return f"{city}天气晴朗"

rprint(convert_to_openai_tool(get_weather))
```

输出

```json
{
 'type': 'function',
 'function': {
     'name': 'get_weather',
     'description': '天气查询工具',
     'parameters': {
         'properties': {
             'dt': {
                 'description': '日期',
                 'type': 'string'
             },
             'city': {
                 'default': '北京',
                 'description': '城市名称',
                 'type': 'string'
             }
         },
         'required': [
             'dt'
         ],
         'type': 'object'
     }
 }
}
```

## 3、工具的定义方式2：使用@tool装饰器（推荐）🌟

使用 @tool 装饰器修饰，可以自动将普通 Python 函数转化为智能体可调用的工具。

此方式 最直接，代码量极少，非常适合快速验证想法或创建参数简单的工具。

### 3.1 自定义工具描述：`description`

#### 情况1：仅提供`docstring`信息

在`bind_tools()`调用时，先将函数封装为 `BaseTool` 类型的对象，再传递给 `convert_to_openai_tool`  函数，生成工具的描述。

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools import tool

@tool
def get_weather(city: str):
    return f"{city}天气晴朗"

print(convert_to_openai_tool(get_weather))
```

`@tool`会从 `docstring` 生成描述信息，同样要求遵循 `Google docstring` 规范。如果没有`docstring`则报错，如下。

```text
Traceback...
ValueError: Function must have a docstring if description not provided.
```

补充 `docstring`

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools import tool

@tool
def get_weather(city: str):
    """
    天气查询工具
    """
    return f"{city}天气晴朗"

print(convert_to_openai_tool(get_weather))
```

输出如下

```json
{
 'type': 'function',
 'function': {
     'name': 'get_weather',
     'description': '天气查询工具',
     'parameters': {
         'properties': {
             'city': {
                 'type': 'string'
             }
         },
         'required': [
             'city'
         ],
         'type': 'object'
     }
 }
}
```

#### 情况2：添加工具描述：`description`

`@tool` 的参数 `description` 可以更改工具描述，优先级高于 `docstring` 的函数说明

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools import tool
from rich import print as rprint

@tool(description="根据城市名称查询当日天气的工具")
def get_weather(city: str):
    """
    天气查询工具
    """
    return f"{city}天气晴朗"

rprint(convert_to_openai_tool(get_weather))
```

输出

```json
{
 'type': 'function',
 'function': {
     'name': 'get_weather',
     'description': '根据城市名称查询当日天气的工具',
     'parameters': {
         'properties': {
             'city': {
                 'type': 'string'
             }
         },
         'required': [
             'city'
         ],
         'type': 'object'
     }
 }
}
```

#### 情况3：解析`docstring：parse_docstring`

当我们没有向 `@tool`传递 `description` 参数时，默认情况下， `tool` 会将 `docstring` 整体视为`description`，如下

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from rich import print as rprint

@tool
def get_weather(city: str, units: str = "celsius", include_forecast: bool =
False) -> str:
    """
    获取当日天气，可选择是否同时查询未来五日天气预报

    Args:
        city: 城市
        units: 气温单位，可选：celsius-摄氏度，fahrenheit-华氏度
        include_forecast: 是否包含未来五日的天气预报
    """
    temp = 22 if units == "celsius" else 72
    result = f'{city}当天气温: {temp} {"摄氏度" if units == "celsius" else "华氏度"}'
    if include_forecast:
        result += "\n未来五天都是晴天"
    return result

rprint(convert_to_openai_tool(get_weather))
```

输出如下

```json
{
 'type': 'function',
 'function': {
     'name': 'get_weather',
     'description': '获取当日天气，可选择是否同时查询未来五日天气预报\n\nArgs:\n    city: 城市\n    units: 气温单位，可选：celsius-摄氏度，fahrenheit-华氏度\n    include_forecast: 是否包含未来五日的天气预报',
     'parameters': {
         'properties': {
             'city': {
                 'type': 'string'
             },
             'units': {
                 'default': 'celsius',
                 'type': 'string'
             },
             'include_forecast': {
                 'default': False,
                 'type': 'boolean'
             }
         },
         'required': [
             'city'
         ],
         'type': 'object'
     }
 }
}
```

通过将 parse_docstring 设置为True，docstring会被解析，填充到相应的字段描述中。

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from rich import print as rprint

@tool(parse_docstring=True)
def get_weather(city: str, units: str = "celsius", include_forecast: bool =
False) -> str:
    """
    获取当日天气，可选择是否同时查询未来五日天气预报

    Args:
        city: 城市
        units: 气温单位，可选：celsius-摄氏度，fahrenheit-华氏度
        include_forecast: 是否包含未来五日的天气预报
    """
    temp = 22 if units == "celsius" else 72
    result = f'{city}当天气温: {temp} {"摄氏度" if units == "celsius" else "华氏度"}'
    if include_forecast:
        result += "\n未来五天都是晴天"
    return result

rprint(convert_to_openai_tool(get_weather))
```

输出

```json
{
 'type': 'function',
 'function': {
     'name': 'get_weather',
     'description': '获取当日天气，可选择是否同时查询未来五日天气预报',
     'parameters': {
         'properties': {
             'city': {
                 'description': '城市',
                 'type': 'string'
             },
             'units': {
                 'default': 'celsius',
                 'description': '气温单位，可选：celsius-摄氏度，fahrenheit-华氏度',
                 'type': 'string'
             },
             'include_forecast': {
                 'default': 'False',
                 'description': '是否包含未来五日的天气预报',
                 'type': 'boolean'
             }
         },
         'required': [
             'city'
         ],
         'type': 'object'
     }
 }
}
```

要注意：不使用` @tool `装饰器时，`docstring`不合法会被视为普通文本，作为 `description`，但如果使用了` @tool `时 `docstring `不合法，将会抛出异常

```python
from langchain_core.utils.function_calling import convert_to_openai_tool

@tool(parse_docstring=True)
def get_weather(city: str, units: str = "celsius", include_forecast: bool =
False) -> str:
    """
    获取当日天气，可选择是否同时查询未来五日天气预报
    Args:
        city: 城市
        units: 气温单位，可选：celsius-摄氏度，fahrenheit-华氏度
        include_forecast: 是否包含未来五日的天气预报
    """
    temp = 22 if units == "celsius" else 72
    result = f'{city}当天气温: {temp} {"摄氏度" if units == "celsius" else "华氏度"}'
    if include_forecast:
        result += "\n未来五天都是晴天"
    return result

convert_to_openai_tool(get_weather)
```

报错如下

```text
Traceback...
ValueError: Found invalid Google-Style docstring.
```

### 3.2 更改工具名称：name_or_callable

默认情况，使用函数名作为工具名称，但可以向`@tool` 传参 `name_or_callable`，以更改工具名称。

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools import tool

@tool(name_or_callable="getWeather")
def get_weather(city: str):
    """
    天气查询工具
    """
    return f"{city}天气晴朗"

print(convert_to_openai_tool(get_weather))
```

输出如下

```json
{
 'type': 'function',
 'function': {
     'name': 'getWeather',
     'description': '天气查询工具',
     'parameters': {
         'properties': {
             'city': {
                 'type': 'string'
             }
         },
         'required': [
             'city'
         ],
         'type': 'object'
     }
 }
}
```

说明：@tool中参数name_or_callable名称可以省略

```python
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.tools import tool

@tool("getWeather")
def get_weather(city: str):
    """
    天气查询工具
    """
    print("天气晴朗")

print(convert_to_openai_tool(get_weather))
```

输出如下

```json
{
 'type': 'function',
 'function': {
     'name': 'getWeather',
     'description': '天气查询工具',
     'parameters': {
         'properties': {
             'city': {
                 'type': 'string'
             }
         },
         'required': [
             'city'
         ],
         'type': 'object'
     }
  }
}
```

说明：不要使用config或runtime作为参数名，这些是LangChain内部保留的。

开发中，习惯使用函数名作为工具名称，不推荐自定义工具名称。

### 3.3 自定义args_schema

#### 3.3.1 方式1：使用Pydantic模型定义

当工具的参数变得复杂，需要 枚举值、 范围限制 或 更复杂的业务逻辑验证 时，Pydantic 模型是理想的选择，提供强大的类型检查和数据验证。

使用Pydantic 的主要优势在于能够精确控制工具参数的格式和验证规则，让大模型更准确地理解如何调用工具。

##### 3.3.1.1 pydantic类型的定义

###### **① BaseModel基类**

通过继承核心基类 BaseModel 定义数据模型，从而声明字段结构、类型约束、默认值以0及校验规则。

```python
from pydantic import BaseModel

class WeatherInput(BaseModel):
    city: str

print(WeatherInput(city="北京"))
```

输出如下

```text
city='北京'
```

注意：BaseModel子类初始化时，不接收位置参数，字段值必须以关键字参数的形式传入，否则报错。

```python
from pydantic import BaseModel

class WeatherInput(BaseModel):
    city: str

print(WeatherInput("北京"))
```

报错如下

```text
TypeError
......
TypeError: BaseModel.__init__() takes 1 positional argument but 2 were
given
```

这是因为BaseModel的初始化函数签名如下

```python
def __init__(self, /, **data: Any) -> None:
```

由此可知，所有关键字参数都会被收集到字典 data 中，然后 data 会按照参数类型注解进行校验，失败时抛出异常。

###### **② Field**

`Field()`：用来“定制字段”的函数，可用于设置默认值、描述等。

举例1：设置默认值

```python
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(default= "北京")

print(WeatherInput())
```

输出

```text
city='北京'
```

举例2：设置参数的描述信息每个字段的 description 参数至关重要，它直接影响大模型理解参数含义的能力。

```python
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(
        default= "北京",
        description="城市"
    )
    include_forecast: bool = Field(
        default=False,
        description="是否包含未来五日天气预报"
    )

print(WeatherInput())
```

输出

```text
city='北京' include_forecast=False
```

###### **③ Literal**

可以使用 Literal类型限定参数为固定选项。

Literal：表示字段不能是任意某种类型的值，而只能是几个固定字面量之一。

举例1：

```python
from pydantic import BaseModel
from typing import Literal

class WeatherInput(BaseModel):
    city: str
    unit: Literal["celsius", "fahrenheit"]

print("===============> 合法 <===============")
print(WeatherInput(city="北京", unit="celsius"))

print("===============> 非法 <===============")
try:
    print(WeatherInput(city="北京", unit="kelvin"))
except Exception as e:
    print("报错类型：", type(e).__name__)
    print(e)
```

输出

```text
===============> 合法 <===============
city='北京' unit='celsius'
===============> 非法 <===============
报错类型： ValidationError
1 validation error for WeatherInput
unit
Input should be 'celsius' or 'fahrenheit' [type=literal_error,
input_value='kelvin', input_type=str]
 For further information visit
<https://errors.pydantic.dev/2.12/v/literal_error>
```

举例2：

```python
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(
        default= "北京",
        description="城市"
    )
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="气温单位"
    )
    include_forecast: bool = Field(
        default=False,
        description="是否包含未来五日天气预报"
    )

print(WeatherInput())
```

输出

```text
city='北京' unit='celsius' include_forecast=False
```

##### 3.3.1.2 使用Pydantic定义args_schema

通过 `@tool(args_schema=PydanticModelCls)` 将这个 `Pydantic` 模型与工具函数关联。

利用 Pydantic 的类型系统进行参数验证，当大模型需要调用工具前，Pydantic 会自动验证参数的类型和有效性。

举例：

```python
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

class WeatherInput(BaseModel):
    city: str = Field(
        default= "北京",
        description="城市"
    )
    unit: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="气温单位"
    )
    include_forecast: bool = Field(
        default=False,
        description="是否包含未来五日天气预报"
    )

@tool(args_schema=WeatherInput)
def get_weather(city: str, unit: str = "celsius", include_forecast: bool =
False) -> str:
    """获取当日天气，可选未来五日天气预报"""
    temp = 22 if unit == "celsius" else 72
    result = f'{city}当天气温: {temp} {"摄氏度" if unit == "celsius" else "华氏度"}'
    if include_forecast:
        result += "\n未来五天都是晴天"
    return result

convert_to_openai_tool(get_weather)
```

输出

```json
{
 'type': 'function',
 'function': {
     'name': 'get_weather',
     'description': '获取当日天气，可选未来五日天气预报',
     'parameters': {
         'properties': {
             'city': {
                 'default': '北京',
                 'description': '城市',
                 'type': 'string'
             },
             'unit': {
                 'default': 'celsius',
                 'description': '气温单位',
                 'enum': [
                     'celsius',
                     'fahrenheit'
                 ],
                 'type': 'string'
             },
             'include_forecast': {
                 'default': False,
                 'description': '是否包含未来五日天气预报',
                 'type': 'boolean'
             }
         },
         'type': 'object'
     }
 }
}
```

#### 3.3.2 方式2：使用Json Schema定义

在 LangChain 中，还可以直接使用 JSON Schema 字典 来定义工具的参数模式。这种方式提供了极大的灵活性。

因为工具参数模式可以基于数据库配置或用户输入在运行时动态生成，所以这种方式特别适合参数结构需要动态生成的场景。

举例：

```json
{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "获取当日天气，可选未来五日天气预报",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string"},
                "include_forecast": {"type": "boolean"}
            },
            "required": ["location", "units", "include_forecast"]
        },
    }
}
```

应该传递给args_schema的只有 parameters 对应的JSON字符串，即

```json
{
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string"},
        "include_forecast": {"type": "boolean"}
    },
    "required": ["location", "units", "include_forecast"]
}
```

举例：

通过 `@tool(args_schema=json_schema_dict)` 将一个符合 JSON Schema 标准的字典与工具函数关联。

```python
from langchain.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

weather_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string"},
        "include_forecast": {"type": "boolean"}
    },
    "required": ["location", "units", "include_forecast"]
}

@tool(args_schema=weather_schema)
def get_weather(city: str, unit: str = "celsius", include_forecast: bool =
False) -> str:
    """获取当日天气，可选未来五日天气预报"""
    temp = 22 if unit == "celsius" else 72
    result = f'{city}当天气温: {temp} {"摄氏度" if unit == "celsius" else "华氏度"}'
    if include_forecast:
        result += "\n未来五天都是晴天"
    return result

print(convert_to_openai_tool(get_weather))
```

输出

```json
{
 'type': 'function',
 'function': {
     'name': 'get_weather',
     'description': '获取当日天气，可选未来五日天气预报',
     'parameters': {
         'type': 'object',
         'properties': {
             'location': {
                 'type': 'string'
             },
             'units': {
                 'type': 'string'
             },
             'include_forecast': {
                 'type': 'boolean'
             }
         },
         'required': [
             'location',
             'units',
             'include_forecast'
         ]
     }
 }
}
```

## 4、工具的应用案例

### 4.1 案例1：使用args_schema

arg_schema给出明确的参数信息

模型初始化：

```python
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
# 从.env文件中加载环境变量
load_dotenv(override=True)

CLOSEAI_API_KEY = os.getenv("CLOSEAI_API_KEY")
CLOSEAI_BASE_URL = os.getenv("CLOSEAI_BASE_URL")

model = init_chat_model(
    model="gpt-5.4-mini",
    model_provider="openai",
    api_key=CLOSEAI_API_KEY,
    base_url=CLOSEAI_BASE_URL
)
```

工具的定义与模拟调用：

```python
from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

class WeatherSchema(BaseModel):
    city: str = Field(default="北京", description="城市名称")
    if_forecast: bool = Field(default=False, description="是否包含明日天气预报")

@tool("get_weather_and_forecast", description="查询当日天气，可以包含明日天气预报", args_schema=WeatherSchema)
def get_weather(city: str, if_forecast: bool):
    res = f"{city} 今天天气不错"
    if if_forecast:
        res += "\n明天也不错"
    return res

print(convert_to_openai_tool(get_weather))

# 工具绑定到模型
model_with_tools = model.bind_tools([get_weather])

messages = [HumanMessage("今天杭州天气如何？明天呢？")] # 维护消息列表
response = model_with_tools.invoke(messages)
messages.append(response)
tool_calls = response.tool_calls

for tool_call in tool_calls:
    if tool_call["name"] == "get_weather_and_forecast":
        tool_msg = get_weather.invoke(tool_call)
        messages.append(tool_msg)
# todo: more exceptions
final_response = model_with_tools.invoke(messages)
messages.append(final_response)
for msg in messages:
    msg.pretty_print()
```

```json
LLM
 |
 | tool_call
 ↓
判断工具是否存在
 |
 +----存在
 |       |
 |       ↓
 |    执行工具
 |
 +----不存在
         |
         ↓
     返回错误信息
```



输出

```text
{'type': 'function', 'function': {'name': 'get_weather_and_forecast',
'description': '查询当日天气，可以包含明日天气预报', 'parameters':
{'properties': {'city': {'default': '北京', 'description': '城市名称',
'type': 'string'}, 'if_forecast': {'default': False, 'description': '是
否包含明日天气预报', 'type': 'boolean'}}, 'type': 'object'}}}
================================ Human Message =================================

今天杭州天气如何？,明天呢？
================================== Ai Message ==================================
Tool Calls:
get_weather_and_forecast (call_c82UrCpAdVAqiHfl4OkZocbg)
Call ID: call_c82UrCpAdVAqiHfl4OkZocbg
Args:
 city: 杭州
 if_forecast: True
================================= Tool Message =================================
Name: get_weather_and_forecast

杭州 今天天气不错
明天也不错
================================== Ai Message ==================================

杭州今天：天气不错。
杭州明天：也不错。
```

### 4.2 案例2：撰写docstring

可以在docstring中撰写参数的描述信息，此时参数默认值和类型都要通过函数签名传递。

模型初始化：

```python
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os


# 从.env文件中加载环境变量
load_dotenv(override=True)

CLOSEAI_API_KEY = os.getenv("CLOSEAI_API_KEY")
CLOSEAI_BASE_URL = os.getenv("CLOSEAI_BASE_URL")

model = init_chat_model(
    model="gpt-5.4-mini",
    model_provider="openai",
    api_key=CLOSEAI_API_KEY,
    base_url=CLOSEAI_BASE_URL
)
```

工具的定义与模拟调用：

```python
from langchain.tools import tool
from langchain.messages import HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_tool


@tool("get_weather_and_forecast", parse_docstring=True)
def get_weather(city: str="北京", if_forecast: bool=False):
    """
    查询当日天气，可以包含明日天气预报

    Args:
         city: 城市名称
         if_forecast: 是否包含明日天气预报
    """
    res = f"{city} 今天天气不错"
    if if_forecast:
        res += "\n明天要下雨"
    return res

print(convert_to_openai_tool(get_weather))

model_with_tools = model.bind_tools([get_weather])

messages = [HumanMessage("今天杭州天气如何？明天呢？")]
response = model_with_tools.invoke(messages)
messages.append(response)
tool_calls = response.tool_calls

# 将工具调用的结果添加到消息列表中
for tool_call in tool_calls:
    if tool_call["name"] == "get_weather_and_forecast":
        # 返回值tool_msg类型是ToolMessage
        tool_msg = get_weather.invoke(tool_call)
        messages.append(tool_msg)

final_response = model_with_tools.invoke(messages)

messages.append(final_response)

for msg in messages:
    msg.pretty_print()
```

注意：要正确解析docstring，必须在@tool中将 parse_docstring 设置为 True。

输出

```text
{'type': 'function', 'function': {'name': 'get_weather_and_forecast',
'description': '查询当日天气，可以包含明日天气预报', 'parameters':
{'properties': {'city': {'default': '北京', 'description': '城市名称',
'type': 'string'}, 'if_forecast': {'default': False, 'description': '是
否包含明日天气预报', 'type': 'boolean'}}, 'type': 'object'}}}
================================ Human Message
=================================

今天杭州天气如何？明天呢？
================================== Ai Message
==================================
Tool Calls:
get_weather_and_forecast (call_y5vpFccj2DfWSAF6ng5dZOYH)
Call ID: call_y5vpFccj2DfWSAF6ng5dZOYH
 Args:
  city: 杭州
   if_forecast: True
================================= Tool Message
=================================
Name: get_weather_and_forecast

杭州 今天天气不错
明天要下雨
================================== Ai Message
==================================

杭州今天“天气不错”，明天“要下雨”。
如果你愿意，我也可以帮你继续看一下适合出行/穿衣的建议。
```

### 4.3 案例3：多工具调用

大模型调用工具是单次推理，即每次运行调用一个工具，当调用多个工具时，需要用户自己管理多次调用循环。

```python
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 1.定义工具
# 定义股票查询工具
@tool(parse_docstring=True)
def get_stock_price(company: str, timeframe: str = "today") -> str:
    """获取指定公司的股票价格信息

    Args:
        company: 公司名称（如：苹果公司, 微软公司, 谷歌公司）
        timeframe: 时间范围（today-今日, week-本周, month-本月）
    """
    # 模拟股票数据
    mock_data = {
        "苹果公司": {"today": 185.20, "week": 183.50, "month": 180.75},
        "微软公司": {"today": 415.86, "week": 412.30, "month": 405.42},
        "谷歌公司": {"today": 15.42, "week": 15.20, "month": 14.85}
    }

    if company in mock_data:
        price = mock_data[company].get(timeframe, "未知时间范围")
        return f"{company} {timeframe}价格: {price}美元"
    else:
        return f"未找到股票代码 {company} 的数据"


# 定义新闻搜索工具
@tool(parse_docstring=True)
def search_news(company: str) -> str:
    """搜索指定公司的财经新闻

    Args:
        company: 公司名称
    Returns:
        公司的财经新闻，每个新闻占一行
    """
    # 模拟新闻数据
    mock_news = {
        "苹果公司": [
            "苹果发布新款iPhone，股价上涨3%",
            "苹果与欧盟达成反垄断和解协议",
            "苹果将在印度扩大生产规模"
        ],
        "微软公司": [
            "微软Azure云业务季度增长超预期",
            "微软完成对Nuance的收购",
            "微软推出新一代AI助手Copilot"
        ],
        "谷歌公司": [
            "谷歌发布新AI模型，性能提升20%",
            "谷歌与OpenAI合作，开发新的AI助手",
            "谷歌在欧洲展开AI研究项目"
        ]
    }

    news_list = mock_news.get(company, [f"未找到{company}的相关新闻"])
    return "\n".join(news_list)


# rprint(convert_to_openai_tool(search_news))

# 2.初始化模型并绑定工具
tools = [get_stock_price, search_news]
model_with_tools = model.bind_tools(tools)

message_list = []
human_message = HumanMessage(content="苹果公司今天的股价是多少？最近有什么新闻？")
# human_message = HumanMessage(content="比较一下微软和苹果的股价")
# human_message = HumanMessage(content="腾讯最近有什么重大新闻？")
# human_message = HumanMessage(content="海水为什么是咸的？")
message_list.append(human_message)

# 3.工具调用
while True:
    response = model_with_tools.invoke(message_list)

    message_list.append(response)

    # 如果模型不需要调用工具，直接退出循环
    if not response.tool_calls:
        print("没有工具调用，直接返回答案")
        break

    # 如果有调用工具，处理工具调用响应
    # 4.开发者根据模型的响应，调用工具并获取结果
    for tool_call in response.tool_calls:
        if tool_call["name"] == "get_stock_price":
            stock_result = get_stock_price.invoke(tool_call)
            print("stock_result", stock_result)
            message_list.append(stock_result)
        if tool_call["name"] == "search_news":
            news_result = search_news.invoke(tool_call)
            print("news_result", news_result)
            message_list.append(news_result)


# print("response", response)
# print(response.content)

for msg in message_list:
    msg.pretty_print()
```

输出：

```text
stock_result content='苹果公司 today价格: 185.2美元'
name='get_stock_price' tool_call_id='call_cpGOhWce8rlIFSZ2G9w7ouON'
news_result content='苹果发布新款iPhone，股价上涨3%\n苹果与欧盟达成反垄断和解协
议\n苹果将在印度扩大生产规模' name='search_news'
tool_call_id='call_W9zjkAO8TNCmUbed2ld9tsxl'
没有工具调用，直接返回答案
================================ Human Message
=================================

苹果公司今天的股价是多少？最近有什么新闻？
================================== Ai Message
==================================
Tool Calls:
get_stock_price (call_cpGOhWce8rlIFSZ2G9w7ouON)
Call ID: call_cpGOhWce8rlIFSZ2G9w7ouON
Args:
 company: 苹果公司
 timeframe: today
search_news (call_W9zjkAO8TNCmUbed2ld9tsxl)
Call ID: call_W9zjkAO8TNCmUbed2ld9tsxl
Args:
 company: 苹果公司
================================= Tool Message
=================================
Name: get_stock_price

苹果公司 today价格: 185.2美元
================================= Tool Message
=================================
Name: search_news

苹果发布新款iPhone，股价上涨3%
苹果与欧盟达成反垄断和解协议
苹果将在印度扩大生产规模
================================== Ai Message
==================================

苹果公司今天股价是 **185.2 美元**。

最近新闻包括：
- 苹果发布了**新款 iPhone**，带动股价上涨约 **3%**
- 苹果与欧盟达成了**反垄断和解协议**
- 苹果计划在**印度扩大生产规模**
如果你愿意，我也可以帮你继续整理这些新闻对苹果股价的可能影响。
```

### 4.4 案例4：多工具调用

```python
from langchain.tools import tool
from langchain.messages import HumanMessage


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """
    获取当日天气

    Args:
        city: 城市名称
    """
    return f'{city}当天晴朗'

@tool(parse_docstring=True)
def get_news() -> str:
    """
    获取当日新闻
    """
    return "近期，受全球储蓄芯片短缺等多重因素影响，多地回收商称废旧手机回收市场迎来“火热潮”，回收价格普遍上涨，旧手机成“香饽饽”。"

model_with_tools = model.bind_tools([get_weather, get_news])

messages = [
    HumanMessage("今天杭州天气如何？今天新闻是什么？别瞎编")
]

response = model_with_tools.invoke(messages)
response.pretty_print()
```

输出

```text
==================================•[1m Ai Message•[0m==================================

我来帮您查询杭州的天气和今日新闻。
Tool Calls:
get_weather (call_00_uspkbqR7N7wIewhr8hHZAXoM)
Call ID: call_00_uspkbqR7N7wIewhr8hHZAXoM
Args:
city: 杭州
get_news (call_01_ksyZoO2kXFBXXp1dH4dnxZAN)
Call ID: call_01_ksyZoO2kXFBXXp1dH4dnxZAN
Args:
```

此时可以遍历tool_calls列表，挨个调用工具。

```python
messages.append(response)
for tool_call in response.tool_calls:
    if tool_call["name"] == "get_weather":
        tool_msg = get_weather.invoke(tool_call)
        print(tool_msg)
        messages.append(tool_msg)
    elif tool_call["name"] == "get_news":
        tool_msg = get_news.invoke(tool_call)
        print(tool_msg)
        messages.append(tool_msg)
    else:
        raise Exception("不存在的工具")

final_response = model.invoke(messages)
messages.append(final_response)

for msg in messages:
    msg.pretty_print()
```

输出

```text
content='杭州当天晴朗' name='get_weather'
tool_call_id='call_00_PhpRzVvHYLkqOgQMj4keeAso'
content='近期，受全球储蓄芯片短缺等多重因素影响，多地回收商称废旧手机回收市场迎来“火
热潮”，回收价格普遍上涨，旧手机成“香饽饽”。' name='get_news'
tool_call_id='call_01_ALWxzasKCDJzm7fgwiXVBorg'
================================•[1m Human Message•[0m=================================

今天杭州天气如何？今天新闻是什么？别瞎编
==================================•[1m Ai Message•[0m==================================

我来帮您查询杭州的天气和今天的新闻。
Tool Calls:
get_weather (call_00_PhpRzVvHYLkqOgQMj4keeAso)
Call ID: call_00_PhpRzVvHYLkqOgQMj4keeAso
Args:
city: 杭州
get_news (call_01_ALWxzasKCDJzm7fgwiXVBorg)
Call ID: call_01_ALWxzasKCDJzm7fgwiXVBorg
Args:
=================================•[1m Tool Message•[0m=================================
Name: get_weather

杭州当天晴朗
=================================•[1m Tool Message•[0m=================================
Name: get_news

近期，受全球储蓄芯片短缺等多重因素影响，多地回收商称废旧手机回收市场迎来“火热潮”，回
收价格普遍上涨，旧手机成“香饽饽”。
==================================•[1m Ai Message•[0m==================================

根据查询结果：
**杭州天气**：今天杭州天气晴朗。

**今日新闻**：近期，受全球芯片短缺等多重因素影响，多地回收商称废旧手机回收市场迎来“火
热潮”，回收价格普遍上涨，旧手机成“香饽饽”。

以上信息基于实时查询，供您参考。
```

## 5、拓展：强制使用工具

### 5.1 tool_choice参数说明

bind_tools 可以传递参数 tool_choice，用于控制是否强制使用工具。

该字段最终会作为 payload 的 tool_choice 字段传递给模型，OpenAI和Deepseek的官方API服务对于tool_choice 的取值做了相同的规定。

OpenAI官方文档

![image-20260802184610676](https://i.postimg.cc/zzS0HHBF/image-20260802184610676.png?dl=1)

Deepseek官方文档

![image-20260802184602547](https://i.postimg.cc/gzPg0HYW/image-20260802184602547.png?dl=1)

none：模型不会调用任何工具。

auto： 默认值，模型可以自主决定不调用或调用任意数量的工具。

required：模型必须调用工具，数量不限。此外， tool_choice 还支持传递 any，等价于 required。

### 5.2 none值举例

模型初始化：

```python
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# 从.env文件中加载环境变量
load_dotenv(override=True)

CLOSEAI_API_KEY = os.getenv("CLOSEAI_API_KEY")
CLOSEAI_BASE_URL = os.getenv("CLOSEAI_BASE_URL")

model = init_chat_model(
    model="gpt-5.4-mini",
    model_provider="openai",
    api_key=CLOSEAI_API_KEY,
    base_url=CLOSEAI_BASE_URL
)
```

举例：

```python
from langchain.tools import tool
from langchain.messages import HumanMessage


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """
    获取当日天气

    Args:
        city: 城市名称
    """
    return f'{city}当天晴朗'

model_with_tools = model.bind_tools([get_weather], tool_choice="none")

messages = [
    HumanMessage("今天北京天气如何？别瞎编")
]

response = model_with_tools.invoke(messages)
response.pretty_print()
```

用户要求查询天气，并提供了天气查询工具，但模型不会调用。

输出

```text
==================================•[1m Ai Message•[0m==================================

我无法直接获取实时天气数据。
如果您需要查询今天北京的准确天气情况，建议您：
1. 打开手机天气应用（如苹果天气、墨迹天气等）。
2. 在搜索引擎（如百度、谷歌）搜索“北京天气”。
3. 查看中国气象局官网（<http://www.weather.com.cn>）或权威天气平台。

这样可以确保您获得最准确的实时信息。
```

### 5.3 auto值举例

tool_choice 的默认值就是 auto。

**举例1：需要调用工具的场景**

```python
from langchain.tools import tool
from langchain.messages import HumanMessage


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """
    获取当日天气

    Args:
        city: 城市名称
    """
    return f'{city}当天晴朗'

model_with_tools = model.bind_tools([get_weather], tool_choice="auto")

messages = [
    HumanMessage("今天杭州天气如何？别瞎编")
]

response = model_with_tools.invoke(messages)
response.pretty_print()
```

输出

```text
==================================•[1m Ai Message•[0m==================================

我来帮您查询杭州今天的天气情况。
Tool Calls:
get_weather (call_00_yMeVRVUhjsGETejnWXdW4Hhp)
Call ID: call_00_yMeVRVUhjsGETejnWXdW4Hhp
Args:
 city: 杭州
```

**举例2：不需要调用工具的场景**

```python
from langchain.tools import tool
from langchain.messages import HumanMessage
@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """
    获取当日天气

    Args:
        city: 城市名称
    """
    return f'{city}当天晴朗'

model_with_tools = model.bind_tools([get_weather], tool_choice="auto")

messages = [
    HumanMessage("你好啊")
]

response = model_with_tools.invoke(messages)
response.pretty_print()
```

输出

```text
==================================•[1m Ai Message•[0m==================================

你好！很高兴见到你！有什么我可以帮助你的吗？
```

### 5.4 required值举例

**举例1：需要调用工具的场景**

```python
from langchain.tools import tool
from langchain.messages import HumanMessage

@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """
    获取当日天气

    Args:
        city: 城市名称
    """
    return f'{city}当天晴朗'

model_with_tools = model.bind_tools([get_weather], tool_choice="required")

messages = [
    HumanMessage("今天杭州天气如何？别瞎编")
]

response = model_with_tools.invoke(messages)
response.pretty_print()
```

输出

```text
==================================•[1m Ai Message•[0m==================================
Tool Calls:
get_weather (call_00_7JaeJXjd5Dc4GT9YIDV0ayJU)
Call ID: call_00_7JaeJXjd5Dc4GT9YIDV0ayJU
Args:
 city: 杭州
```

**举例2：不需要调用工具的场景**

```python
from langchain.tools import tool
from langchain.messages import HumanMessage


@tool(parse_docstring=True)
def get_weather(city: str) -> str:
    """
    获取当日天气

    Args:
        city: 城市名称
    """
    return f'{city}当天晴朗'

model_with_tools = model.bind_tools([get_weather], tool_choice="required")

messages = [
    HumanMessage("你好啊")
]

response = model_with_tools.invoke(messages)
response.pretty_print()
```

即便此时不需要，模型依然会调用工具

输出

```text
==================================•[1m Ai Message•[0m==================================
Tool Calls:
get_weather (call_00_4bLIBRda88ZcvFL70f7XOEmg)
Call ID: call_00_4bLIBRda88ZcvFL70f7XOEmg
Args:
 city: 北京
```

此外， any 行为和 required 一致。举例：略

### 5.5 强制调用特定的工具

某些场景下我们希望调用特定的工具，仍然可以用 tool_choice 解决。

```python
from langchain.tools import tool
from langchain.messages import HumanMessage
@tool(parse_docstring=True)
def get_weather1(city: str) -> str:
    """
    获取当日天气

    Args:
        city: 城市名称
    """
    return f'{city}当天晴朗'

@tool(parse_docstring=True)
def get_weather2(city: str) -> str:
    """
    获取当日天气

    Args:
        city: 城市名称
    """
    return f'{city}当天晴朗'

model_with_tools = model.bind_tools([get_weather1, get_weather2],
tool_choice="get_weather2")

messages = [
    HumanMessage("杭州今天天气如何？")
]

response = model_with_tools.invoke(messages)
response.pretty_print()
```

输出

```text
==================================•[1m Ai Message•[0m==================================
Tool Calls:
get_weather2 (call_00_mO5sCOWWnE3bWfNXtNhOjPNq)
Call ID: call_00_mO5sCOWWnE3bWfNXtNhOjPNq
Args:
 city: 杭州
```

反过来指定 get_weather1

```python
from langchain.tools import tool
from langchain.messages import HumanMessage


@tool(parse_docstring=True)
def get_weather1(city: str) -> str:
    """
    获取当日天气

    Args:
        city: 城市名称
    """
    return f'{city}当天晴朗'
@tool(parse_docstring=True)
def get_weather2(city: str) -> str:
    """
    获取当日天气

    Args:
        city: 城市名称
    """
    return f'{city}当天晴朗'

model_with_tools = model.bind_tools([get_weather1, get_weather2], tool_choice="get_weather1")

messages = [
    HumanMessage("杭州今天天气如何？")
]

response = model_with_tools.invoke(messages)
response.pretty_print()
```

输出

```text
==================================•[1m Ai Message•[0m==================================
Tool Calls:
get_weather1 (call_00_NK8ukyIAIzFL9FK1G3W4sK0s)
Call ID: call_00_NK8ukyIAIzFL9FK1G3W4sK0s
Args:
 city: 杭州
```

## 6、实践经验总结

### **1. 清晰的描述**

```python
# ✅ 好
@tool(parse_docstring=True)
def search_flights(origin: str, destination: str, date: str) -> str:
    """
    搜索航班信息

    Args:
        origin: 出发城市，如"北京"
        destination: 目的地城市，如"上海"
        date: 出发日期，格式 YYYY-MM-DD

    Returns:
        可用航班的 JSON 列表
    """
```

### **2. 功能单一**

```python
# ❌ 不好：一个工具做太多事
@tool
def do_everything(action: str, data: str) -> str:
    """做各种事情"""
    if action == "weather": ...
    elif action == "calculate": ...
    elif action == "search": ...

# ✅ 好：每个工具做一件事
@tool
def get_weather(city: str) -> str:
    """获取天气"""
    ...

@tool
def calculator(operation: str, a: float, b: float) -> str:
    """计算"""
    ...
```

### **3. 如何处理工具失败？**

三层防护：

#### 第1层：工具内部处理

```python
@tool
def divide(a: float, b: float) -> str:
    """
    除法计算

    Args:
        a: 被除数
        b: 除数
    """
    try:
        if b == 0:
            return "错误：除数不能为零"
        result = a / b
        return f"{a} / {b} = {result}"
    except Exception as e:
        return f"计算错误：{e}"
```

#### 第2层：Agent 级重试（使用 prompt）

```python
agent = create_agent(
    model=model,
    tools=[...],
    prompt="如果工具失败，尝试使用其他方法解决问题。"
)
```

#### 第3层：调用级重试

在大模型应用（如 LangChain）中，网络请求和外部工具调用是最容易掉链子的地方。 @retry  就像是一个 容错保险。

```python
from tenacity import retry, stop_after_attempt

# 1. 配置重试规则：如果失败，最多尝试 3 次（即第 1 次正常调用 + 2 次重试）
@retry(stop=stop_after_attempt(3))
def call_agent(question):
    # 2. 核心业务逻辑：调用 LangChain 的 Agent
    return agent.invoke({"messages": [{"role": "user", "content":
question}]})
```

它的工作流程：

① 你调用 call_agent("你好")。

② 程序进入函数，执行 agent.invoke(...)。

③ 如果执行成功：正常返回结果， @retry  什么都不做。

④ 如果执行失败（报错）： @retry  会拦截这个错误，不让程序直接崩溃。它会默默地帮你再次触发agent.invoke(...)。

⑤ 如果连续 3 次都报错：它终于放弃了，把第 3 次的报错真正抛出来，程序此时才会报错中止。

#### **4. 返回字符串**

```python
# ✅ 好：返回字符串
@tool
def get_user_info(user_id: str) -> str:
    """获取用户信息"""
    user = {"id": user_id, "name": "张三"}
    return json.dumps(user, ensure_ascii=False)  # 转成 JSON 字符串


# ❌ 不好：返回字典（某些情况可能有问题）
@tool
def get_user_info(user_id: str) -> dict:
    """获取用户信息"""
    return {"id": user_id, "name": "张三"}
```

在编写传统的 Python 代码时，返回字典（dict）显然更方便后续代码处理。但在 LangChain 的工具（Tools）生态中，强烈建议工具返回字符串（str）。因为：

1）大模型（LLM）的本质只吃“文本”

2）避免大模型“胡思乱想”（乱码与格式问题）

如果你返回一个包含中文的字典 {"name": "张三"}，LangChain 在强制将其转换为字符串时，默认可能会采用 Unicode 编码，变成 {"name": "\u5f20\u4e09"}。

大模型虽然能理解 Unicode，但极易受到干扰。直接看到中文 张三 的大模型，和看到\u5f20\u4e09  的大模型，其输出的稳定性和准确率是有差距的。通过手动 json.dumps(...,ensure_ascii=False)，你确保了喂给大模型的是最干净、最直观的纯文本。

json.dumps()是 Python 标准库 json模块中的函数，用于将 Python 对象（如字典、列表）序列化成一个 JSON 格式的字符串。

ensure_ascii=False : 这个参数默认值为 True，表示所有非 ASCII 字符（如中文）会被转换成\uXXXX形式的转义序列。设置为 False后，中文、表情符号等字符就能在 JSON 字符串中正常显示，而不是一堆乱码。

#### **5. 选择同步 vs 异步**

- 同步工具：简单场景，CPU 密集型任务

- 异步工具：IO 密集型（API 调用、数据库、文件操作）

```python
# 同步
@tool
def sync_tool(x: str) -> str:
    return process(x)

# 异步
@tool
async def async_tool(x: str) -> str:
    return await async_process(x)
```
