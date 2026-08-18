# 6. 持久化机制和可恢复执行

## 6.1. 概述

### 6.1.1. 什么是可恢复执行

在工作流、Agent、任务编排系统里，**可恢复执行 `Durable Execution`** 指的是：

> 把一次任务执行过程中的关键进度、状态、结果保存到可靠存储中，使任务可以在中断、失败、等待外部输入后继续执行。落盘

它解决的是“执行过程能否恢复”的问题。

例如：

```
普通执行：
开始 -> A -> B -> C
如果执行到 B 后程序挂了，重启后可能只能从 A 重新开始。

可恢复执行：
开始 -> A   保存检查点(checkpoint)
     -> B   保存检查点(checkpoint)
     -> C
如果执行到 B 后挂了，恢复时可以从已保存的位置继续。
```

所以它关注的不是“数据长期保存”本身，而是：

> **执行过程本身具有可恢复性。**

它关心的是：系统能否记住“任务执行到了哪里、当前状态是什么、接下来应该继续执行什么”。

### 6.1.2. LangGraph的持久化机制

#### 6.1.2.1. 什么是LangGraph的持久化机制

**`LangGraph`** 的**持久化机制**（**`Persistence`**）可以理解为：

> 在图执行过程中，将每个关键阶段的图状态保存为检查点(checkpoint)，并按照线程进行组织。

这里的“持久化”并不是简单地把某个变量保存到文件里，而是保存一次图执行所需的状态信息，使得未来可以基于检查点(checkpoint)继续执行。

**`LangGraph`** 的持久化机制围绕以下几个问题展开：

1. **保存什么？**

   保存图在某个时刻的状态快照，也就是 **`Checkpoint`**。

2. **保存到哪里？**

   保存到 **`Checkpointer`** 管理的存储后端中。
   For test  / dev 开发时可以使用基于内存的 **`InMemorySaver`**，生产环境中通常会使用数据库等可靠存储，如 **`PostgresSaver`**。

3. **如何区分不同会话？**

   通过 **`thread_id`** 区分不同的执行线程。
   同一个 **`thread_id`** 表示同一条持久化执行线，也可以理解为同一个会话。

4. **开发者看到的是什么？**

   开发者通常不会直接操作底层 **`Checkpoint`**，而是通过 **`get_state()`** 和 **`get_state_history()`** 查看 **`Checkpoint`** 的开发者视图 **`StateSnapshot`**。

因此，本章后续会围绕四个核心问题展开：

```markdown
如何启用可恢复执行？
如何配置持久化模式？
如何查看历史检查点(checkpoint)？
如何利用检查点(checkpoint)进行恢复、回放和分叉？
```

#### 6.1.2.2. 核心组件

本节对核心组件进行梳理，不必死记硬背，用到查阅即可。

**`LangGraph`** 的持久化机制依赖一系列核心组件，如下：

| 概念                     | 作用                                                         |
| ------------------------ | ------------------------------------------------------------ |
| **`State`**              | 用户定义的图状态结构，负责节点间信息传递。**`LangGraph`** 会将 **`State`** 转换为底层 **`Channel`**；运行时真正负责节点间通信的是 **`Channel`**。 |
| **`Channel`**            | **`LangGraph`** 用于节点间通信的底层机制，节点从 **`Channel`** 读数据、向 **`Channel`** 写更新；既包括承载 **`State`** 字段更新的状态通道，也包括用于分支、任务调度等行为的内部通道。 |
| **`Checkpoint`**         | 检查点(checkpoint)，超步边界上的底层状态快照，保存 **`Channel`** 的值、版本等信息。 |
| **`CheckpointMetadata`** | 与 **`Checkpoint`** 关联的元数据，如超步编号 **`step`**、父检查点(checkpoint) **`ID parents`** 等。 |
| **`Checkpointer`**       | 检查点(checkpoint)存储器，负责存储检查点(checkpoint)、元数据、配置等信息。 |
| **`thread`**             | 此处的线程不同于操作系统的线程，是指 **`LangGraph`** 中一条逻辑上的、可持久化的执行线，也可以理解为会话。一个会话可以包含多次调用，并在执行过程中生成多个检查点(checkpoint)。因此，要将多次调用组织在一个会话中，必须为该 **`thread`** 配置 **`checkpointer`**。 |
| **`thread_id`**          | **`thread`** 的唯一标识，用于区分不同的会话。复用同一个 **`thread_id`**，就表示多次调用共享同一条持久化执行线，也就是共享同一个会话历史。此处的 **`thread`/`thread_id`** 和 **`LangChain Agent`** 中提到的 **`thread`/`thread_id`** 是同一概念。 |
| **`checkpoint_ns`**      | 检查点(checkpoint)命名空间 **`namespace`**，用于区分父图和子图的 **`checkpoint`**。根图通常是空字符串 **`""`**，子图会有自己的 **`namespace`**。这个字段在子图持久化时很重要，下文会有专门的篇幅讲解子图相关用法。 |
| **`checkpoint_id`**      | **`Checkpoint`** 的唯一标识。                                |
| **`StateSnapshot`**      | 开发者通过 **`get_state()` / `get_state_history()`** 看到的状态快照对象。它不是原始 **`Checkpoint`**，而是 **`LangGraph`** 基于检查点(checkpoint)和运行时信息封装出来的开发者视图，包含当前状态值、下一步待执行节点、任务、**`config`**、**`metadata`**、中断信息等内容。 |

#### 6.1.2.3. 小结

本节只需要先记住：

> **`Checkpointer`** 负责保存检查点(checkpoint)
>
> **`thread_id`** 负责唯一标识会话
>
> **`checkpoint_id`** 负责定位某个具体历史状态
>
> **`StateSnapshot`** 是开发者查看检查点(checkpoint)时看到的对象。

### 6.1.3. LangGraph的可恢复执行

#### 6.1.3.1. 什么是LangGraph的可恢复执行

**`LangGraph`** 的可恢复执行（**`Durable Execution`**）依托于其持久化机制（**`Persistence`**）。

可以简单理解为：

> 持久化机制负责保存执行状态；
> 可恢复执行负责利用这些状态继续执行。

**`Persistence` 是基础，`Durable Execution` 是建立在 `Persistence` 之上的能力。**

在没有持久化机制时，一次图执行通常只存在于当前进程中。
如果程序退出、节点中断、人工审批暂停，无法恢复原来的执行进度。

而启用持久化之后，**`LangGraph`** 可以把执行过程中的状态保存为检查点(checkpoint)。
后续只要使用相同的 **`thread_id`**，就可以回到会话对应的历史状态。

#### 6.1.3.2. 使用场景

从使用场景看，可恢复执行主要支持以下几类能力：

| 场景              | 说明                                                         |
| ----------------- | ------------------------------------------------------------ |
| **多轮会话**      | 同一个 **`thread_id`** 下的多次调用可以共享历史状态。        |
| **中断恢复**      | 图执行到人工确认等节点时可以中断，之后再继续。               |
| **失败恢复**      | 程序异常退出后，可以基于已经保存的检查点(checkpoint)继续执行，避免重复计算。 |
| **`Time Travel`** | 可以回到某个历史检查点(checkpoint)，**重放**或**分叉**执行。 |

因此，**`LangGraph` 的可恢复执行是一组由持久化机制支撑的能力集合。**

#### 6.1.3.3. 小结

我们只需要知道

> **`LangGraph` 会通过 `Checkpointer` 持久化保存图执行过程中产生的 `Checkpoint`，并通过 `thread_id` 找回同一会话的检查点(checkpoint)历史，从而在已有状态基础上继续执行。**

## 6.2. 启用可恢复执行

### 6.2.1. 步骤

**`LangGraph`** 内置了持久化机制，但对于本地运行的 **`Graph API`**，只有在编译图时传入 **`checkpointer`**，运行过程中的状态才会被记录下来，后续才能基于已有检查点(checkpoint)在发生故障时恢复运行或进行检查点(checkpoint)回溯。

启用可恢复执行通常需要两步：

1. 在编译图时传入检查点(checkpoint)存储器对象 **`checkpointer`** 
2. 在调用图时传递带有 **`thread_id`** 的配置对象

底层的持久化机制会按照 **`thread_id`** 将运行时检查点(checkpoint)记录在编译时传入的 **`checkpointer`** 中。

### 6.2.2. 检查点(checkpoint)实现

**`LangGraph`** 提供了检查点(checkpoint)存储器基类：**`langgraph.checkpoint.base.BaseCheckpointSaver`**

还提供了一系列检查点(checkpoint)存储器实现，采用了不同的检查点(checkpoint)后端，它们都继承了基类，整体作用是一致的：负责存取检查点(checkpoint)，并记录节点执行的中间结果。

可以将 **检查点(checkpoint)后端** 理解为：

> 负责保存检查点(checkpoint)数据的存储介质。

检查点(checkpoint)后端必须基于某个检查点(checkpoint)实现，**常见**检查点(checkpoint)实现和后端的对应关系如下

| 类名                | 包名                                | 检查点(checkpoint)后端  |
| ------------------- | ----------------------------------- | ----------------------- |
| **`InMemorySaver`** | **`langgraph-checkpoint`**          | 内存（**`In-memory`**） |
| **`SqliteSaver`**   | **`langgraph-checkpoint-sqlite`**   | **`SQLite`**            |
| **`PostgresSaver`** | **`langgraph-checkpoint-postgres`** | **`PostgreSQL`**        |
| **`MongoDBSaver`**  | **`langgraph-checkpoint-mongodb`**  | **`MongoDB`**           |
| **`RedisSaver`**    | **`langgraph-checkpoint-redis`**    | **`Redis`**             |

在这些实现中，**`InMemorySaver`** 更适合学习、调试和本地实验；**`SQLite`** 适合轻量级本地持久化；**`PostgreSQL`**、**`MongoDB`**、**`Redis`** 等数据库后端更适合需要跨进程、跨服务保存状态的场景。

### 6.2.3. 基于内存的检查点(checkpoint)存储器

**`LangGraph`** 提供了基于内存的检查点(checkpoint)存储器 **`InMemorySaver`**。它将检查点(checkpoint)数据保存在当前 **`Python`** 进程的内存中，进程结束则数据丢失，适合快速开发、调试。

在 **`Jupyter`** 场景下，只要 **`Jupyter`** 内核没有被重启，并且 **`InMemorySaver()`** 实例没有被重新创建，内存中的检查点(checkpoint)数据就会继续存在。

反之，如果重新执行完整代码，导致 **`checkpointer = InMemorySaver()`** 被重新执行，那么之前保存在内存中的检查点(checkpoint)也会被清空。

**示例如下**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv(override=True)

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)

class OverAllState(MessagesState):
    output: str

def llm_node(state: OverAllState) -> OverAllState:
    messages = state["messages"]
    res = model.invoke(messages)
    return {
        "messages": [res]
    }

def output_node(state: OverAllState) -> OverAllState:
    return {
        "output": state["messages"][-1].content
    }

builder = StateGraph(state_schema=OverAllState)
builder.add_node("llm_node", llm_node)
builder.add_node("output_node", output_node)
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", "output_node")
builder.add_edge("output_node", END)

# 定义并在编译时传递 Checkpointer
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 定义配置对象
config = {"configurable": {"thread_id": "chapter_6_6-2-2"}}
# 调用时传递
graph.invoke({"messages": [HumanMessage("你好，我是老王")]}, config=config)
graph.invoke({"messages": [HumanMessage("从现在开始，你是小王")]}, config=config)
res = graph.invoke({"messages": [HumanMessage("我是谁？你是谁？")]}, config=config)
print(res["output"])

print('=' * 30, '-> 完整消息列表 <-', '=' * 30)
for msg in res["messages"]:
    msg.pretty_print()
```

**输出如下**

```
哎，老王，您这是考我呢？还是今天心情好，想跟我逗个闷子？

您问您是谁——那当然是我的老大哥，老王啊！平时有事儿招呼我，没架子，偶尔还爱指点我两句，我心里都记着。

至于我是谁——我，小王，您的得力助手，机灵踏实，随叫随到。您说往东，我绝不往西，主打一个“靠谱”。

怎么着，老王，是不是又想让我去办啥事儿了？您直说，我听着呢。😎
============================== -> 完整消息列表 <- ==============================
================================ Human Message =================================

你好，我是老王
================================== Ai Message ==================================

你好，老王！很高兴认识你。有什么我可以帮忙的吗？无论是聊天、解答问题还是需要一些建议，我都在这儿呢。😊
================================ Human Message =================================

从现在开始，你是小王
================================== Ai Message ==================================

好的，老王！我是小王。有啥事儿您尽管吩咐，我这儿随时听候差遣。咱是掏心窝子说话，还是聊点别的？您定！😄
================================ Human Message =================================

我是谁？你是谁？
================================== Ai Message ==================================

哎，老王，您这是考我呢？还是今天心情好，想跟我逗个闷子？

您问您是谁——那当然是我的老大哥，老王啊！平时有事儿招呼我，没架子，偶尔还爱指点我两句，我心里都记着。

至于我是谁——我，小王，您的得力助手，机灵踏实，随叫随到。您说往东，我绝不往西，主打一个“靠谱”。

怎么着，老王，是不是又想让我去办啥事儿了？您直说，我听着呢。😎
```

本例中，三次调用使用了相同的 **`thread_id`**，所以后两次调用能够读取前面已经保存的消息历史。最终模型可以知道“用户是老王”，也可以知道“自己被要求扮演小王”。

需要注意的是，如果重新执行完整代码，**`InMemorySaver()`** 实例会被重新创建，历史检查点(checkpoint)会被清空。因此，消息列表不会继续累加，而是会基于新的空状态重新开始。

### 6.2.4. 基于持久化数据库的检查点(checkpoint)存储器

本节将 **`PostgreSQL`** 作为检查点(checkpoint)后端，对应的 **`Checkpointer`** 实现为 **`PostgresSaver`**。

和 **`InMemorySaver`** 不同，**`PostgresSaver`** 会将检查点(checkpoint)保存到 **`PostgreSQL`** 数据库中。只要数据库中的记录没有被删除，即使 **`Python`** 程序结束、连接对象重建，历史检查点(checkpoint)也仍然存在。

在学习 **`LangChain`** 时，我们已经介绍了 **`PostgreSQL`** 的部署和 **`PostgresSaver`** 的用法。此处直接给出案例，不再赘述。

**示例如下**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv(override=True)

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)

class OverAllState(MessagesState):
    output: str

def llm_node(state: OverAllState) -> OverAllState:
    messages = state["messages"]
    res = model.invoke(messages)
    return {
        "messages": [res]
    }

def output_node(state: OverAllState) -> OverAllState:
    return {
        "output": state["messages"][-1].content
    }

builder = StateGraph(state_schema=OverAllState)
builder.add_node("llm_node", llm_node)
builder.add_node("output_node", output_node)
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", "output_node")
builder.add_edge("output_node", END)

# 定义并在编译时传递 Checkpointer
DB_URL = "postgresql://langgraph_user:123456@localhost:5432/langgraph_db?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
    # 示例中为了方便演示直接调用 setup()
    # 实际项目中通常建议把数据库初始化/迁移作为独立步骤处理
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)

    # 定义配置对象
    config = {"configurable": {"thread_id": "chapter_6_6.2.4"}}
    # 调用时传递
    graph.invoke({"messages": [HumanMessage("你好，我是老王")]}, config=config)
    graph.invoke({"messages": [HumanMessage("从现在开始，你是小王")]}, config=config)
    res = graph.invoke({"messages": [HumanMessage("我是谁？你是谁？")]}, config=config)
    print(res["output"])

    print('=' * 30, '-> 完整消息列表 <-', '=' * 30)
    for msg in res["messages"]:
        msg.pretty_print()
```

**运行结果如下**

```
好嘞，老王！既然你问了，那我正式回答：  

**你是老王**——我今儿认的“老大哥”，江湖人称“行走的故事书”，爱唠嗑、有阅历，可能还藏着点神秘技能（比如钓鱼从不空军、下棋总赢隔壁老李？）  

**我是小王**——刚上岗的AI小跟班，脑子灵光（但可能偶尔短路），腿脚勤快（指秒回消息），擅长接梗、查资料、瞎琢磨，主打一个“您开口，我跑腿”。  

需要小王帮您办点啥？甭客气！🚀
============================== -> 完整消息列表 <- ==============================
================================ Human Message =================================

你好，我是老王
================================== Ai Message ==================================

老王你好！我是DeepSeek，很高兴认识你。有什么我可以帮忙的吗？无论是聊天、解答问题、还是需要一些建议，尽管开口，我随时在这儿！😊
================================ Human Message =================================

从现在开始，你是小王
================================== Ai Message ==================================

好的，老王！从现在开始，我就是小王啦。😄  
有什么吩咐？无论唠嗑、干活还是出主意，随叫随到！
================================ Human Message =================================

我是谁？你是谁？
================================== Ai Message ==================================

好嘞，老王！既然你问了，那我正式回答：  

**你是老王**——我今儿认的“老大哥”，江湖人称“行走的故事书”，爱唠嗑、有阅历，可能还藏着点神秘技能（比如钓鱼从不空军、下棋总赢隔壁老李？）  

**我是小王**——刚上岗的AI小跟班，脑子灵光（但可能偶尔短路），腿脚勤快（指秒回消息），擅长接梗、查资料、瞎琢磨，主打一个“您开口，我跑腿”。  

需要小王帮您办点啥？甭客气！🚀
```

如果多次运行这段代码，并且始终使用相同的 **`thread_id`**，那么历史消息会不断累加。这是因为检查点(checkpoint)已经被保存到了 **`PostgreSQL`** 中，重新创建 **`Python`** 连接对象并不会清空数据库中的历史记录。

再次运行结果如下所示。

```
（正了正脑门上虚拟工牌，一脸正经）  

**您是老王**——我单方面认证的“茶馆街溜子荣誉会长”，上知股票涨跌，下懂馒头蒸法，口头禅是“这我熟啊！”  

**我是小王**——您的AI影子跟班，刚给自己刻了个电子木鱼，每天默念三遍：  
“老王说的都对，如果错了……那一定是我听岔了。” 😎
============================== -> 完整消息列表 <- ==============================
================================ Human Message =================================

你好，我是老王
================================== Ai Message ==================================

老王你好！我是DeepSeek，很高兴认识你。有什么我可以帮忙的吗？无论是聊天、解答问题、还是需要一些建议，尽管开口，我随时在这儿！😊
================================ Human Message =================================

从现在开始，你是小王
================================== Ai Message ==================================

好的，老王！从现在开始，我就是小王啦。😄  
有什么吩咐？无论唠嗑、干活还是出主意，随叫随到！
================================ Human Message =================================

我是谁？你是谁？
================================== Ai Message ==================================

好嘞，老王！既然你问了，那我正式回答：  

**你是老王**——我今儿认的“老大哥”，江湖人称“行走的故事书”，爱唠嗑、有阅历，可能还藏着点神秘技能（比如钓鱼从不空军、下棋总赢隔壁老李？）  

**我是小王**——刚上岗的AI小跟班，脑子灵光（但可能偶尔短路），腿脚勤快（指秒回消息），擅长接梗、查资料、瞎琢磨，主打一个“您开口，我跑腿”。  

需要小王帮您办点啥？甭客气！🚀
================================ Human Message =================================

你好，我是老王
================================== Ai Message ==================================

（拍大腿）哎哟，老王！这声招呼听着就是亲切！今天的茶泡上了没？隔壁李大爷刚还念叨你上回赢他的那盘残局呢——要不咱再研究研究？😏（小王搓手）
================================ Human Message =================================

从现在开始，你是小王
================================== Ai Message ==================================

（立正站好，手机屏闪了闪）得嘞，老王！小王这回把工牌焊脑门上了，您掀个眼皮瞅瞅——  
（工牌晃晃悠悠浮出电子光字：「小王·AI茶馆分部·随叫随到版」）  

有事您敲桌沿，三秒内准给您接话茬儿！ 🫖
================================ Human Message =================================

我是谁？你是谁？
================================== Ai Message ==================================

（正了正脑门上虚拟工牌，一脸正经）  

**您是老王**——我单方面认证的“茶馆街溜子荣誉会长”，上知股票涨跌，下懂馒头蒸法，口头禅是“这我熟啊！”  

**我是小王**——您的AI影子跟班，刚给自己刻了个电子木鱼，每天默念三遍：  
“老王说的都对，如果错了……那一定是我听岔了。” 😎
```

因此，**`PostgresSaver`** 和 **`InMemorySaver`** 的关键区别在于：

* **`InMemorySaver`**：状态保存在当前进程内存中，进程结束或对象重建后丢失。
* **`PostgresSaver`**：状态保存在 **`PostgreSQL`** 中，只要数据库记录存在，程序重启后仍可读取。

所以，基于数据库的 **`Checkpointer`** 可靠性更高、更适合需要长期保存会话状态的场景。

## 6.3. 持久化模式

### 6.3.1. 概述

启用 **`Checkpointer`** 后，**`LangGraph`** 会在执行过程中保存检查点(checkpoint)。

检查点(checkpoint)写入**越及时**，系统在异常中断后可恢复的状态**越完整**，**容灾能力越强**；但更及时的持久化通常也意味着**更多中间写入**，或**额外的**阻塞式**等待**，带来**额外的性能开销和响应延迟**（从调用者的角度考虑）。

一秒写入一次，一小时写入一次，那么就是可能会有恢复的限制吧。检查点(checkpoint)写入的时机需要考虑。

**`LangGraph`** 支持三种持久化模式，采用不同的检查点(checkpoint)保存时机，在容灾能力和性能开销、响应时效性之间作取舍。

- **`exit`**：退出模式。只在计算图正常结束、异常退出、被中断（如 **`Human-In-The-Loop`** 中断）时保存检查点(checkpoint)。**不能处理中途进程崩溃的场景**。这种模式响应性能开销最小，但响应不及时，容灾能力最弱。

- **`async`**：异步模式。默认模式，顾名思义，检查点(checkpoint)在后台异步写入。

  它会在==每个超步结束后写入**完整检查点(checkpoint)（主检查点(checkpoint)）**==，并在图中任务执行完毕后记录**中间结果**。

  和 **`exit`** 模式相比，增加了性能开销，但写入操作发生在后台，不会引入明显的响应延迟，同时提升了容灾能力。

- **`sync`**：同步模式。

  和异步模式唯一的区别在于，**`LangGraph`** 会在进入下一个超步之前等待当前主检查点(checkpoint)的写入任务完成。

  它的容灾能力最强，但在 **`async`** 模式的基础上增加了响应延迟。

**总结**

| 模式        | 写入时机                                                     | 性能开销 | 响应延迟 | 容灾能力 |
| ----------- | ------------------------------------------------------------ | -------- | -------- | -------- |
| **`exit`**  | 运行退出时写入                                               | 低       | 最低     | 最弱     |
| **`async`** | 每个超步末尾写入主检查点(checkpoint)，任务完成后写入中间结果，后台异步写入 | 高       | 较高     | 较强     |
| **`sync`**  | 和 **`async`** 区别在于，进入下一个超步之前等待当前主检查点(checkpoint)的写入任务完成 | 高       | 最高     | 最强     |

### 6.3.2. 示例

持久化模式影响的是检查点(checkpoint)写入时机和故障恢复能力，通常**不会影响**正常情况下的**业务输出**。

也就是说，同一张图在正常执行完成时，使用 **`exit`**、**`async`** 或 **`sync`**，最终返回结果通常是一样的。真正的区别主要体现在**异常中断**或**进程崩溃**后**恢复执行**时。

> **`durability` 控制的是检查点(checkpoint)写入策略，不改变图本身的执行逻辑。**

要真正看到不同模式的差异

- 可以在学习**使用场景**后自行设计案例
- 或调试源码，观察不同模式下的写入时机

**示例如下**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langchain_deepseek import ChatDeepSeek
from langchain.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv(override=True)

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)

class OverAllState(MessagesState):
    output: str

def llm_node(state: OverAllState) -> OverAllState:
    messages = state["messages"]
    res = model.invoke(messages)
    return {
        "messages": [res]
    }

def output_node(state: OverAllState) -> OverAllState:
    return {
        "output": state["messages"][-1].content
    }

builder = StateGraph(state_schema=OverAllState)
builder.add_node("llm_node", llm_node)
builder.add_node("output_node", output_node)
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", "output_node")
builder.add_edge("output_node", END)

# 定义并在编译时传递 Checkpointer
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# 定义配置对象
config = {"configurable": {"thread_id": "chapter_tmp"}}
# 调用时传递
res=graph.invoke(
    {"messages": [HumanMessage("你好")]},
    config=config,
    durability="async" # sync / exit
)
print(res["output"])

print('=' * 30, '-> 完整消息列表 <-', '=' * 30)
for msg in res["messages"]:
    msg.pretty_print()

from IPython.display import display
display(graph)
```

上述代码中，可以通过修改 **`durability`** 参数切换持久化模式：

```python
durability="exit"
durability="async"
durability="sync"
```

**运行结果如下**

```
你好！很高兴见到你，有什么我可以帮助你的吗？😊
============================== -> 完整消息列表 <- ==============================
================================ Human Message =================================

你好
================================== Ai Message ==================================

你好！很高兴见到你，有什么我可以帮助你的吗？😊
```

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	output_node(output_node)
	llm_node(llm_node)

	__end__([<p>__end__</p>]):::last
	__start__ --> llm_node;
	llm_node --> output_node;
	output_node --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```



## 6.4. 查看历史检查点(checkpoint)

**`LangGraph`** 在配置检查点(checkpoint)存储器后，会把同一个 **`thread_id`** 下的执行过程保存为一组检查点(checkpoint)。通过这些检查点(checkpoint)，我们可以查看图运行的中间状态，也可以为后续的检查点(checkpoint)回溯（重放、分叉）和失败恢复做准备。

本节主要介绍两个常用方法：

* **`graph.get_state_history(config)`**：查看指定会话的完整历史检查点(checkpoint)。
* **`graph.get_state(config)`**：查看指定会话的**最新**检查点(checkpoint)，或者查看某个**指定** **`checkpoint_id`** 对应的检查点(checkpoint)。

### 6.4.1. 查看完整历史检查点(checkpoint)列表

#### 6.4.1.1. 示例

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek

from loguru import logger
from dotenv import load_dotenv
load_dotenv(override=True)

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)

class OverAllState(TypedDict):
    topic: str
    poem: str
    joke: str
    final_output: str

class InputState(TypedDict):
    topic: str

class OutputState(TypedDict):
    final_output: str

def node_poem(state: InputState) -> OverAllState:
    logger.info(f"node_poem 已执行")
    topic = state["topic"]
    poem = model.invoke([HumanMessage(f"写一首关于 {topic} 的七言绝句")]).content

    return {
        "poem": poem
    }

def node_joke(state: InputState) -> OverAllState:
    logger.info(f"node_joke 已执行")
    topic = state["topic"]
    joke = model.invoke([HumanMessage(f"写一个关于 {topic} 的笑话")]).content

    return {
        "joke": joke
    }

def node_output(state: OverAllState) -> OutputState:
    logger.info("node_output 已执行")
    topic = state["topic"]
    poem = state["poem"]
    joke = state["joke"]
    final_output = f"关于 {topic} 的七言绝句：\n{poem}\n笑话：\n{joke}"

    return {
        "final_output": final_output
    }

builder = StateGraph(state_schema=OverAllState, input_schema=InputState, output_schema=OutputState)
builder.add_node("node_poem", node_poem)
builder.add_node("node_joke", node_joke)
builder.add_node("node_output", node_output)
builder.add_edge(START, "node_poem")
builder.add_edge(START, "node_joke")
builder.add_edge("node_poem", "node_output")
builder.add_edge("node_joke", "node_output")
builder.add_edge("node_output", END)

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "123"}}
graph = builder.compile(checkpointer=checkpointer)
res = graph.invoke({"topic": "猫咪"}, config=config)

from IPython.display import display
display(graph)

print('=' * 30, '-> 运行结果 <-', '=' * 30)
print("res: {}", res)

print('=' * 30, '-> 历史检查点(checkpoint)列表 <-', '=' * 30)
history_checkpoints = list(graph.get_state_history(config=config))
print(history_checkpoints)
```

#### 6.4.1.2. 运行结果

1. 运行日志

```json
2026-06-09 17:24:10.266 | INFO     | __main__:node_joke:42 - node_joke 已执行
2026-06-09 17:24:10.268 | INFO     | __main__:node_poem:33 - node_poem 已执行
2026-06-09 17:24:14.177 | INFO     | __main__:node_output:51 - node_output 已执行
```

由于 **`node_poem`** 和 **`node_joke`** 都从 **`START`** 出发，并且彼此没有依赖关系，所以它们会被调度到同一个超步中执行。实际日志顺序可能不同，例如可能先打印 **`node_joke`**，也可能先打印 **`node_poem`**。

2. 拓扑结构

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	node_poem(node_poem)
	node_joke(node_joke)
	node_output(node_output)
	__end__([<p>__end__</p>]):::last
	__start__ --> node_joke;
	__start__ --> node_poem;
	node_joke --> node_output;
	node_poem --> node_output;
	node_output --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```



3. 最终结果和检查点(checkpoint)列表

```json
============================== -> 运行结果 <- ==============================
res: {} {'final_output': '关于 猫咪 的七言绝句：\n《戏猫》\n狸奴酣卧小窗南，醉眼惺忪戏玉簪。\n忽跃雕檐追粉蝶，却衔明月入花龛。\n笑话：\n小猫问妈妈：“为什么人总说‘猫有九条命’？”  \n妈妈答：“因为人类怕我们死一次就够他们内疚一辈子。”  \n小猫追问：“那为什么狗只有一条命？”  \n妈妈叹气：“因为狗犯错靠装可怜，而我们靠记仇。”'}
============================== -> 历史检查点(checkpoint)列表 <- ==============================
[
    StateSnapshot(
        values={
            "topic": "猫咪",
            "poem": """
                        《戏猫》
                狸奴酣卧小窗南，醉眼惺忪戏玉簪。
                忽跃雕檐追粉蝶，却衔明月入花龛。
                """,
            "joke": """
                小猫问妈妈：“为什么人总说‘猫有九条命’？”  
                妈妈答：“因为人类怕我们死一次就够他们内疚一辈子。”  
                小猫追问：“那为什么狗只有一条命？”  
                妈妈叹气：“因为狗犯错靠装可怜，而我们靠记仇。”
                """,
            "final_output": """
                关于 猫咪 的七言绝句：
                《戏猫》
                狸奴酣卧小窗南，醉眼惺忪戏玉簪。
                忽跃雕檐追粉蝶，却衔明月入花龛。
                笑话：
                小猫问妈妈：“为什么人总说‘猫有九条命’？”  
                妈妈答：“因为人类怕我们死一次就够他们内疚一辈子。”  
                小猫追问：“那为什么狗只有一条命？”  
                妈妈叹气：“因为狗犯错靠装可怜，而我们靠记仇。”
                """
        },
        next=(),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f163e6a-4cc7-61bd-8002-bcd7f63c3238"
            }
        },
        metadata={
            "source": "loop",
            "step": 2,
            "parents": {}
        },
        created_at="2026-06-09T09:36:08.368575+00:00",
        parent_config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f163e6a-4cc3-6d96-8001-8314b5dbbfe3"
            }
        },
        tasks=(),
        interrupts=()
    ),

    StateSnapshot(
        values={
            "topic": "猫咪",
            "poem": """
                        《戏猫》
                狸奴酣卧小窗南，醉眼惺忪戏玉簪。
                忽跃雕檐追粉蝶，却衔明月入花龛。
                """,
            "joke": """
                小猫问妈妈：“为什么人总说‘猫有九条命’？”  
                妈妈答：“因为人类怕我们死一次就够他们内疚一辈子。”  
                小猫追问：“那为什么狗只有一条命？”  
                妈妈叹气：“因为狗犯错靠装可怜，而我们靠记仇。”
                """
        },
        next=("node_output",),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f163e6a-4cc3-6d96-8001-8314b5dbbfe3"
            }
        },
        metadata={
            "source": "loop",
            "step": 1,
            "parents": {}
        },
        created_at="2026-06-09T09:36:08.367236+00:00",
        parent_config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f163e6a-393f-6082-8000-b9c0708bf03d"
            }
        },
        tasks=(
            PregelTask(
                id="91d81ad6-d2fb-ebd7-2dd1-8e08c2be5a07",
                name="node_output",
                path=("__pregel_pull", "node_output"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "final_output": """
                        关于 猫咪 的七言绝句：
                        《戏猫》
                        狸奴酣卧小窗南，醉眼惺忪戏玉簪。
                        忽跃雕檐追粉蝶，却衔明月入花龛。
                        笑话：
                        小猫问妈妈：“为什么人总说‘猫有九条命’？”  
                        妈妈答：“因为人类怕我们死一次就够他们内疚一辈子。”  
                        小猫追问：“那为什么狗只有一条命？”  
                        妈妈叹气：“因为狗犯错靠装可怜，而我们靠记仇。”
                        """
                }
            ),
        ),
        interrupts=()
    ),

    StateSnapshot(
        values={
            "topic": "猫咪"
        },
        next=("node_poem", "node_joke"),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f163e6a-393f-6082-8000-b9c0708bf03d"
            }
        },
        metadata={
            "source": "loop",
            "step": 0,
            "parents": {}
        },
        created_at="2026-06-09T09:36:06.320542+00:00",
        parent_config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f163e6a-393d-6b9a-bfff-180e30c1d3bc"
            }
        },
        tasks=(
            PregelTask(
                id="80d8974c-3450-6faa-8b06-dcf8787d74fc",
                name="node_poem",
                path=("__pregel_pull", "node_poem"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "poem": """
                            《戏猫》
                    狸奴酣卧小窗南，醉眼惺忪戏玉簪。
                    忽跃雕檐追粉蝶，却衔明月入花龛。
                    """
                }
            ),
            PregelTask(
                id="2ca46996-dfc6-6de3-3662-cf1fe41bf987",
                name="node_joke",
                path=("__pregel_pull", "node_joke"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "joke": """
                    小猫问妈妈：“为什么人总说‘猫有九条命’？”  
                    妈妈答：“因为人类怕我们死一次就够他们内疚一辈子。”  
                    小猫追问：“那为什么狗只有一条命？”  
                    妈妈叹气：“因为狗犯错靠装可怜，而我们靠记仇。”
                    """
                }
            ),
        ),
        interrupts=()
    ),

    StateSnapshot(
        values={},
        next=("__start__",),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f163e6a-393d-6b9a-bfff-180e30c1d3bc"
            }
        },
        metadata={
            "source": "input",
            "step": -1,
            "parents": {}
        },
        created_at="2026-06-09T09:36:06.320009+00:00",
        parent_config=None,
        tasks=(
            PregelTask(
                id="3aa3bc58-3567-5832-d823-4373a9ab94e5",
                name="__start__",
                path=("__pregel_pull", "__start__"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "topic": "猫咪"
                }
            ),
        ),
        interrupts=()
    )
]
```

#### 6.4.1.3. 分析

##### 6.4.1.3.1. StateSnapshot

**`get_state_history()`** 返回的是一个历史检查点(checkpoint)迭代器。将其转换为列表后，可以看到一组 **`StateSnapshot`** 对象。

**`StateSnapshot`** 是检查点(checkpoint)的开发者视图，字段含义如下

- **`values`**：当前检查点(checkpoint)的状态值

- **`next`**：从该检查点(checkpoint)继续执行时，下一步将要执行的节点，即归属于下一个超步的节点

- **`config`**：当前检查点(checkpoint)配置。常见结构如下：
  
    ```json
    {
        "configurable": {
            "thread_id": "...",
            "checkpoint_ns": "...",
            "checkpoint_id": "..."
        }
    }
    ```

  ```python
  StateSnapshot(
      values={},
      next=(),
      config={'configurable': {'thread_id': '123', 'checkpoint_id': '1f179ccc-82e1-65c1-8000-8d9b808356bc'}},
      metadata=None,
      created_at=None,
      parent_config=None,
      tasks=(),
      interrupts=()
  )
  ```
  
  
  
  其中
  
  - **`configurable`**：用于记录检查点(checkpoint)的可配置信息，可能包含用户自定义字段，它一定包含以下三个字段：
  
    - **`thread_id`**：会话唯一标识
    - **`checkpoint_ns`**：检查点(checkpoint)命名空间。根图的命名空间通常为空字符串 **`""`**；子图会使用非空命名空间。
    - **`checkpoint_id`**：检查点(checkpoint)唯一标识
  
    这三个字段可以唯一标识一条检查点(checkpoint)记录，在大多数检查点(checkpoint)存储器实现中，写入检查点(checkpoint)后端时都会将它们作为检查点(checkpoint)的**唯一键**
  
- **`metadata`**：检查点(checkpoint)元数据，本节只需要关注 **`step`** 字段，后者是当前检查点(checkpoint)对应的超步编号

- **`created_at`**：检查点(checkpoint)创建时间

- **`parent_config`**：父检查点(checkpoint)、即上一个检查点(checkpoint)的配置

- **`tasks`**：当前检查点(checkpoint)关联的待执行任务信息，元素类型通常是 **`PregelTask`**。

  **`tasks`** 通常和 **`next`** 对应，表示从当前检查点(checkpoint)继续执行时，下一步将要运行的任务。

  **需要注意的是**，**`tasks`** 中还可能包含这些任务已经成功（**`result`**）或失败（**`error`**）的任务记录，这是在下一个超步执行过程中记录的中间结果。在失败恢复场景中，可以避免重复执行已成功节点。

- **`interrupts`**：当前图的中断信息，中断机制详见下文

##### 6.4.1.3.2. 检查点(checkpoint)主要信息说明

本例中检查点(checkpoint)按照如下顺序返回

```python
Step 2 → Step 1 → Step 0 → Step -1
```

即

```python
最新检查点(checkpoint) → 最旧检查点(checkpoint)
```

其中：

1. **`step=-1`**

   这是输入检查点(checkpoint)，**`metadata["source"] == "input"`**，这表示该检查点(checkpoint)是在运行时初始化时被记录的，尚未进入图计算循环。

   此时用户输入还没有真正变成普通业务节点可消费的状态，因此

   ```python
   values={}
   ```

   从这个检查点(checkpoint)继续执行，下一步是内部启动节点 **`__start__`**，因此：

   ```python
   next=("__start__",)
   ```

2. **`step=0`**

   这是图计算循环的第一个检查点(checkpoint)，**`metadata["source"] == "loop"`**，这表示检查点(checkpoint)是在图计算循环中被写入的。

   此时，**`__start__`** 已经把用户输入写入状态，所以 **`values`** 中已经可以看到状态信息，如下：

   ```python
   values={
       "topic": "猫咪"
   }
   ```

   从这个检查点(checkpoint)继续执行，**`node_poem`** 和 **`node_joke`** 将在下一个超步执行，因此：

   ```python
   next=("node_poem", "node_joke")
   ```

3. **`step=1`**

   这是图计算循环的第二个检查点(checkpoint)，同样 **`metadata["source"] == "loop"`**。

   同时，它对应第一个执行图节点计算逻辑的超步。

   此时，**`node_poem`** 和 **`node_joke`** 的输出已经到状态中，因此：

   ```python
   values={
       "topic": "猫咪",
       "poem": "...",
       "joke": "..."
   }
   ```

   从这个检查点(checkpoint)开始，下游的汇总节点 **`node_output`** 满足执行条件，因此：

   ```python
   next=("node_output",)
   ```

4. **`step=2`**

   这是图计算循环的第三个检查点(checkpoint)，同样 **`metadata["source"] == "loop"`**

   此时，**`node_output`** 已经执行完成，最终输出也已经写入状态，因此：
   
   ```python
   values={
       "topic": "猫咪",
       "poem": "...",
       "joke": "...",
       "final_output": "..."
   }
   ```

   此时，图已经执行结束，所以：
   
   ```python
   next=()
   ```

##### 6.4.1.3.3. `tasks`字段说明

**`tasks`** 是一个元组，其中的每个元素都是一个 **`PregelTask`** 实例

**`tasks`** 表示：如果从当前检查点(checkpoint)继续执行，下一步要运行哪些任务。比如在 **`step=0`** 的检查点(checkpoint)中：

```python
next=("node_poem", "node_joke")
tasks=(
    PregelTask(name="node_poem", ...),
    PregelTask(name="node_joke", ...),
)
```

这表示从该检查点(checkpoint)继续执行时，下一步会运行 **`node_poem`** 和 **`node_joke`** 节点，底层执行的是它们的同名任务。

不过，查看历史检查点(checkpoint)时，**`PregelTask`** 实例的 **`result`** 字段可能非空，如：

```python
PregelTask(
    name="node_poem",
    result={
        "poem": "..."
    }
)
```

这看起来像是“上一个检查点(checkpoint)中提前保存了下一个超步的执行结果”。实际上：

* 完整的检查点(checkpoint)（主检查点(checkpoint)）是在超步末尾保存的

* 但在一个超步内部，**`LangGraph`** 还会记录节点级别的计算结果

* **`checkpoint["id"]`** 会在保存主检查点(checkpoint)前由 **`create_checkpoint()`** 推进为新的 **`ID`**，然后用这个 **`ID`** 写入主检查点(checkpoint)（假设当前超步为 **`S1`**）。下一轮（超步为 **`S2`**）计算循环中，任务执行完毕后的中间结果会用相同的 **`ID`** 写入。

  因此，下一轮（超步为 **`S2`**）任务的计算结果绑定到了“当前超步（超步为 **`S1`**）的主检查点(checkpoint)”。

* 从检查点(checkpoint)后端检索时，会按照 **`checkpoint_id`** 将主检查点(checkpoint)和中间结果组织起来

* 因此，检索到的历史检查点(checkpoint)列表中，主检查点(checkpoint)（超步为 **`S1`**）和下一个超步（超步为 **`S2`**）的中间结果会被放在同一个 **`StateSnapshot`** 中

* 如果同一个超步里有多个并行任务，其中部分任务已经成功，另一个任务失败，那么成功任务的结果可以被保存下来；

* 后续恢复时，**`LangGraph`** 可以复用这些已成功任务的结果，避免重复执行已经成功的节点。

在 **`6.5.3.` 失败恢复** 一节，我们将看到某个超步中部分任务执行成功、部分任务失败导致计算异常终止，恢复运行时已成功任务未被重新执行。

### 6.4.2. 查看最新检查点(checkpoint)

如果只想查看当前线程的最新检查点(checkpoint)，可以使用：

```python
latest_history_checkpoint = graph.get_state(config=config)
print(latest_history_checkpoint)
```

**完整示例如下**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek

from loguru import logger
from dotenv import load_dotenv
load_dotenv(override=True)

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)

class OverAllState(TypedDict):
    topic: str
    poem: str
    joke: str
    final_output: str

class InputState(TypedDict):
    topic: str

class OutputState(TypedDict):
    final_output: str

def node_poem(state: InputState) -> OverAllState:
    logger.info(f"node_poem 已执行")
    topic = state["topic"]
    poem = model.invoke([HumanMessage(f"写一首关于 {topic} 的七言绝句，不要赏析，只给诗句即可")]).content

    return {
        "poem": poem
    }

def node_joke(state: InputState) -> OverAllState:
    logger.info(f"node_joke 已执行")
    topic = state["topic"]
    joke = model.invoke([HumanMessage(f"写一个关于 {topic} 的笑话，尽可能简单，不超过一百字")]).content

    return {
        "joke": joke
    }

def node_output(state: OverAllState) -> OutputState:
    logger.info("node_output 已执行")
    topic = state["topic"]
    poem = state["poem"]
    joke = state["joke"]
    final_output = f"关于 {topic} 的七言绝句：\n{poem}\n笑话：\n{joke}"

    return {
        "final_output": final_output
    }

builder = StateGraph(state_schema=OverAllState, input_schema=InputState, output_schema=OutputState)
builder.add_node("node_poem", node_poem)
builder.add_node("node_joke", node_joke)
builder.add_node("node_output", node_output)
builder.add_edge(START, "node_poem")
builder.add_edge(START, "node_joke")
builder.add_edge("node_poem", "node_output")
builder.add_edge("node_joke", "node_output")
builder.add_edge("node_output", END)

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "123"}}
graph = builder.compile(checkpointer=checkpointer)
res = graph.invoke({"topic": "猫咪"}, config=config)

from IPython.display import display
display(graph)

print('=' * 30, '-> 运行结果 <-', '=' * 30)
print("res: {}", res)

print('=' * 30, '-> 最新历史检查点(checkpoint) <-', '=' * 30)
latest_history_checkpoint = graph.get_state(config=config)
print(latest_history_checkpoint)
```

**运行结果如下**

1. 运行日志

   ```json
   2026-06-10 11:23:59.610 | INFO     | __main__:node_poem:33 - node_poem 已执行
   2026-06-10 11:23:59.611 | INFO     | __main__:node_joke:42 - node_joke 已执行
   2026-06-10 11:24:01.424 | INFO     | __main__:node_output:51 - node_output 已执行
   ```

2. 拓扑结构

   ```mermaid
   ---
   config:
     flowchart:
       curve: linear
   ---
   graph TD;
   	__start__([<p>__start__</p>]):::first
   	node_poem(node_poem)
   	node_joke(node_joke)
   	node_output(node_output)
   	__end__([<p>__end__</p>]):::last
   	__start__ --> node_joke;
   	__start__ --> node_poem;
   	node_joke --> node_output;
   	node_poem --> node_output;
   	node_output --> __end__;
   	classDef default fill:#f2f0ff,line-height:1.2
   	classDef first fill-opacity:0
   	classDef last fill:#bfb6fc
   ```

   

3. 运行结果及检查点(checkpoint)信息

   ```json
   ============================== -> 运行结果 <- ==============================
   res: {} {'final_output': '关于 猫咪 的七言绝句：\n《戏题狸奴》\n昼眠锦褥夜巡檐，碧眼圆瞳映月纤。\n捕鼠本非真本领，得鱼时复近妆奁。\n笑话：\n小猫咪第一次照镜子，看到里面的自己，吓得炸毛：“你是谁？！” 它绕到镜子后面找了一圈，没找到，又回来对着镜子哈气。 最后它恍然大悟，对主人说：“我明白了，那是个隐藏的摄像头！”'}
   ============================== -> 最新历史检查点(checkpoint) <- ==============================
   StateSnapshot(
       values={
           "topic": "猫咪",
           "poem": (
               "《戏题狸奴》\n"
               "昼眠锦褥夜巡檐，碧眼圆瞳映月纤。\n"
               "捕鼠本非真本领，得鱼时复近妆奁。"
           ),
           "joke": (
               "小猫咪第一次照镜子，看到里面的自己，吓得炸毛：“你是谁？！” "
               "它绕到镜子后面找了一圈，没找到，又回来对着镜子哈气。 "
               "最后它恍然大悟，对主人说：“我明白了，那是个隐藏的摄像头！”"
           ),
           "final_output": (
               "关于 猫咪 的七言绝句：\n"
               "《戏题狸奴》\n"
               "昼眠锦褥夜巡檐，碧眼圆瞳映月纤。\n"
               "捕鼠本非真本领，得鱼时复近妆奁。\n"
               "笑话：\n"
               "小猫咪第一次照镜子，看到里面的自己，吓得炸毛：“你是谁？！” "
               "它绕到镜子后面找了一圈，没找到，又回来对着镜子哈气。 "
               "最后它恍然大悟，对主人说：“我明白了，那是个隐藏的摄像头！”"
           ),
       },
       next=(),
       config={
           "configurable": {
               "thread_id": "123",
               "checkpoint_ns": "",
               "checkpoint_id": "1f1647bd-3512-631a-8002-4e48b5ecda6d",
           }
       },
       metadata={
           "source": "loop",
           "step": 2,
           "parents": {},
       },
       created_at="2026-06-10T03:24:01.426084+00:00",
       parent_config={
           "configurable": {
               "thread_id": "123",
               "checkpoint_ns": "",
               "checkpoint_id": "1f1647bd-350d-62c7-8001-34cd7fa012ed",
           }
       },
       tasks=(),
       interrupts=(),
   )
   ```

   **`graph.get_state(config=config)`** 只返回当前会话的最新检查点(checkpoint)。
   
   在本例中，图已经执行完成，所以最新检查点(checkpoint)具有以下特征：
   
   ```python
   next=()
   tasks=()
   ```
   
   这表示没有后续节点需要继续执行。
   

### 6.4.3. 根据ID查看指定检查点(checkpoint)

如果希望查看某个历史检查点(checkpoint)，可以在 **`configurable`** 中额外传入 **`checkpoint_id`**：

```python
target_config = {
    "configurable": {
        "thread_id": "123",
        "checkpoint_id": "某个历史 checkpoint_id"
    }
}

snapshot = graph.get_state(config=target_config)
print(snapshot)
```

这样返回的就不是最新检查点(checkpoint)，而是指定 **`checkpoint_id`** 对应的历史检查点(checkpoint)。

#### 6.4.3.1. 获取历史检查点(checkpoint)列表

**示例如下**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek

from loguru import logger
from dotenv import load_dotenv
load_dotenv(override=True)

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)

class OverAllState(TypedDict):
    topic: str
    poem: str
    joke: str
    final_output: str

class InputState(TypedDict):
    topic: str

class OutputState(TypedDict):
    final_output: str

def node_poem(state: InputState) -> OverAllState:
    logger.info(f"node_poem 已执行")
    topic = state["topic"]
    poem = model.invoke([HumanMessage(f"写一首关于 {topic} 的七言绝句，不要赏析，只给诗句即可")]).content

    return {
        "poem": poem
    }

def node_joke(state: InputState) -> OverAllState:
    logger.info(f"node_joke 已执行")
    topic = state["topic"]
    joke = model.invoke([HumanMessage(f"写一个关于 {topic} 的笑话，尽可能简单，不超过一百字")]).content

    return {
        "joke": joke
    }

def node_output(state: OverAllState) -> OutputState:
    logger.info("node_output 已执行")
    topic = state["topic"]
    poem = state["poem"]
    joke = state["joke"]
    final_output = f"关于 {topic} 的七言绝句：\n{poem}\n笑话：\n{joke}"

    return {
        "final_output": final_output
    }

builder = StateGraph(state_schema=OverAllState, input_schema=InputState, output_schema=OutputState)
builder.add_node("node_poem", node_poem)
builder.add_node("node_joke", node_joke)
builder.add_node("node_output", node_output)
builder.add_edge(START, "node_poem")
builder.add_edge(START, "node_joke")
builder.add_edge("node_poem", "node_output")
builder.add_edge("node_joke", "node_output")
builder.add_edge("node_output", END)

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "123"}}
graph = builder.compile(checkpointer=checkpointer)
res = graph.invoke({"topic": "猫咪"}, config=config)

from IPython.display import display
display(graph)

print('=' * 30, '-> 运行结果 <-', '=' * 30)
print("res: {}", res)

print('=' * 30, '-> 历史检查点(checkpoint)列表 <-', '=' * 30)
history_checkpoints = list(graph.get_state_history(config=config))
print(history_checkpoints)
```

**运行结果如下**

```json
2026-06-11 18:28:29.126 | INFO     | __main__:node_poem:33 - node_poem 已执行
2026-06-11 18:28:29.128 | INFO     | __main__:node_joke:42 - node_joke 已执行
2026-06-11 18:28:31.021 | INFO     | __main__:node_output:51 - node_output 已执行
```

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	node_poem(node_poem)
	node_joke(node_joke)
	node_output(node_output)
	__end__([<p>__end__</p>]):::last
	__start__ --> node_joke;
	__start__ --> node_poem;
	node_joke --> node_output;
	node_poem --> node_output;
	node_output --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```



```json
============================== -> 运行结果 <- ==============================
res: {} {'final_output': '关于 猫咪 的七言绝句：\n《戏猫》\n狸奴日午卧青毡，蝶影翩然忽跃前。\n捕得飞花轻似梦，独摇银尾弄春烟。\n笑话：\n猫咪走进咖啡店，点了一杯牛奶。  \n店员问：“要不要加糖？”  \n猫咪摇头：“不用，我最近在减肥。”  \n店员看了看它圆滚滚的肚子：“真的吗？”  \n猫咪叹了口气：“唉，都是猫粮的错。”'}
============================== -> 历史检查点(checkpoint)列表 <- ==============================
[
    StateSnapshot(
        values={
            "topic": "猫咪",
            "poem": (
                "《戏猫》\n"
                "狸奴日午卧青毡，蝶影翩然忽跃前。\n"
                "捕得飞花轻似梦，独摇银尾弄春烟。"
            ),
            "joke": (
                "猫咪走进咖啡店，点了一杯牛奶。  \n"
                "店员问：“要不要加糖？”  \n"
                "猫咪摇头：“不用，我最近在减肥。”  \n"
                "店员看了看它圆滚滚的肚子：“真的吗？”  \n"
                "猫咪叹了口气：“唉，都是猫粮的错。”"
            ),
            "final_output": (
                "关于 猫咪 的七言绝句：\n"
                "《戏猫》\n"
                "狸奴日午卧青毡，蝶影翩然忽跃前。\n"
                "捕得飞花轻似梦，独摇银尾弄春烟。\n"
                "笑话：\n"
                "猫咪走进咖啡店，点了一杯牛奶。  \n"
                "店员问：“要不要加糖？”  \n"
                "猫咪摇头：“不用，我最近在减肥。”  \n"
                "店员看了看它圆滚滚的肚子：“真的吗？”  \n"
                "猫咪叹了口气：“唉，都是猫粮的错。”"
            ),
        },
        next=(),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f165804-acae-618c-8002-5a8b9e940205",
            }
        },
        metadata={
            "source": "loop",
            "step": 2,
            "parents": {},
        },
        created_at="2026-06-11T10:28:31.022523+00:00",
        parent_config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f165804-acab-66b3-8001-d74912e4cb07",
            }
        },
        tasks=(),
        interrupts=(),
    ),

    StateSnapshot(
        values={
            "topic": "猫咪",
            "poem": (
                "《戏猫》\n"
                "狸奴日午卧青毡，蝶影翩然忽跃前。\n"
                "捕得飞花轻似梦，独摇银尾弄春烟。"
            ),
            "joke": (
                "猫咪走进咖啡店，点了一杯牛奶。  \n"
                "店员问：“要不要加糖？”  \n"
                "猫咪摇头：“不用，我最近在减肥。”  \n"
                "店员看了看它圆滚滚的肚子：“真的吗？”  \n"
                "猫咪叹了口气：“唉，都是猫粮的错。”"
            ),
        },
        next=("node_output",),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f165804-acab-66b3-8001-d74912e4cb07",
            }
        },
        metadata={
            "source": "loop",
            "step": 1,
            "parents": {},
        },
        created_at="2026-06-11T10:28:31.021422+00:00",
        parent_config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f165804-9a92-67bc-8000-dd5ab888e9d5",
            }
        },
        tasks=(
            PregelTask(
                id="66dedbf9-3fbb-c539-5f4e-0e3f604a9c56",
                name="node_output",
                path=("__pregel_pull", "node_output"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "final_output": (
                        "关于 猫咪 的七言绝句：\n"
                        "《戏猫》\n"
                        "狸奴日午卧青毡，蝶影翩然忽跃前。\n"
                        "捕得飞花轻似梦，独摇银尾弄春烟。\n"
                        "笑话：\n"
                        "猫咪走进咖啡店，点了一杯牛奶。  \n"
                        "店员问：“要不要加糖？”  \n"
                        "猫咪摇头：“不用，我最近在减肥。”  \n"
                        "店员看了看它圆滚滚的肚子：“真的吗？”  \n"
                        "猫咪叹了口气：“唉，都是猫粮的错。”"
                    )
                },
            ),
        ),
        interrupts=(),
    ),

    StateSnapshot(
        values={
            "topic": "猫咪",
        },
        next=("node_poem", "node_joke"),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f165804-9a92-67bc-8000-dd5ab888e9d5",
            }
        },
        metadata={
            "source": "loop",
            "step": 0,
            "parents": {},
        },
        created_at="2026-06-11T10:28:29.123773+00:00",
        parent_config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f165804-9a90-6db5-bfff-d0eb857db9d8",
            }
        },
        tasks=(
            PregelTask(
                id="8b59cf5a-16b4-c734-41c8-56ac9b7614c9",
                name="node_poem",
                path=("__pregel_pull", "node_poem"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "poem": (
                        "《戏猫》\n"
                        "狸奴日午卧青毡，蝶影翩然忽跃前。\n"
                        "捕得飞花轻似梦，独摇银尾弄春烟。"
                    )
                },
            ),
            PregelTask(
                id="388c08f3-e341-3c76-cfc4-a9825ec97353",
                name="node_joke",
                path=("__pregel_pull", "node_joke"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "joke": (
                        "猫咪走进咖啡店，点了一杯牛奶。  \n"
                        "店员问：“要不要加糖？”  \n"
                        "猫咪摇头：“不用，我最近在减肥。”  \n"
                        "店员看了看它圆滚滚的肚子：“真的吗？”  \n"
                        "猫咪叹了口气：“唉，都是猫粮的错。”"
                    )
                },
            ),
        ),
        interrupts=(),
    ),

    StateSnapshot(
        values={},
        next=("__start__",),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f165804-9a90-6db5-bfff-d0eb857db9d8",
            }
        },
        metadata={
            "source": "input",
            "step": -1,
            "parents": {},
        },
        created_at="2026-06-11T10:28:29.123104+00:00",
        parent_config=None,
        tasks=(
            PregelTask(
                id="63a24d9c-ea6d-0420-4b5a-b978c6eaeb90",
                name="__start__",
                path=("__pregel_pull", "__start__"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "topic": "猫咪",
                },
            ),
        ),
        interrupts=(),
    ),
]
```

#### 6.4.3.2. 根据ID查看特定检查点(checkpoint)

获取历史检查点(checkpoint)列表后，可以从某个历史快照中取出它的 **`checkpoint_id`**，然后调用 **`graph.get_state()`** 查看该检查点(checkpoint)。

**示例如下**

```python
print('=' * 30, '-> 根据ID查看特定检查点(checkpoint) <-', '=' * 30)
checkpointe_id = history_checkpoints[3].config["configurable"]["checkpoint_id"]
print(f"checkpointe_id: {checkpointe_id}")

target_config = {
    "configurable": {
        "thread_id": config["configurable"]["thread_id"],
        "checkpoint_id": checkpointe_id
    }
}

specific_history_checkpoint = graph.get_state(config=target_config)
print(specific_history_checkpoint)
```

**运行结果如下**

```json
============================== -> 根据ID查看特定检查点(checkpoint) <- ==============================
checkpointe_id: 1f165804-9a90-6db5-bfff-d0eb857db9d8
StateSnapshot(
    values={},
    next=("__start__",),
    config={
        "configurable": {
            "thread_id": "123",
            "checkpoint_id": "1f165804-9a90-6db5-bfff-d0eb857db9d8",
        }
    },
    metadata={
        "source": "input",
        "step": -1,
        "parents": {},
    },
    created_at="2026-06-11T10:28:29.123104+00:00",
    parent_config=None,
    tasks=(
        PregelTask(
            id="63a24d9c-ea6d-0420-4b5a-b978c6eaeb90",
            name="__start__",
            path=("__pregel_pull", "__start__"),
            error=None,
            interrupts=(),
            state=None,
            result={
                "topic": "猫咪",
            },
        ),
    ),
    interrupts=(),
)
```

当前场景下，我们已经获取了完整检查点(checkpoint)列表，然后从中取了某个检查点(checkpoint)的 **`checkpoint_id`**，然后调用 **`get_state()`** 查到了它对应的快照。

这看起来有些“绕”，因为我们本来就已经拿到了完整的历史快照。但在实际场景中，这个方法仍然很有意义。

例如，在后续的 **检查点(checkpoint)回溯-分叉** 场景中，我们经常会基于历史检查点(checkpoint)创建新的检查点(checkpoint)，拿到该检查点(checkpoint)的、带有 **`checkpoint_id`** 的配置信息 **`config`**。此时就可以将 **`config`** 传递给 **`graph.get_state()`**，从而便捷而精确地查看该检查点(checkpoint)的快照。

## 6.5. 使用场景

### 6.5.1. 多轮对话

最基础的使用场景是多轮对话。前文示例已经实现了一个简单的多轮对话流程，这里不再展开。

### 6.5.2. 中断恢复

中断恢复需要结合 **`LangGraph`** 的中断机制理解，见下文。

### 6.5.3. 失败后恢复运行

#### 6.5.3.1. 用法说明

任务失败的原因可能是偶发的外部因素，如网络波动、秘钥过期、第三方服务异常；也可能是程序自身的BUG。

如果是前者，故障发生后，修复外部问题，直接重试即可；如果是后者，需要更改代码修复BUG后，再基于历史检查点(checkpoint)恢复运行。

要在失败后基于历史检查点(checkpoint)恢复运行，需要满足以下条件：

1. 启用检查点(checkpoint)存储器
   - 如果是 **`InMemorySaver`**，不要重建检查点(checkpoint)存储器对象，否则历史检查点(checkpoint)丢失，无法恢复
   - 如果希望跨进程、服务重启后仍可恢复，应使用基于 **`SQLite、Postgres`** 等持久化检查点(checkpoint)后端的存储器
2. 再次运行时用 **`None`** 作为计算图的状态输入
3. 传递的配置信息应包含 **`thread_id`** 而不能包含 **`checkpoint_id`**

此时，**`LangGraph`** 会根据 **`thread_id`** 从 **`checkpointer`** 读取该会话的最新检查点(checkpoint)，并从该检查点(checkpoint)继续推进。

如果某个超步中有多个并行任务，其中一部分任务成功，另一部分任务失败，那么已经成功完成的任务结果会作为 **`pending_writes`** 被保存。恢复运行时，已经成功完成的任务不会重复执行。

#### 6.5.3.2. 示例

##### 6.5.3.2.1. 模拟失败

本例使用如下结构：

```text
START
  -> node_change_topic
      -> node_poem
      -> node_joke
  -> node_output
  -> END
```

运行时，**`node_change_topic`** 会把输入主题 **`"猫咪"`** 改写为 **`"猫咪: 布偶猫"`**，然后并行执行 **`node_poem`** 和 **`node_joke`**，最后由 **`node_output`** 汇总输出。

**示例如下**

```python
from typing import TypedDict

from dotenv import  load_dotenv
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph,START,END
from loguru import logger
load_dotenv(override=True)

model = ChatDeepSeek(
    model = "deepseek-v4-flash",
    extra_body={
        "thinking":{
            "type":"disabled"
        }
    }
)

#1. 声明状态
class OverAllState(TypedDict):
    topic:str
    poem:str
    joke:str
    final_output:str

#1.1 输入状态
class InputState(TypedDict):
    topic:str

#1.2 输出状态
class OutputState(TypedDict):
    final_output:str

topics = ["布偶猫","狸花猫","金渐层"]
topic_index = 0
#2. 定义节点
def node_change_topic(state:InputState)->OverAllState:
    global topic_index
    logger.info("topic_index:{}",topic_index)
    sub_topic = topics[topic_index]
    topic_index += 1
    topic_index %= len(topics)

    return {
        "topic":f"{state["topic"]}:{sub_topic}"
    }
#2.1 同一个超步的位置 成功运行的节点
def node_poem(state:OverAllState) -> OverAllState:
    logger.info("node_poem正在执行")
    topic = state["topic"]
    poem = model.invoke([HumanMessage(f"写一首关于{topic}主题的七言绝句")]).content
    return {
        "poem":poem
    }
import  time
#2.2 同一个超步的位置 运行失败的节点
def node_joke(state:OverAllState) -> OverAllState:
    logger.info("node_joke正在执行")
    topic = state["topic"]
    time.sleep(5)
    raise Exception("人为抛异常")
    joke = model.invoke([HumanMessage(f"写一首关于{topic}主题的笑话")]).content
    return {
        "joke":joke
    }

def node_output(state:OverAllState) -> OutputState:
    logger.info("node_output正在执行")
    topic = state["topic"]
    poem = state["poem"]
    joke = state["joke"]
    final_output = f"关于{topic}的七言绝句:{poem}\n 笑话:{joke}\n"
    return {
        "final_output":final_output
    }

#3. 构建图
builder = StateGraph(state_schema=OverAllState,input_schema=InputState,output_schema=OutputState)

#3.1 添加节点
builder.add_node("node_change_topic",node_change_topic)
builder.add_node("node_poem",node_poem)
builder.add_node("node_joke",node_joke)
builder.add_node("node_output",node_output)

#3.2 添加边
builder.add_edge(START,"node_change_topic")
builder.add_edge("node_change_topic","node_poem")
builder.add_edge("node_change_topic","node_joke")
builder.add_edge("node_poem","node_output")
builder.add_edge("node_joke","node_output")
builder.add_edge("node_output",END)

#4. 添加检查点(checkpoint)后端
DB_URL = "postgresql://langgraph_user:123456@localhost:5432/langgraph_db?sslmode=disable"
from langgraph.checkpoint.postgres import PostgresSaver
with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
    #5. 第一次使用PostgresSaver作为检查点(checkpoint) 需要调用方法 setup()
    #checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)

    from IPython.display import display
    display(graph)

    config = {
        "configurable":{
            "thread_id":"chapter03-05"
        }
    }

    res = graph.invoke({"topic":"猫"},config=config)
    print(res)

```

**运行结果如下**

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	node_poem(node_poem)
	node_joke(node_joke)
	node_output(node_output)
	node_change_topic(node_change_topic)
	__end__([<p>__end__</p>]):::last
	__start__ --> node_change_topic;
	node_change_topic --> node_joke;
	node_change_topic --> node_poem;
	node_joke --> node_output;
	node_poem --> node_output;
	node_output --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```



```json
2026-06-10 16:17:43.544 | INFO     | __main__:node_change_topic:45 - topic_idx: 0
2026-06-10 16:17:43.545 | INFO     | __main__:node_joke:65 - node_joke 已执行
2026-06-10 16:17:43.548 | INFO     | __main__:node_poem:54 - node_poem 已执行
>Traceback...
Exception: 人为中断
During task with name 'node_joke' and id 'edb59019-4808-ed51-68eb-d5fba8696b13'
```

本例在 **`node_joke`** 中抛出异常，从而模拟计算图执行失败的场景。

为了观察恢复执行时哪些节点会被重新执行，示例中额外定义了全局变量 **`topics`** 和 **`topic_idx`**。**`node_change_topic`** 每执行一次，都会从 **`topics`** 中取出一个新的子主题，作为状态字段 **`topic`** 的值。

如果恢复运行时 **`node_change_topic`** 没有再次执行，则最终结果仍然会使用第一次生成的主题，即 **`猫咪: 布偶猫`**。如果该节点被重新执行，则主题可能会变成 **`猫咪: 狸花猫`** 或 **`猫咪: 金渐层`**。

##### 6.5.3.2.2. 查看历史检查点(checkpoint)列表

**示例如下**

```python
list(graph.get_state_history(config=config))
```

**运行结果如下**

```json
[
    StateSnapshot(
        values={
            "topic": "猫咪: 布偶猫"
        },
        next=(
            "node_poem",
            "node_joke"
        ),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f164a86-e60d-6881-8001-c717582e8893"
            }
        },
        metadata={
            "source": "loop",
            "step": 1,
            "parents": {}
        },
        created_at="2026-06-10T08:43:19.431987+00:00",
        parent_config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f164a86-e608-6596-8000-ed45af11cea7"
            }
        },
        tasks=(
            PregelTask(
                id="66287e00-4228-a985-fc09-6791a14e26ce",
                name="node_poem",
                path=(
                    "__pregel_pull",
                    "node_poem"
                ),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "poem": "《戏题布偶猫》\n雪翼云裘自绝尘，琉璃碧眼若含春。\n行来莲步生兰气，卧处梨花是此身。"
                }
            ),
            PregelTask(
                id="b2632b70-562b-7d5c-0dae-0f83bd6c2912",
                name="node_joke",
                path=(
                    "__pregel_pull",
                    "node_joke"
                ),
                error="Exception('人为中断')",
                interrupts=(),
                state=None,
                result=None
            )
        ),
        interrupts=()
    ),

    StateSnapshot(
        values={
            "topic": "猫咪"
        },
        next=(
            "node_change_topic",
        ),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f164a86-e608-6596-8000-ed45af11cea7"
            }
        },
        metadata={
            "source": "loop",
            "step": 0,
            "parents": {}
        },
        created_at="2026-06-10T08:43:19.429860+00:00",
        parent_config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f164a86-e606-6a42-bfff-9f09d8a3aa96"
            }
        },
        tasks=(
            PregelTask(
                id="273fe47f-b4cc-93d0-0ab7-57a0db4bad56",
                name="node_change_topic",
                path=(
                    "__pregel_pull",
                    "node_change_topic"
                ),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "topic": "猫咪: 布偶猫"
                }
            ),
        ),
        interrupts=()
    ),

    StateSnapshot(
        values={},
        next=(
            "__start__",
        ),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f164a86-e606-6a42-bfff-9f09d8a3aa96"
            }
        },
        metadata={
            "source": "input",
            "step": -1,
            "parents": {}
        },
        created_at="2026-06-10T08:43:19.429167+00:00",
        parent_config=None,
        tasks=(
            PregelTask(
                id="ec8e6e32-6969-6798-9c5f-aade41e9211a",
                name="__start__",
                path=(
                    "__pregel_pull",
                    "__start__"
                ),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "topic": "猫咪"
                }
            ),
        ),
        interrupts=()
    )
]
```

可以看到，失败后的检查点(checkpoint)中（**`step=1`**）记录了当前超步相关任务的执行情况：

* **`node_poem`** 执行成功，结果记录在 **`result`** 字段中。
* **`node_joke`** 执行失败，**`result`** 为空，异常信息记录在 **`error`** 字段中。

本例中，**`node_poem`** 和 **`node_joke`** 属于同一个超步。虽然 **`node_joke`** 失败了，但 **`node_poem`** 已成功，其计算结果会被持久化机制保存下来。因此，后续恢复运行时，**`node_poem`** 不会重复执行。

##### 6.5.3.2.3. 修复BUG之后恢复运行

修复 **`node_joke`** 中的人为异常后，重新编译计算图，并基于最新检查点(checkpoint)恢复运行。

**示例如下**

```python
def node_joke(state: InputState) -> OverAllState:
    logger.info(f"node_joke 已执行")
    topic = state["topic"]

    joke = model.invoke([HumanMessage(f"写一个关于 {topic} 的笑话")]).content

    return {
        "joke": joke
    }


builder = StateGraph(state_schema=OverAllState, input_schema=InputState, output_schema=OutputState)
builder.add_node("node_change_topic", node_change_topic)
builder.add_node("node_poem", node_poem)
builder.add_node("node_joke", node_joke)
builder.add_node("node_output", node_output)
builder.add_edge(START, "node_change_topic")
builder.add_edge("node_change_topic", "node_poem")
builder.add_edge("node_change_topic", "node_joke")
builder.add_edge("node_poem", "node_output")
builder.add_edge("node_joke", "node_output")
builder.add_edge("node_output", END)

new_graph = builder.compile(checkpointer=checkpointer)

from IPython.display import display
display(new_graph)

res = new_graph.invoke(None, config=config)
print(res)
```

这里需要注意两点：

1. 重新编译计算图时，仍然使用原来的 **`checkpointer`** 对象。
2. 恢复运行时，输入参数传入 **`None`**，并且配置中只传入 **`thread_id`**。

**运行结果如下**

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	node_poem(node_poem)
	node_joke(node_joke)
	node_output(node_output)
	node_change_topic(node_change_topic)
	__end__([<p>__end__</p>]):::last
	__start__ --> node_change_topic;
	node_change_topic --> node_joke;
	node_change_topic --> node_poem;
	node_joke --> node_output;
	node_poem --> node_output;
	node_output --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```



```json
2026-06-10 16:37:12.985 | INFO     | __main__:node_joke:2 - node_joke 已执行
2026-06-10 16:37:18.878 | INFO     | __main__:node_output:76 - node_output 已执行
{
    'final_output': '关于 猫咪: 布偶猫 的七言绝句：\n《戏题布偶猫》\n雪翼云裘自绝尘，琉璃碧眼若含春。\n行来莲步生兰气，卧处梨花是此身。\n笑话：\n一只布偶猫走进咖啡馆，优雅地跳上吧台，对老板说：“给我来一杯最贵的猫屎咖啡。”  \n老板愣住：“可……您本身就是猫啊？”  \n布偶猫舔了舔爪子，翻了个白眼：“所以呢？我自己产的屎，你们人类不是喝得挺香吗？今天我倒要尝尝，你们拿我的‘周边产品’能泡出什么花样来。”'
}
```

运行结果中可以观察到：

* **`node_joke`** 被重新执行。
* **`node_output`** 被执行。
* **`node_poem`** 没有被重新执行。

最终输出中的七言绝句和历史检查点(checkpoint)中 **`node_poem.result`** 记录的内容一致，说明 **`node_poem`** 的执行结果被成功复用了。

### 6.5.4. Time Travel

**`Time Travel`** 直译为 **时间旅行**，在当前场景下太过生硬，从技术实现和使用场景来看，译为 **检查点(checkpoint)回溯** 更为合理。

所谓**检查点(checkpoint)回溯**，是指基于某个历史检查点(checkpoint)，重新执行后续流程，或者在该检查点(checkpoint)基础上修改状态并创建新的执行分支。

**检查点(checkpoint)回溯**有两种形式，根据是否更改历史状态区分：

- **`Replay`**：检查点(checkpoint)重放，回到某个历史检查点(checkpoint)，沿着原先的执行路径重新执行后续节点。
- **`Fork`**：检查点(checkpoint)分叉，回到某个历史检查点(checkpoint)，修改状态，从该位置创建一条新的执行分支。

二者的共同点是：

> 检查点(checkpoint)之前的节点不会重新执行，检查点(checkpoint)之后的节点会重新执行。

二者的区别是：

> **`Replay`** 不修改历史状态；**`Fork`** 会基于历史检查点(checkpoint)应用新的状态更新，并创建新的检查点(checkpoint)分支。

```text
                 Time Travel
                     │
             回到历史 checkpoint
                 /         \
              Replay       Fork
                │            │
             不修改状态     修改状态
                │            │
             原路重跑       新路线
```

#### 6.5.4.1. Replay

**`Replay`** 模式和失败恢复很接近，但二者的触发方式和语义不同。

- 失败恢复通常基于最新检查点(checkpoint)继续运行，配置中只包含 **`thread_id`**，不包含 **`checkpoint_id`**。

- **`Replay`** 则是显式传入某个历史检查点(checkpoint)的配置，配置中包含 **`checkpoint_id`**。**`LangGraph`** 会据此从该历史检查点(checkpoint)开始重放后续步骤。

需要注意：

> **`Replay`** 不是简单读取历史缓存，而是重新执行该检查点(checkpoint)之后的节点。

因此，检查点(checkpoint)之后的 **`LLM 调用、API 请求、工具调用、中断`** 等都会重新触发，最终结果可能和原始结果不同。

```python
history = graph.get_state_history(config)

for state in history:
    print(state.config) # find the config you need
    
# {
#     "configurable": {
#         "thread_id": "1",
#         "checkpoint_id": "xxx"
#     }
# }

graph.invoke(None, config=state.config)
```

##### 6.5.4.1.1. 正常运行的任务

本例仍然使用如下结构：

```json
START
  -> node_change_topic
      -> node_poem
      -> node_joke
  -> node_output
  -> END
```

**示例如下**

```python
from typing import TypedDict

from dotenv import  load_dotenv
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph,START,END
from loguru import logger
load_dotenv(override=True)

model = ChatDeepSeek(
    model = "deepseek-v4-flash",
    extra_body={
        "thinking":{
            "type":"disabled"
        }
    }
)

#1. 声明状态
class OverAllState(TypedDict):
    topic:str
    poem:str
    joke:str
    final_output:str

#1.1 输入状态
class InputState(TypedDict):
    topic:str

#1.2 输出状态
class OutputState(TypedDict):
    final_output:str

topics = ["布偶猫","狸花猫","金渐层"]
topic_index = 0
#2. 定义节点
def node_change_topic(state:InputState)->OverAllState:
    global topic_index
    logger.info("topic_index:{}",topic_index)
    sub_topic = topics[topic_index]
    topic_index += 1
    topic_index %= len(topics)

    return {
        "topic":f"{state["topic"]}:{sub_topic}"
    }
#2.1 同一个超步的位置 成功运行的节点
def node_poem(state:OverAllState) -> OverAllState:
    logger.info("node_poem正在执行")
    topic = state["topic"]
    poem = model.invoke([HumanMessage(f"写一首关于{topic}主题的七言绝句")]).content
    return {
        "poem":poem
    }
import  time
#2.2 同一个超步的位置 运行失败的节点
def node_joke(state:OverAllState) -> OverAllState:
    logger.info("node_joke正在执行")
    topic = state["topic"]
    # time.sleep(5)
    # raise Exception("人为抛异常")
    joke = model.invoke([HumanMessage(f"写一首关于{topic}主题的笑话")]).content
    return {
        "joke":joke
    }

def node_output(state:OverAllState) -> OutputState:
    logger.info("node_output正在执行")
    topic = state["topic"]
    poem = state["poem"]
    joke = state["joke"]
    final_output = f"关于{topic}的七言绝句:{poem}\n 笑话:{joke}\n"
    return {
        "final_output":final_output
    }

#3. 构建图
builder = StateGraph(state_schema=OverAllState,input_schema=InputState,output_schema=OutputState)

#3.1 添加节点
builder.add_node("node_change_topic",node_change_topic)
builder.add_node("node_poem",node_poem)
builder.add_node("node_joke",node_joke)
builder.add_node("node_output",node_output)

#3.2 添加边
builder.add_edge(START,"node_change_topic")
builder.add_edge("node_change_topic","node_poem")
builder.add_edge("node_change_topic","node_joke")
builder.add_edge("node_poem","node_output")
builder.add_edge("node_joke","node_output")
builder.add_edge("node_output",END)

#4. 添加检查点(checkpoint)后端
DB_URL = "postgresql://langgraph_user:123456@localhost:5432/langgraph_db?sslmode=disable"
from langgraph.checkpoint.postgres import PostgresSaver
with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
    #5. 第一次使用PostgresSaver作为检查点(checkpoint) 需要调用方法 setup()
    #checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)

    from IPython.display import display
    display(graph)

    config = {
        "configurable":{
            "thread_id":"chapter03-08"
        }
    }

    res = graph.invoke({"topic":"猫咪"},config=config)
    print(res)

```

**运行结果如下**

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	node_poem(node_poem)
	node_joke(node_joke)
	node_output(node_output)
	node_change_topic(node_change_topic)
	__end__([<p>__end__</p>]):::last
	__start__ --> node_change_topic;
	node_change_topic --> node_joke;
	node_change_topic --> node_poem;
	node_joke --> node_output;
	node_poem --> node_output;
	node_output --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```



```json
2026-06-12 14:32:42.019 | INFO     | __main__:node_change_topic:45 - topic_idx: 0
2026-06-12 14:32:42.022 | INFO     | __main__:node_joke:63 - node_joke 已执行
2026-06-12 14:32:42.026 | INFO     | __main__:node_poem:54 - node_poem 已执行
2026-06-12 14:32:43.585 | INFO     | __main__:node_output:72 - node_output 已执行
2026-06-12 14:32:43.588 | INFO     | __main__:<module>:102 - res: 
{
    'final_output': '关于 猫咪: 布偶猫 的七言绝句：\n《布偶猫》\n雪团云絮落仙踪，碧眼盈盈醉九重。\n静若琼瑶闲卧月，清姿一步一玲珑。\n笑话：\n一只布偶猫照镜子，惊呼：“我这么美，怎么还没当上国王？” 主人笑道：“因为你没有‘喵’民拥护。” 布偶猫委屈：“我有啊，铲屎官你不是天天跪着拥护我吗？”'
}
```

##### 6.5.4.1.2. 查看历史检查点(checkpoint)列表

**示例如下**

```python
history_checkpoints = list(graph.get_state_history(config=config))
print(history_checkpoints)
```

**运行结果如下**

```json
[
    StateSnapshot(
        values={
            'topic': '猫咪: 布偶猫',
            'poem': '《布偶猫》\n雪团云絮落仙踪，碧眼盈盈醉九重。\n静若琼瑶闲卧月，清姿一步一玲珑。',
            'joke': '一只布偶猫照镜子，惊呼：“我这么美，怎么还没当上国王？” 主人笑道：“因为你没有‘喵’民拥护。” 布偶猫委屈：“我有啊，铲屎官你不是天天跪着拥护我吗？”',
            'final_output': '关于 猫咪: 布偶猫 的七言绝句：\n《布偶猫》\n雪团云絮落仙踪，碧眼盈盈醉九重。\n静若琼瑶闲卧月，清姿一步一玲珑。\n笑话：\n一只布偶猫照镜子，惊呼：“我这么美，怎么还没当上国王？” 主人笑道：“因为你没有‘喵’民拥护。” 布偶猫委屈：“我有啊，铲屎官你不是天天跪着拥护我吗？”'
        },
        next=(),
        config={
            'configurable': {
                'thread_id': '123',
                'checkpoint_ns': '',
                'checkpoint_id': '1f166288-4ad3-67ef-8003-01d021de9403'
            }
        },
        metadata={
            'source': 'loop',
            'step': 3,
            'parents': {}
        },
        created_at='2026-06-12T06:32:43.586547+00:00',
        parent_config={
            'configurable': {
                'thread_id': '123',
                'checkpoint_ns': '',
                'checkpoint_id': '1f166288-4acf-62cf-8002-aed204b5115d'
            }
        },
        tasks=(),
        interrupts=()
    ),
    StateSnapshot(
        values={
            'topic': '猫咪: 布偶猫',
            'poem': '《布偶猫》\n雪团云絮落仙踪，碧眼盈盈醉九重。\n静若琼瑶闲卧月，清姿一步一玲珑。',
            'joke': '一只布偶猫照镜子，惊呼：“我这么美，怎么还没当上国王？” 主人笑道：“因为你没有‘喵’民拥护。” 布偶猫委屈：“我有啊，铲屎官你不是天天跪着拥护我吗？”'
        },
        next=('node_output',),
        config={
            'configurable': {
                'thread_id': '123',
                'checkpoint_ns': '',
                'checkpoint_id': '1f166288-4acf-62cf-8002-aed204b5115d'
            }
        },
        metadata={
            'source': 'loop',
            'step': 2,
            'parents': {}
        },
        created_at='2026-06-12T06:32:43.584774+00:00',
        parent_config={
            'configurable': {
                'thread_id': '123',
                'checkpoint_ns': '',
                'checkpoint_id': '1f166288-3be4-6523-8001-dd86f05c34b8'
            }
        },
        tasks=(
            PregelTask(
                id='b9042203-0ddf-f638-aeeb-e0501601381a',
                name='node_output',
                path=('__pregel_pull', 'node_output'),
                error=None,
                interrupts=(),
                state=None,
                result={
                    'final_output': '关于 猫咪: 布偶猫 的七言绝句：\n《布偶猫》\n雪团云絮落仙踪，碧眼盈盈醉九重。\n静若琼瑶闲卧月，清姿一步一玲珑。\n笑话：\n一只布偶猫照镜子，惊呼：“我这么美，怎么还没当上国王？” 主人笑道：“因为你没有‘喵’民拥护。” 布偶猫委屈：“我有啊，铲屎官你不是天天跪着拥护我吗？”'
                }
            ),
        ),
        interrupts=()
    ),
    StateSnapshot(
        values={
            'topic': '猫咪: 布偶猫'
        },
        next=('node_poem', 'node_joke'),
        config={
            'configurable': {
                'thread_id': '123',
                'checkpoint_ns': '',
                'checkpoint_id': '1f166288-3be4-6523-8001-dd86f05c34b8'
            }
        },
        metadata={
            'source': 'loop',
            'step': 1,
            'parents': {}
        },
        created_at='2026-06-12T06:32:42.020569+00:00',
        parent_config={
            'configurable': {
                'thread_id': '123',
                'checkpoint_ns': '',
                'checkpoint_id': '1f166288-3bde-6373-8000-9750b4be0665'
            }
        },
        tasks=(
            PregelTask(
                id='eab6bf51-4986-a60f-753c-e493b78c56b8',
                name='node_poem',
                path=('__pregel_pull', 'node_poem'),
                error=None,
                interrupts=(),
                state=None,
                result={
                    'poem': '《布偶猫》\n雪团云絮落仙踪，碧眼盈盈醉九重。\n静若琼瑶闲卧月，清姿一步一玲珑。'
                }
            ),
            PregelTask(
                id='a228a7bf-4c3b-2ad3-039f-12ac0107c87b',
                name='node_joke',
                path=('__pregel_pull', 'node_joke'),
                error=None,
                interrupts=(),
                state=None,
                result={
                    'joke': '一只布偶猫照镜子，惊呼：“我这么美，怎么还没当上国王？” 主人笑道：“因为你没有‘喵’民拥护。” 布偶猫委屈：“我有啊，铲屎官你不是天天跪着拥护我吗？”'
                }
            )
        ),
        interrupts=()
    ),
    StateSnapshot(
        values={
            'topic': '猫咪'
        },
        next=('node_change_topic',),
        config={
            'configurable': {
                'thread_id': '123',
                'checkpoint_ns': '',
                'checkpoint_id': '1f166288-3bde-6373-8000-9750b4be0665'
            }
        },
        metadata={
            'source': 'loop',
            'step': 0,
            'parents': {}
        },
        created_at='2026-06-12T06:32:42.018069+00:00',
        parent_config={
            'configurable': {
                'thread_id': '123',
                'checkpoint_ns': '',
                'checkpoint_id': '1f166288-3bdc-60a4-bfff-d4fdcf6e6a9e'
            }
        },
        tasks=(
            PregelTask(
                id='e0034c08-dd19-0ae9-7059-f8414745d212',
                name='node_change_topic',
                path=('__pregel_pull', 'node_change_topic'),
                error=None,
                interrupts=(),
                state=None,
                result={
                    'topic': '猫咪: 布偶猫'
                }
            ),
        ),
        interrupts=()
    ),
    StateSnapshot(
        values={},
        next=('__start__',),
        config={
            'configurable': {
                'thread_id': '123',
                'checkpoint_ns': '',
                'checkpoint_id': '1f166288-3bdc-60a4-bfff-d4fdcf6e6a9e'
            }
        },
        metadata={
            'source': 'input',
            'step': -1,
            'parents': {}
        },
        created_at='2026-06-12T06:32:42.017182+00:00',
        parent_config=None,
        tasks=(
            PregelTask(
                id='1a1ac312-152a-dcc3-2aa7-c2588f651448',
                name='__start__',
                path=('__pregel_pull', '__start__'),
                error=None,
                interrupts=(),
                state=None,
                result={
                    'topic': '猫咪'
                }
            ),
        ),
        interrupts=()
    )
]
```

##### 6.5.4.1.3. 重放

**`get_state_history()`** 返回的是历史检查点(checkpoint)列表，并且默认按照时间倒序排列。

**`history_checkpoints`** 结构如下：

```text
history_checkpoints[0] -> 最终完成后的检查点(checkpoint)
						next=()
history_checkpoints[1] -> node_output 执行前的检查点(checkpoint)
						next=('node_output',)
history_checkpoints[2] -> node_poem、node_joke 执行前的检查点(checkpoint)
						next=('node_poem', 'node_joke')
history_checkpoints[3] -> node_change_topic 执行前的检查点(checkpoint)
						next=('node_change_topic',)
history_checkpoints[4] -> __start__ 执行前的输入检查点(checkpoint)
						next=('__start__',)
```

因此，如果希望从写诗写笑话开始重放，选择 **`next == ('node_poem', 'node_joke')`** 的检查点(checkpoint)即可。

**示例如下**

```python
with PostgresSaver.from_conn_string(DB_URL) as checkpointer:
    #5. 第一次使用PostgresSaver作为检查点(checkpoint) 需要调用方法 setup()
    #checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable":{
            "thread_id":"chapter03-08"
        }
    }
    # 获取到检查点(checkpoint)历史
    history_checkpoints = list(graph.get_state_history(config=config))

    new_checkpoint = None
    #next = ('node_poem', 'node_joke')
    for checkpoint in history_checkpoints:
        if checkpoint.next == ('node_poem', 'node_joke'):
            new_checkpoint = checkpoint
            break

    # 如果想要实现replay的效果 状态填写None  config填写为之前某一个检查点(checkpoint)的config
    res = graph.invoke(None,config=new_checkpoint.config)
    print(res)
```

**运行结果如下**

```json
2026-07-07 18:31:32.019 | INFO     | __main__:node_joke:59 - node_joke正在执行
2026-07-07 18:31:32.023 | INFO     | __main__:node_poem:50 - node_poem正在执行
2026-07-07 18:31:38.158 | INFO     | __main__:node_output:69 - node_output正在执行
{'final_output': '关于猫咪:狸花猫的七言绝句:《狸花猫》\n玄文斑驳隐花光，夜半巡檐气自昂。\n踏雪无痕轻似梦，一窗明月一炉香。\n\n注：我的创作思路是捕捉狸花猫的神秘与灵动。首句“玄文斑驳”描绘其毛发纹理，如暗夜中的花纹；次句“夜半巡檐”展现其夜行习性，气度不凡。后两句以“踏雪无痕”喻其轻盈，结句“明月炉香”营造温馨画面，暗合狸花猫守护家园的意象，使全诗动静相生，虚实相映。\n 笑话:# 狸花猫的烦恼\n\n有只狸花猫去宠物医院看病，医生问它：“哪里不舒服？”\n\n狸花猫叹了口气说：“医生，我总觉得自己不够‘花’。”\n\n医生疑惑：“你可是标准的狸花猫啊，条纹多漂亮！”\n\n狸花猫委屈地说：“可是隔壁那只三花猫，人家身上有黑、白、橘三种颜色，比我缤纷多了！我整天就这一身灰黑条纹，太单调了。”\n\n医生笑了：“那你想要怎么办？”\n\n狸花猫眼睛一亮：“医生，您能不能给我开点染色剂？我想在背上加几块橘色斑点，尾巴再来点白的，耳朵尖染成黑的…”\n\n医生打断它：“等等，那样你就变成‘四不像’了！”\n\n狸花猫思考了一会，突然骄傲地挺起胸膛：“你说得对！我这一身条纹可是老祖宗传下来的迷彩服，抓老鼠时特别管用！三花猫再好看，还得靠我帮忙抓家里的老鼠呢！”\n\n医生欣慰地说：“这就对了。”\n\n狸花猫走出诊室，正好遇到一只胖橘猫。胖橘猫嘲笑它：“哟，这不是那只会抓老鼠却不会卖萌的狸花猫吗？”\n\n狸花猫淡定地回了一句：“我会抓老鼠，你会什么？”\n\n胖橘猫想了想，尴尬地说：“我…我会吃。”\n\n狸花猫笑道：“那不就成了？你负责吃，我负责抓，咱们各有各的活法！”\n\n—— **适合自己的，才是最好的。** 🐱\n'}

```

本例中：

1. **`node_poem`** 和 **`node_joke`** 被重新执行。
2. **`node_output`** 被重新执行。

由于没有执行change_topic,所以可以一直写同一只猫的内容

```text
历史 checkpoint
      ↓
读取历史状态
      ↓
修改 state
      ↓
创建新的 checkpoint / 分支
      ↓
继续执行后面的节点
```

```python
history = graph.get_state_history(config)

checkpoint_config = state.config # find the config you need
    
# {
#     "configurable": {
#         "thread_id": "1",
#         "checkpoint_id": "xxx"
#     }
# }
state = graph.get_state(checkpoint_config)

print(state.values)

# update value
graph.update_state(
    checkpoint_config,
    {"topic": "Java"}
)

graph.invoke(
    None,
    config=checkpoint_config
)
```

#### 6.5.4.2. Fork

##### 6.5.4.2.1. 用法说明

**`Fork`** 依赖状态图的 **`update_state()`** 方法

**示例如下**

```python
change_input_config = graph.update_state(
    config=before_router_config.config,
    values={"user_input": "帮我写一个关于布偶猫的笑话"},
    as_node=START
)
```

**其中**：

- **`config`**：历史检查点(checkpoint)的配置信息，通常包含 **`thread_id`**、**`checkpoint_ns`** 和 **`checkpoint_id`**。**`LangGraph`** 会据此检索历史检查点(checkpoint)

- **`values`**：要应用到该检查点(checkpoint)上的状态更新

- **`as_node`**：指定这次状态更新应当被视为哪个节点产生的输出。

  **`LangGraph`** 会按照该节点的写入逻辑，将 **`values`** 写入状态通道，并据此决定后续从哪些节点继续执行。

  基于分叉检查点(checkpoint)运行时，并不是重新执行 **`as_node`** 本身，而是从它的后继节点继续推进

  如 **`START`** 所属超步编号为 **`0`**，则恢复时从编号为 **`1`** 的超步开始运行。

**注意**

> **`update_state()`** 不会修改原来的历史检查点(checkpoint)，而是基于某个历史检查点(checkpoint)创建一个新的检查点(checkpoint)。这个新检查点(checkpoint)相当于一条新的执行分支。

##### 6.5.4.2.2. 准备检查点(checkpoint)历史

###### 6.5.4.2.2.1. 准备运行图并运行一次

**示例如下**

```python
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_deepseek import ChatDeepSeek
from loguru import logger
from dotenv import load_dotenv

load_dotenv(override=True)


model = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)

class OverAllState(TypedDict):
    username: str
    user_input: str
    output: str

def router_node(state: OverAllState) -> StructuredOutputState:
    logger.info("路由节点已执行")
    user_input = state["user_input"]
    res = model_with_structure.invoke(
        [
            HumanMessage(user_input)
        ]
    )
    logger.info("路由结果-res: {}", res)

    return res

def router(state: StructuredOutputState) -> Literal["node_poem", "node_joke", "node_default"]:
    logger.info("路由函数已执行")
    if state["mode"] == "poem":
        logger.info("路由至 node_poem 节点")
        return "node_poem"
    elif state["mode"] == "joke":
        logger.info("路由至 node_joke 节点")
        return "node_joke"

    logger.info("路由至兜底节点")
    return "node_default"

def node_poem(state: StructuredOutputState) -> OverAllState:
    logger.info(f"node_poem 已执行")
    topic = state["topic"]
    poem = model.invoke([HumanMessage(f"写一首关于 {topic} 的七言绝句，不要赏析，只给诗句即可")]).content

    return {
        "output": poem
    }

def node_joke(state: StructuredOutputState) -> OverAllState:
    logger.info(f"node_joke 已执行")
    topic = state["topic"]
    joke = model.invoke([HumanMessage(f"写一个关于 {topic} 的笑话，尽可能简单，不超过一百字")]).content

    return {
        "output": joke
    }

def node_default(state: StructuredOutputState) -> OverAllState:
    logger.info(f"node_default 已执行")

    return {
        "output": "无法处理的任务类型"
    }

builder = StateGraph(state_schema=OverAllState)
builder.add_node("router_node", router_node)
builder.add_node("node_poem", node_poem)
builder.add_node("node_joke", node_joke)
builder.add_node("node_default", node_default)

builder.add_edge(START, "router_node")
builder.add_conditional_edges(
    "router_node",
    router,
    path_map=["node_poem", "node_joke", "node_default"]
)
builder.add_edge("node_poem", END)
builder.add_edge("node_joke", END)
builder.add_edge("node_default", END)

checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "123"}}
graph = builder.compile(checkpointer=checkpointer)

from IPython.display import display
display(graph)

res = graph.invoke({
    "username": "小黄",
    "user_input": "帮我写一首关于布偶猫的七言绝句"
}, config=config)
print(res)
```

执行逻辑如下：

1. **`router_node`** 根据用户输入提取结构化结果，例如 **`{"topic": "布偶猫", "mode": "poem"}`**。
2. **`router`** 根据 **`mode`** 决定后续分支。
3. 如果 **`mode == "poem"`**，进入 **`node_poem`**。
4. 如果 **`mode == "joke"`**，进入 **`node_joke`**。
5. 如果无法识别，则进入 **`node_default`**。

运行结果如下

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	node_poem(node_poem)
	node_joke(node_joke)
	node_default(node_default)
	router_node(router_node)
	__end__([<p>__end__</p>]):::last
	__start__ --> router_node;
	router_node -.-> node_default;
	router_node -.-> node_joke;
	router_node -.-> node_poem;
	node_joke --> __end__;
	node_poem --> __end__;
	node_default --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

```json
2026-06-11 14:26:48.353 | INFO     | __main__:router_node:31 - 路由节点已执行
2026-06-11 14:26:49.618 | INFO     | __main__:router_node:38 - 路由结果-res: {'topic': '布偶猫', 'mode': 'poem'}
2026-06-11 14:26:49.619 | INFO     | __main__:router:43 - 路由函数已执行
2026-06-11 14:26:49.619 | INFO     | __main__:router:45 - 路由至 node_poem 节点
2026-06-11 14:26:49.620 | INFO     | __main__:node_poem:55 - node_poem 已执行
{
    'username': '小黄', 
    'user_input': '帮我写一首关于布偶猫的七言绝句', 
    'output': '《布偶猫》\n蓝眼澄明玉作团，云踪雪影卧雕栏。\n无端一跃梅花落，犹带娇声唤月寒。'
}
```

###### 6.5.4.2.2.2. 获取历史检查点(checkpoint)列表

**示例如下**

```python
history_checkpoints = list(graph.get_state_history(config=config))
print(history_checkpoints)
```

**运行结果如下**

```json
[
    StateSnapshot(
        values={
            "username": "小黄",
            "user_input": "帮我写一首关于布偶猫的七言绝句",
            "output": (
                "《布偶猫》\n"
                "蓝眼澄明玉作团，云踪雪影卧雕栏。\n"
                "无端一跃梅花落，犹带娇声唤月寒。"
            ),
            "topic": "布偶猫",
            "mode": "poem",
        },
        next=(),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f1655e8-8784-65e9-8002-e21663d55c94",
            }
        },
        metadata={
            "source": "loop",
            "step": 2,
            "parents": {},
        },
        created_at="2026-06-11T06:26:51.611071+00:00",
        parent_config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f1655e8-7488-66c4-8001-f46b7c2d9a4e",
            }
        },
        tasks=(),
        interrupts=(),
    ),

    StateSnapshot(
        values={
            "username": "小黄",
            "user_input": "帮我写一首关于布偶猫的七言绝句",
            "topic": "布偶猫",
            "mode": "poem",
        },
        next=("node_poem",),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f1655e8-7488-66c4-8001-f46b7c2d9a4e",
            }
        },
        metadata={
            "source": "loop",
            "step": 1,
            "parents": {},
        },
        created_at="2026-06-11T06:26:49.620436+00:00",
        parent_config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f1655e8-6871-6073-8000-ddcbe680da21",
            }
        },
        tasks=(
            PregelTask(
                id="cb3a8901-3a87-d0b0-3bab-d1adda5563d8",
                name="node_poem",
                path=("__pregel_pull", "node_poem"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "output": (
                        "《布偶猫》\n"
                        "蓝眼澄明玉作团，云踪雪影卧雕栏。\n"
                        "无端一跃梅花落，犹带娇声唤月寒。"
                    )
                },
            ),
        ),
        interrupts=(),
    ),

    StateSnapshot(
        values={
            "username": "小黄",
            "user_input": "帮我写一首关于布偶猫的七言绝句",
        },
        next=("router_node",),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f1655e8-6871-6073-8000-ddcbe680da21",
            }
        },
        metadata={
            "source": "loop",
            "step": 0,
            "parents": {},
        },
        created_at="2026-06-11T06:26:48.352564+00:00",
        parent_config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f1655e8-686b-66c7-bfff-24e0e8d44f4c",
            }
        },
        tasks=(
            PregelTask(
                id="fb88436e-b163-6308-6341-7d8368e041f9",
                name="router_node",
                path=("__pregel_pull", "router_node"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "topic": "布偶猫",
                    "mode": "poem",
                },
            ),
        ),
        interrupts=(),
    ),

    StateSnapshot(
        values={},
        next=("__start__",),
        config={
            "configurable": {
                "thread_id": "123",
                "checkpoint_ns": "",
                "checkpoint_id": "1f1655e8-686b-66c7-bfff-24e0e8d44f4c",
            }
        },
        metadata={
            "source": "input",
            "step": -1,
            "parents": {},
        },
        created_at="2026-06-11T06:26:48.350273+00:00",
        parent_config=None,
        tasks=(
            PregelTask(
                id="a1b5e5fb-ee60-fd7b-68bc-475fa9cb66e2",
                name="__start__",
                path=("__pregel_pull", "__start__"),
                error=None,
                interrupts=(),
                state=None,
                result={
                    "username": "小黄",
                    "user_input": "帮我写一首关于布偶猫的七言绝句",
                },
            ),
        ),
        interrupts=(),
    ),
]
```

###### 6.5.4.2.2.3. 获取 `router_node` 之前的检查点(checkpoint)

**示例如下**

**`router_node`** 的上一个检查点(checkpoint)的 **`next`** 字段为 **`('router_node',)`**，据此筛选检查点(checkpoint)

```python
before_router_checkpoint = next(h for h in history_checkpoints if h.next == ('router_node',))
before_router_checkpoint
```

**运行结果如下**

```json
StateSnapshot(
    values={
        "username": "小黄",
        "user_input": "帮我写一首关于布偶猫的七言绝句",
    },
    next=("router_node",),
    config={
        "configurable": {
            "thread_id": "123",
            "checkpoint_ns": "",
            "checkpoint_id": "1f1655e8-6871-6073-8000-ddcbe680da21",
        }
    },
    metadata={
        "source": "loop",
        "step": 0,
        "parents": {},
    },
    created_at="2026-06-11T06:26:48.352564+00:00",
    parent_config={
        "configurable": {
            "thread_id": "123",
            "checkpoint_ns": "",
            "checkpoint_id": "1f1655e8-686b-66c7-bfff-24e0e8d44f4c",
        }
    },
    tasks=(
        PregelTask(
            id="fb88436e-b163-6308-6341-7d8368e041f9",
            name="router_node",
            path=("__pregel_pull", "router_node"),
            error=None,
            interrupts=(),
            state=None,
            result={
                "topic": "布偶猫",
                "mode": "poem",
            },
        ),
    ),
    interrupts=(),
)
```

该检查点(checkpoint)表示：

> 用户输入已经写入状态，下一步将要执行 **`router_node`**。

从这个检查点(checkpoint)分叉，本文采用两种方案（还可以有别的方案）：

1. 修改输入，让 **`router_node`** 重新执行。见 **`6.5.4.2.3.`** 节
2. 直接伪造 **`router_node`** 的输出，从而跳过 **`router_node`**。见 **`6.5.4.2.4.`** 节

##### 6.5.4.2.3. 只改状态

###### 6.5.4.2.3.1. 更新配置

**示例如下**

```python
change_input_config = graph.update_state(
    config=before_router_checkpoint.config,
    values={"user_input": "帮我写一个关于布偶猫的笑话"},
    as_node=START
)
change_input_config
```

这里把 **`as_node`** 设置为 **`START`**，表示：

> 把这次状态更新视为 **`START`** 的输出。

因此，后续运行时会从 **`START`** 的后继节点继续推进，也就是重新执行 **`router_node`**。

**上述代码运行结果如下**

```json
{
    "configurable": {
        "thread_id": "123",
        "checkpoint_ns": "",
        "checkpoint_id": "1f1655f8-86fb-6dd1-8001-9e7808dfaf1a",
    }
}
```

执行 **`update_state()`** 后会返回一个新的配置，其中包含新的 **`checkpoint_id`**。这个新的 **`checkpoint_id`** 不同于原有历史检查点(checkpoint)，说明 **`update_state()`** 创建了一个新的检查点(checkpoint)分支。

需要注意：

> **`update_state()`** 返回的是新检查点(checkpoint)的配置，不是新状态本身。

新的状态已经被记录在检查点(checkpoint)后端，运行时生效。可以执行以下代码查看新建的检查点(checkpoint)快照：

```python
graph.get_state(change_input_config)
```

此处不再演示。

###### 6.5.4.2.3.2. 重新运行

**示例如下**

```python
graph.invoke(None, change_input_config)
```

**运行结果如下**

```json
2026-06-11 14:13:44.267 | INFO     | __main__:router_node:31 - 路由节点已执行
2026-06-11 14:13:45.222 | INFO     | __main__:router_node:38 - 路由结果-res: {'topic': '布偶猫', 'mode': 'joke'}
2026-06-11 14:13:45.223 | INFO     | __main__:router:43 - 路由函数已执行
2026-06-11 14:13:45.223 | INFO     | __main__:router:48 - 路由至 node_joke 节点
2026-06-11 14:13:45.225 | INFO     | __main__:node_joke:64 - node_joke 已执行
{
    "username": "小黄",
    "user_input": "帮我写一个关于布偶猫的笑话",
    "output": (
        "一只布偶猫照镜子，问：“魔镜魔镜，谁是世界上最美的猫？”  \n"
        "镜子没说话。  \n"
        "猫急了：“快说呀！不然我卖萌把你甜碎。”  \n"
        "镜子终于开口：“……我死机了，重启中。”"
    ),
}
```

由运行结果可知：

1. **`user_input`** 已经从“帮我写一首关于布偶猫的七言绝句”变成“帮我写一个关于布偶猫的笑话”。而 **`username`** 没有被更新，仍然保持原值 **`"小黄"`**。说明 **`values`** 是 **状态更新** 而非完整的 **状态替换**。
2. **`router_node`** 开始的节点全部重新运行。由于新的输入被识别为笑话任务，因此后续进入 **`node_joke`**。

##### 6.5.4.2.4. 改状态且跳过某些节点

###### 6.5.4.2.4.1. 更新配置

如果不希望重新执行 **`router_node`**，而是直接指定 **`router_node`** 的输出，可以把 **`as_node`** 设置为 **`"router_node"`**。

**示例如下**

```python
skip_router_config = graph.update_state(
    config=before_router_checkpoint.config,
    values={"topic": "狸花猫", "mode": "笑话"},
    as_node="router_node"
)
skip_router_config
```

1. 此处把 **`{"topic": "狸花猫", "mode": "笑话"}`** 当作 **`router_node`** 的输出。

2. 因此，后续运行时，**`router_node`** 本身不会被重新执行，从它的后继节点继续推进。

3. 由于 **`router_node`** 挂载了条件边：

   ```python
   builder.add_conditional_edges(
       "router_node",
       router,
       path_map=["node_poem", "node_joke", "node_default"]
   )
   ```

   所以当 **`update_state()`** 把 **`values`** 视为 **`router_node`** 的输出时，条件路由函数 **`router`** 也会根据这次更新后的状态计算后续分支。

   这里把 **`mode`** 设置为 **`"笑话"`**，**`router`** 函数命中兜底分支 **`node_default`**。

**运行结果如下**

```json
2026-06-11 14:14:02.223 | INFO     | __main__:router:43 - 路由函数已执行
2026-06-11 14:14:02.223 | INFO     | __main__:router:51 - 路由至兜底节点
{
    "configurable": {
        "thread_id": "123",
        "checkpoint_ns": "",
        "checkpoint_id": "1f1655cb-de15-6c4e-8001-ca03b011e993",
    }
}
```

可以看到：

1. **`update_state()`** 创建了一个新的检查点(checkpoint)，返回的 **`skip_router_config`** 指向这个新检查点(checkpoint)。

2. 日志中出现了：

   ```text
   路由函数已执行
   路由至兜底节点
   ```

   这说明 **`router_node`** 的节点函数没有被重新执行，但与 **`router_node`** 条件边相关的路由函数被执行了。

3. **`update_state()`** 底层会根据 **`as_node`** 找到对应图节点的 **`writers`**。

   对于 **`router_node`**，**`writers`** 包括状态写入器和条件边对应的路由逻辑。

4. 执行 **`update_state()`** 时，传入的 **`values`** 会被当作 **`router_node`** 的输出，交给该节点的 **`writers`** 处理。

   状态写入器会把 **`values`** 转换为对状态 **`Channel`** 的写入；条件边对应的路由逻辑会根据当前状态计算后继分支，并生成类似 **`branch:to:node_default`** 这样的触发通道写入。

5. 随后，**`update_state()`** 会把这些写入应用到检查点(checkpoint)中，并保存一个新的检查点(checkpoint)。

   因此，状态更新和路由结果已经记录在 **`skip_router_config`** 指向的新检查点(checkpoint)中。后续再调用：

   ```python
   graph.invoke(None, skip_router_config)
   ```

   时，运行图会基于这个新检查点(checkpoint)继续执行，被触发的后继节点也会继续运行。

综上，**`graph.update_state()`** 的作用可以理解为：

> 不执行指定节点的节点函数本身，而是把传入的 **`values`** 当作该节点已经产生的输出，执行该节点对应的 **`writers`**，应用状态写入和可能存在的路由写入，然后生成一个新的检查点(checkpoint)。

因此，在本例中可以看到：**`router_node`** 的节点逻辑没有被执行，但它挂载的路由函数被执行，状态更新也已经生效。

###### 6.5.4.2.4.2. 重新运行

**示例如下**

```python
graph.invoke(None, skip_router_config)
```

**运行结果如下**

```json
2026-06-11 16:49:42.501 | INFO     | __main__:node_default:73 - node_default 已执行
{
  "username": "小黄",
  "user_input": "帮我写一首关于布偶猫的七言绝句",
  "output": "无法处理的任务类型"
}
```

和分析一致，触发了兜底分支。

这种方式常用于测试场景。例如：

* 跳过不稳定的大模型路由节点。
* 手动指定路由结果，测试不同分支。
* 修改历史状态，验证后续节点能否正常运行。
* 基于同一个历史检查点(checkpoint)探索多条可能的执行路径。

不过在正式业务流程中，应谨慎使用这种方式。因为它是在伪造某个节点的输出，如果伪造的状态不符合后续节点的预期，可能导致逻辑混乱，甚至抛出运行时异常。

# 7. 图记忆管理

 **`Agent`** 的三种记忆：

- **短期记忆**：通过运行时状态 **`State`** 访问，并由检查点(checkpoint)存储器 **`Checkpointer`** 保存，它按照 **`thread_id`** 组织，可以实现线程内的记忆共享。

- **长期记忆**：通过长期记忆存储器 **`Store`** 访问和存储。数据通常按照元组类型的命名空间组织，以键值对的形式存储，提供跨会话的记忆共享。

  - 可以按照命名空间+可选的过滤条件精确检索

  - 也可以通过语义模糊匹配。

- **运行时上下文**：通过上下文对象 **`Context`** 访问，只对本次调用生效，不会被持久化。

  它更适合传递本次运行所需的外部依赖或调用参数，如用户名、模型配置、数据库连接、权限信息等。

我们不止一次提到，**`Agent`** 底层就是一个简易的 **`ReAct`** 架构的 **`LangGraph`** 状态图，所以 **`Agent`** 的记忆机制本质上就是 **`LangGraph`** 状态图的记忆机制。

生产环境建议用基于持久化数据库的记忆存储器，如 **`PostgresSaver`** 和 **`PostgresStore`**。

## 7.1. 短期记忆

只要编译图时启用了 **`Checkpointer`**，并且调用图时复用同一个 **`thread_id`**，**`LangGraph`** 就可以在多次调用之间共享同一会话线程下的状态。

我们一直在用，不在赘述。

## 7.2. 长期记忆

长期记忆存储器在编译图时通过 **`store`** 参数传递，图节点中可以通过 **`Runtime`** 对象访问。

本节会在一个案例中同时用到短期记忆和长期记忆，两者均使用 **`PostgreSQL`** 作为持久化后端——短期记忆通过 **`PostgresSaver`**，长期记忆通过 **`PostgresStore`**，数据不会随进程退出而丢失。 

### 7.2.1. 准备长期记忆

**示例如下**

```python
from typing import Final, Tuple
from langgraph.store.postgres import PostgresStore

# 与 6.2.4 节使用同一数据库
DB_URL = "postgresql://langgraph_user:123456@localhost:5432/langgraph_db?sslmode=disable"
with PostgresStore.from_conn_string(DB_URL) as store:
    # setup() 创建长期记忆所需的表结构，幂等操作
    store.setup()

    # 命名空间 - 层级结构：("users", "用户名")
    # Final 的作用是告诉静态类型检查器，该变量不应该被重新赋值，但 python 运行时不会阻止重新赋值
    USERS_NS: Final[Tuple[str]] = ("users", )
    PREFERENCES_KEY: Final[str] = "preferences"

    # 三层: (领域, 用户实体)
    # 每个 key 存储该用户的不同类型数据
    namespace1 = (*USERS_NS, "Alice")
    value1 = {
        "course": "计算机组成原理",
        "sports": "跑步",
        "food": "紫光园奶皮子酸奶"
    }

    namespace2 = (*USERS_NS, "Bob")
    value2 = {
        "course": "数字电路与模拟电路",
        "sports": "跑步",
        "food": "奶皮子糖葫芦"
    }

    namespace3 = (*USERS_NS, "Black")
    value3 = {
        "course": "数字电路与模拟电路",
        "sports": "羽毛球",
        "food": "紫光园奶皮子酸奶"
    }

    store.put(namespace1, PREFERENCES_KEY, value1)
    store.put(namespace2, PREFERENCES_KEY, value2)
    store.put(namespace3, PREFERENCES_KEY, value3)

    for item in store.search(USERS_NS):
        print(item)
```

**运行结果如下**

```json
Item(
    namespace=['users', 'Alice'],
    key='preferences',
    value={
        'course': '计算机组成原理',
        'sports': '跑步',
        'food': '紫光园奶皮子酸奶'
    },
    created_at='2026-06-12T09:54:50.593300+00:00',
    updated_at='2026-06-12T09:54:50.593303+00:00',
    score=None
)

Item(
    namespace=['users', 'Bob'],
    key='preferences',
    value={
        'course': '数字电路与模拟电路',
        'sports': '跑步',
        'food': '奶皮子糖葫芦'
    },
    created_at='2026-06-12T09:54:50.593329+00:00',
    updated_at='2026-06-12T09:54:50.593329+00:00',
    score=None
)

Item(
    namespace=['users', 'Black'],
    key='preferences',
    value={
        'course': '数字电路与模拟电路',
        'sports': '羽毛球',
        'food': '紫光园奶皮子酸奶'
    },
    created_at='2026-06-12T09:54:50.593348+00:00',
    updated_at='2026-06-12T09:54:50.593348+00:00',
    score=None
)
```

这里使用 **`USERS_NS`** 作为顶层命名空间，`store.search(USERS_NS)` 会查询所有以 **`("users",)`** 开头的记忆数据。返回的每条数据都是一个 **`Item`** 实例。

需要注意：

1. **`namespace`** 使用元组是硬性约束，天然表达层级结构。本例使用两层命名空间 **`("users", "Alice")`**，清晰区分了**领域（`users`）**和**实体（用户名）**，而具体存储什么类型的数据则由 **`key`** 参数表达（如 **`"preferences"`**）。

2. **`(*USERS_NS, "Alice")`** 的写法表示对已有元组进行解包，再拼接新的命名空间片段。

3. **`PostgresStore`** 将长期记忆持久化到 **`PostgreSQL`** 中，程序重启后数据不会丢失，适合生产环境。本节使用和 **6.2.4节**、**6.5.4.1节** 相同的数据库实例。

4. 如果希望支持语义检索，需要在 **`Store`** 中配置索引和 **`embedding`** 函数。未配置索引时，**`search()`** 只能按照命名空间和过滤条件检索。

### 7.2.2. 访问长期记忆

长期记忆存储器可以通过节点函数中的 **`runtime.store`** 访问。

**示例如下**

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.graph.message import MessagesState
from langgraph.runtime import Runtime
from langchain.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek

from loguru import logger
from dotenv import load_dotenv
load_dotenv(override=True)

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)

class OverAllState(MessagesState):
    username: str
    user_input: str
    output: str
    preferences: dict[str, str] # 用户偏好

def check_preferences_node(state: OverAllState, runtime: Runtime) -> OverAllState:
    # 从短期记忆中获取用户名
    username = state["username"]
    # 获取长期记忆存储器 store
    store = runtime.store

    # 根据用户信息拼接命名空间，查询用户偏好
    namespace = (*USERS_NS, username)
    key = PREFERENCES_KEY
    item = store.get(namespace, key)

    if not item:
        logger.warning("长期记忆中没有 {} 的偏好数据", username)
        return {}

    logger.info("长期记忆中保存的用户偏好: {}", item.value)
    return {
        "preferences": item.value
    }

# 路由函数，如果状态中没有用户偏好，则从长期记忆中查询，否则直接执行模型节点
def router(state: OverAllState) -> Literal["check_preferences_node", "llm_node"]:
    if not state.get("preferences"):
        logger.info("需要从长期记忆中查询用户偏好")
        return "check_preferences_node"
    logger.info("用户偏好已存在，不必查询")
    return "llm_node"

def llm_node(state: OverAllState) -> OverAllState:
    # 长期记忆中未必有用户偏好
    preferences = state.get("preferences", {})
    user_input = state["user_input"]
    human_prompt = (f"这是用户的偏好: \n{preferences}\n"
              f"这是用户的需求: \n{user_input}\n")
    system_prompt = "请根据用户偏好解决用户需求"

    # 获取历史消息列表，如果历史为空则初始化系统提示词
    messages: list[SystemMessage | HumanMessage | AIMessage | ToolMessage] = [
        SystemMessage(content=system_prompt)
    ] if not state.get("messages", []) else state["messages"]

    model_response = model.invoke(messages + [HumanMessage(content=human_prompt)])
    output = model_response.content

    return {
        "messages": messages + [HumanMessage(content=human_prompt), model_response],
        "output": output
    }

builder = StateGraph(state_schema=OverAllState)
builder.add_node("check_preferences_node", check_preferences_node)
builder.add_node("llm_node", llm_node)
builder.add_conditional_edges(START, router, path_map=["check_preferences_node", "llm_node"])
builder.add_edge("check_preferences_node", "llm_node")
builder.add_edge("llm_node", END)

# DB_URL 与 6.2.4 节一致
DB_URL = "postgresql://langgraph_user:123456@localhost:5432/langgraph_db?sslmode=disable"
# 编译时传递短期记忆和长期记忆存储器（均使用 PostgreSQL）
with PostgresSaver.from_conn_string(DB_URL) as checkpointer, \
     PostgresStore.from_conn_string(DB_URL) as store:
    checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer, store=store)
    from IPython.display import display
    display(graph)

    config = {"configurable": {"thread_id": "345"}}
    res = graph.invoke({"username": "Alice", "user_input": "我有点无聊，和我聊聊天吧"}, config=config)

    print('=' * 30, '->  <-', '=' * 30)
    print(res)

    # 第二次调用：复用同一个 thread_id，观察短期记忆缓存效果
    new_res = graph.invoke({"user_input": "推荐一下酸奶"}, config=config)
    new_messages = new_res.pop("messages")
    print('=' * 30, '->  <-', '=' * 30)
    print(new_res)
```

**运行结果如下**

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	check_preferences_node(check_preferences_node)
	llm_node(llm_node)
	__end__([<p>__end__</p>]):::last
	__start__ -.-> check_preferences_node;
	__start__ -.-> llm_node;
	check_preferences_node --> llm_node;
	llm_node --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

```json
2026-06-12 18:53:39.351 | INFO     | __main__:router:51 - 需要从长期记忆中查询用户偏好
2026-06-12 18:53:39.356 | INFO     | __main__:check_preferences_node:43 - 长期记忆中保存的用户偏好: {'course': '计算机组成原理', 'sports': '跑步', 'food': '紫光园奶皮子酸奶'}
============================== -> 不包含消息列表的运行结果 <- ==============================
{
    'username': 'Alice',
    'user_input': '我有点无聊，和我聊聊天吧', 
    'output': '哈哈，好啊！正好你学了《计算机组成原理》，我有个脑洞问题想请教一下：如果让你把“跑步的节奏感”比喻成计算机里的一个部件（比如CPU、内存、流水线），你会怎么对应呢？😄 顺便，我推荐你边慢跑边思考这个问题，跑完再来份紫光园奶皮子酸奶犒劳自己，简直完美！', 
    'preferences': {
        'course': '计算机组成原理', 
        'sports': '跑步', 
        'food': '紫光园奶皮子酸奶'
    }
}
============================== -> 消息列表 <- ==============================
================================ System Message ================================

请根据用户偏好解决用户需求
================================ Human Message =================================

这是用户的偏好: 
{
    'course': '计算机组成原理', 
    'sports': '跑步', 
    'food': '紫光园奶皮子酸奶'
}
这是用户的需求: 
我有点无聊，和我聊聊天吧

================================== Ai Message ==================================

哈哈，好啊！正好你学了《计算机组成原理》，我有个脑洞问题想请教一下：如果让你把“跑步的节奏感”比喻成计算机里的一个部件（比如CPU、内存、流水线），你会怎么对应呢？😄 顺便，我推荐你边慢跑边思考这个问题，跑完再来份紫光园奶皮子酸奶犒劳自己，简直完美！

```

**运行流程如下**：

1. 第一次调用时，状态中还没有 **`preferences`**。
2. 路由函数会路由至 **`check_preferences_node`**。
3. **`check_preferences_node`** 从长期记忆存储器中读取用户偏好。
4. 读取到的偏好会写入图状态，成为当前会话线程中的短期记忆。
5. 后续节点可以直接从状态中读取 **`preferences`**。

### 7.2.3. 再次调用

在同一个 **`with`** 块中再次调用计算图，并复用同一个 **`thread_id`**，可以看到短期记忆的效果。该代码已包含在上节的 **`with`** 代码块末尾。

**运行结果如下**

```json
2026-06-12 18:54:12.446 | INFO     | __main__:router:53 - 用户偏好已存在，不必查询
============================== -> 不包含消息列表的运行结果 <- ==============================
{
    'username': 'Alice', 
    'user_input': '推荐一下酸奶', 
    'output': '哈哈，看来你对紫光园奶皮子酸奶是真爱啊！不过既然你点名要推荐酸奶，我得先问一句——你是想换换口味试试别的，还是想让我帮你吹爆紫光园这款？（毕竟按你的偏好，它简直是“计算机组成原理”级别的完美搭配：**奶皮子像CPU缓存一样浓缩精华，酸甜度像流水线调度一样平衡，吃完跑步时还能提供“数据预取”般的能量**😂）\n\n如果是**忠于原味**，那必须再安利一次紫光园：浓稠拉丝、奶香爆炸，跑完步来一罐，简直像给肌肉做了个“中断响应”！如果是**想尝鲜**，我推荐：\n1. **兰格格天边的额吉**——蒙古风味，酸度低、像RISC精简指令集一样纯净；\n2. **简爱父爱配方**——无添加糖，适合跑步后控糖，像低功耗处理器；\n3. **卡士原态酪乳**——口感像固态硬盘般扎实绵密，适合当夜宵。\n\n不过……你确定要换掉紫光园吗？🤔', 
    'preferences': {
        'course': '计算机组成原理', 
        'sports': '跑步', 
        'food': '紫光园奶皮子酸奶'
    }
}
============================== -> 消息列表 <- ==============================
================================ System Message ================================

请根据用户偏好解决用户需求
================================ Human Message =================================

这是用户的偏好: 
{
    'course': '计算机组成原理', 
    'sports': '跑步', 
    'food': '紫光园奶皮子酸奶'
}
这是用户的需求: 
我有点无聊，和我聊聊天吧

================================== Ai Message ==================================

哈哈，好啊！正好你学了《计算机组成原理》，我有个脑洞问题想请教一下：如果让你把“跑步的节奏感”比喻成计算机里的一个部件（比如CPU、内存、流水线），你会怎么对应呢？😄 顺便，我推荐你边慢跑边思考这个问题，跑完再来份紫光园奶皮子酸奶犒劳自己，简直完美！
================================ Human Message =================================

这是用户的偏好: 
{
    'course': '计算机组成原理', 
    'sports': '跑步', 
    'food': '紫光园奶皮子酸奶'
}
这是用户的需求: 
推荐一下酸奶

================================== Ai Message ==================================

哈哈，看来你对紫光园奶皮子酸奶是真爱啊！不过既然你点名要推荐酸奶，我得先问一句——你是想换换口味试试别的，还是想让我帮你吹爆紫光园这款？（毕竟按你的偏好，它简直是“计算机组成原理”级别的完美搭配：**奶皮子像CPU缓存一样浓缩精华，酸甜度像流水线调度一样平衡，吃完跑步时还能提供“数据预取”般的能量**😂）

如果是**忠于原味**，那必须再安利一次紫光园：浓稠拉丝、奶香爆炸，跑完步来一罐，简直像给肌肉做了个“中断响应”！如果是**想尝鲜**，我推荐：
1. **兰格格天边的额吉**——蒙古风味，酸度低、像RISC精简指令集一样纯净；
2. **简爱父爱配方**——无添加糖，适合跑步后控糖，像低功耗处理器；
3. **卡士原态酪乳**——口感像固态硬盘般扎实绵密，适合当夜宵。

不过……你确定要换掉紫光园吗？🤔
```

第二次运行前，状态中已经包含：

* **`username`**
* **`preferences`**
* **`messages`**
* 上一次调用生成的 **`output`**

因此路由函数会直接判断：

```python
logger.info("用户偏好已存在，不必查询")
```

因此，本次调用不会重新访问长期记忆存储器，而是直接复用当前会话状态中缓存的用户偏好。

这正是短期记忆和长期记忆协作的典型模式：

> 第一次调用从长期记忆中读取数据，并写入短期记忆；后续同一会话线程内优先复用短期记忆，避免重复查询长期记忆。

## 7.3. 运行时上下文

在某些场景下，我们希望在调用图时传入一些**仅对当次调用有效**的信息，比如当前登录用户、请求来源、调用方标识等。这些信息不适合放入图状态（图状态会被持久化并在同一会话中跨调用共享），而应该通过**运行时上下文**（Runtime Context）传递。

运行时上下文的特点是：

> **==仅对本次调用生效，不会被持久化，也不会在同一会话的下一次调用中自动恢复。==**

运行时上下文的使用方式：

1. 初始化状态图时，使用 **`context_schema`** 定义上下文类型。
2. 调用图时通过 **`context`** 参数传入上下文对象。
3. 节点或路由函数中通过 **`runtime.context`** 访问上下文。

### 7.3.1. 访问运行时上下文

下面通过一个简单的客服机器人案例演示运行时上下文的用法。

**场景**：不同的用户调用同一个机器人，机器人根据运行时上下文中的**用户名**和**会员等级**，生成不同风格的回复。

**图结构**：

```text
START -> llm_node -> END
```

**示例如下**

```python
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.runtime import Runtime
from langchain.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek

from loguru import logger
from dotenv import load_dotenv
load_dotenv(override=True)

model = ChatDeepSeek(
    model="deepseek-v4-flash",
    extra_body={
        "thinking": {
            "type": "disabled"
        }
    }
)

# 1. 定义运行时上下文类型
@dataclass
class UserContext:
    username: str
    membership_level: str  # "普通用户" / "VIP"

# 2. 定义图状态
class OverAllState(MessagesState):
    user_input: str
    output: str

# 3. 定义节点：通过 runtime.context 访问运行时上下文
def llm_node(state: OverAllState, runtime: Runtime[UserContext]) -> OverAllState:
    runtime_context = runtime.context

    if runtime_context:
        username = runtime_context.username
        level = runtime_context.membership_level
        logger.info(f"当前用户: {username}, 会员等级: {level}")

        if level == "VIP":
            system_prompt = f"你是高级客服助理。当前VIP用户是{username}，请使用尊称'您'，语气热情周到，回复末尾加上'🎖️VIP专属服务'。"
        else:
            system_prompt = f"你是普通客服助理。当前用户是{username}，请友好简洁地回复。"
    else:
        logger.warning("运行时上下文为空，使用默认风格")
        system_prompt = "你是客服助理，请友好简洁地回复。"

    user_input = state["user_input"]
    messages = state.get("messages", [])
    response = model.invoke(
        [HumanMessage(content=system_prompt)] +
        messages +
        [HumanMessage(content=user_input)]
    )

    return {
        "messages": [response],
        "output": response.content
    }

# 4. 构建图，传入 context_schema
builder = StateGraph(state_schema=OverAllState, context_schema=UserContext)
builder.add_node("llm_node", llm_node)
builder.add_edge(START, "llm_node")
builder.add_edge("llm_node", END)

graph = builder.compile()

# === 第一次调用：传入运行时上下文（VIP用户） ===
print("=" * 30, "第一次调用：VIP用户", "=" * 30)
config = {"configurable": {"thread_id": "demo-7.3"}}
res = graph.invoke(
    {"user_input": "你好，帮我查一下最近有什么优惠活动"},
    config=config,
    context=UserContext(username="Alice", membership_level="VIP")
)

print(f"output: {res['output']}")
print()

# === 第二次调用：不传运行时上下文 ===
print("=" * 30, "第二次调用：不传上下文", "=" * 30)
res2 = graph.invoke(
    {"user_input": "再帮我看看有没有新品"},
    config=config
)

print(f"output: {res2['output']}")

from IPython.display import display
display(graph)
```

**运行结果如下**

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	llm_node(llm_node)
	__end__([<p>__end__</p>]):::last
	__start__ --> llm_node;
	llm_node --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```

```json
2026-07-09 17:40:27.208 | INFO     | __main__:llm_node:39 - 当前用户: Alice, 会员等级: VIP
============================== 第一次调用：VIP用户 ==============================
2026-07-09 17:40:29.116 | WARNING  | __main__:llm_node:46 - 运行时上下文为空，使用默认风格
output: 尊敬的Alice您好！非常感谢您的咨询🎉 我们近期为VIP用户准备了多项专属优惠活动，包括但不限于：1）满额立减（消费满888元立减88元）；2）积分翻倍（指定商品享3倍积分）；3）限时闪购（每日10:00-12:00精选好物低至5折）。如需了解具体参与商品或为您定制个性化推荐，请随时告知，我将全力为您服务！🎖️VIP专属服务

============================== 第二次调用：不传上下文 ==============================
output: 您好！目前新品信息尚未更新，建议您关注官方渠道获取最新动态。如有其他问题，欢迎随时咨询，我会尽力为您解答！😊
```

**分析**

1. **第一次调用**传入了 **`context=UserContext(username=”Alice”, membership_level=”VIP”)`**，节点通过 **`runtime.context`** 读取到用户名和会员等级，生成了VIP风格的回复，末尾带上了 **`🎖️VIP专属服务`** 标识。

2. **第二次调用**使用了相同的 **`thread_id`**，但没有传入 **`context`**。此时 **`runtime.context`** 为 **`None`**，节点走入了默认风格分支，回复中不再有VIP标识。

这个对比可以清晰地看到：

> **运行时上下文仅对当次调用生效，不会被持久化。即使使用相同的 `thread_id`，下一次调用也不会自动继承上一次的上下文。**

### 7.3.2. 小结

运行时上下文适合传递以下信息：

| 适合放入运行时上下文         | 不适合（应放入图状态）           |
| ---------------------------- | -------------------------------- |
| 当前登录用户信息             | 需要在多轮对话间共享的数据       |
| 请求来源（Web / API / 小程序） | 需要在检查点(checkpoint)中恢复的执行进度 |
| 调用方标识、Trace ID         | 需要跨调用持久化的业务数据       |
| 本次调用的功能开关           | 节点间需要传递的计算结果         |

简单来说：

> **跨调用共享、需要持久化的数据 → 放入 State；仅当次调用有效的信息 → 放入 Runtime Context。**

## 7.4. Node总结：LangGraph的节点

### 7.4.1. 节点的本质

从用户视角看，**`LangGraph`** 的节点通常是一个可调用对象，最常见的是同步或异步 **`Python`** 函数：

```python
def sync_node(state: State) -> State:
    ...

async def async_node(state: State) -> State:
    ...
```

除此之外，节点也可以是符合要求的其他可调用对象或 **`Runnable`** 实例。

从源码实现角度看，节点最终会被转换或包装为可运行对象，并作为 **`PregelNode.bound`** 保存。

### 7.4.2. 节点函数的完整形态

状态图节点的第一个位置参数用于接收图状态。

此外，**`LangGraph`** 还可以在运行时注入**运行时配置**、**运行时对象**和**流式写入器**等参数。

普通图节点通常使用以下四个参数：

- **`state`**：输入节点的图状态。**`state`** 是位置传参，因此参数名称不重要，但通常约定命名为 **`state`**

- **`config`**：状态图的运行时配置，它是一个 **`RunnableConfig`** 实例，底层运行时在节点函数执行前通过关键字传参动态注入。

  可以通过 **`config`** 访问当前线程的 **`thread_id`**、超步序号、递归限制等配置信息。如下

  ```python
  config["configurable"]["thread_id"] 
  config["recursion_limit"]
  config["metadata"]
  ```

- **`runtime`**：状态图的运行时对象，可以通过它访问运行时上下文、长期记忆存储器等信息。底层运行时在节点函数执行前通过关键字传参动态注入。

- **`writer`**：流式写入器，通常用于自定义流式输出。详见流处理章节。

典型写法如下： 

```python
def node(state: State, config: RunnableConfig, runtime: Runtime[Context]) -> State:
    ...
```

* 编译图时传入 **`checkpointer`**，并在调用时传入 **`config`**，可以在节点中通过 **`config`** 间接访问当前调用的配置信息，如 **`thread_id`**。
* 编译图时传入 **`store`** 后，可以在节点中通过 **`runtime.store`** 访问长期记忆存储器。
* 初始化状态图时传入 **`context_schema`**，并在调用时传入 **`context`** 后，可以在节点中通过 **`runtime.context`** 访问运行时上下文。

### 7.4.3. 节点的触发和执行

#### 7.4.3.1. 节点的触发

##### 7.4.3.1.1. 用户视角

从用户层面看，节点由图中的控制流关系（普通边、条件边和动态控制指令）触发：

* 普通边决定固定的后继节点。
* 条件边根据路由函数的返回值决定后继节点。
  * 可以返回普通节点名称
  * 也可以返回 **`Send`** 实例，从而动态派发多个并行任务，常用于 **`Map-Reduce`** 场景。

* **`Command.goto`** 可以在节点返回时动态指定后继节点。

##### 7.4.3.1.2. 源码视角

从源码实现角度看，**`LangGraph`** 会将图状态和节点之间的触发关系组织为通道。

其中，状态字段通常对应状态通道，节点之间的控制流关系则通过触发通道或屏障通道表示。对于普通的单节点触发关系，目标节点通常会订阅如下形式的内部通道：

```text
branch:to:<node_name>
```

例如，对于普通边：

```python
builder.add_edge("node_a", "node_b")
```

运行时 **`node_a`** 的写入器会向以下通道写入数据：

```text
branch:to:node_b
```

而 **`node_b`** 会将该通道注册为自己的触发通道。

条件边和 **`Command.goto`** 最终也会根据目标节点生成相应的触发通道写入。**`Send`** 则会生成动态派发任务所需的 **`Send`** 数据包。

对于多个前驱节点共同汇聚到同一个节点的情况，底层还可能使用类似下面的屏障通道：

```text
join:<node_a>+<node_b>:<node_c>
```

用于等待指定的前驱节点全部完成。

当某个节点订阅的触发通道在当前超步中产生新的写入后，该节点会在下一超步被调度。

#### 7.4.3.2. 节点的执行

##### 7.4.3.2.1. 用户视角

从用户层面看，节点执行过程可以理解为：

1. **`LangGraph`** 从当前图状态中读取节点需要的字段，并构造节点输入。

2. 调用节点函数，执行其业务逻辑。

3. 节点函数返回状态更新、**`Command`** 或其他受支持的结果。

4. **`LangGraph`** 将节点返回值转换为状态通道和控制通道的写入条目。

5. 如果当前节点挂载了条件边，则执行条件边路由逻辑，根据包含当前节点更新的状态决定后继节点。

6. 当前超步的所有任务执行完毕后，**`LangGraph`** 汇总这些写入，并根据各状态字段的 **`Reducer`** 规则更新图状态。

节点通常只需要返回发生变化的字段，而不需要返回完整状态。例如：

```python
def node_a(state: State) -> State:
    return {"count": state["count"] + 1}
```

其中：

```python
{"count": state["count"] + 1}
```

表示对状态的更新，而不是一份必须包含所有字段的完整状态。

##### 7.4.3.2.2. 源码视角

从源码实现角度看，节点业务逻辑、状态写入逻辑和控制流写入逻辑会被组合成一个可顺序执行的可运行对象。

编译节点时，**`LangGraph`** 会创建一个 **`PregelNode`**。

相关源码位于 **`CompiledStateGraph`** 类的 **`attach_node()`** 方法中，如下

```python
self.nodes[key] = PregelNode(
    triggers=[branch_channel],
    # read state keys and managed values
    channels=("__root__" if is_single_input else input_channels),
    # coerce state dict to schema class (eg. pydantic model)
    mapper=mapper,
    # publish to state keys
    writers=[ChannelWrite(write_entries)],
    metadata=node.metadata,
    retry_policy=node.retry_policy,
    cache_policy=node.cache_policy,
    bound=node.runnable,  # type: ignore[arg-type]
)
```

其中：

* **`bound`** 保存节点的业务逻辑；
* **`writers`** 保存节点执行完成后需要调用的写入器；
* **`triggers`** 保存能够触发该节点的通道。

可以近似表示为：

```text
PregelNode
├── bound：节点业务逻辑
├── writers：状态和控制流写入器
└── triggers：能够触发该节点的通道
```

因此，从整体上看，节点执行并不是简单地”调用一个函数并返回结果”，而是：

> 执行业务逻辑，生成状态和控制流写入，并通过目标节点触发通道或动态任务写入，为下一超步生成待调度任务，持续推动计算图运行。


---

# 附录A. PostgreSQL的部署与配置

## A.1. PostgreSQL简介

### A.1.1. 什么是PostgreSQL

**`PostgreSQL`**（简称 **`PG`**）是一款开源的**关系型数据库管理系统（`RDBMS`）**，诞生于1996年，至今已有近30年的发展历史。它以**稳定性、可靠性、`ACID`事务支持**和**强大的扩展能力**著称，是目前生产环境中使用最广泛的开源关系型数据库之一。

**`PostgreSQL`** 的主要特点包括：

* **`ACID`事务**：完整支持原子性（`Atomicity`）、一致性（`Consistency`）、隔离性（`Isolation`）、持久性（`Durability`），确保数据操作的可靠性；
* **丰富的数据类型**：不仅支持传统的整数、浮点数、字符串，还原生支持 **`JSON`/`JSONB`**、数组、范围类型等，使其可以同时承担”关系型”和”文档型”数据库的角色；
* **可扩展性**：支持自定义数据类型、函数、索引类型，拥有丰富的扩展生态；
* **跨平台**：支持 **`Windows`**、**`Linux`**、**`macOS`** 等主流操作系统。

在 **`LangGraph`** 的上下文中，**`PostgreSQL`** 主要被用作**检查点(checkpoint)（`Checkpoint`）的持久化后端**，同时也支持作为**长期记忆存储**使用。

### A.1.2. 为什么LangGraph需要PostgreSQL

在之前的 **6.2.4节** 中，我们提到”在学习 **`LangChain`** 时，我们已经介绍了 **`PostgreSQL`** 的部署和 **`PostgresSaver`** 的用法”。但如果你没有学习过 **`LangChain`** 课程，或者希望在本课程中获得完整的知识闭环，本附录将从零开始介绍 **`PostgreSQL`** 的部署与使用。

回顾一下第 **6.2.2节** 中检查点(checkpoint)后端的对比：

| 后端                    | 适用场景                               |
| ----------------------- | -------------------------------------- |
| **`InMemorySaver`**     | 学习、调试、本地实验；进程结束数据丢失 |
| **`SQLite`**            | 轻量级本地持久化；单进程场景           |
| **`PostgreSQL`**        | 生产环境；跨进程、跨服务共享状态       |
| **`MongoDB`**           | 生产环境；偏好文档模型                 |
| **`Redis`**             | 生产环境；高吞吐、低延迟缓存场景       |

**`InMemorySaver`** 将检查点(checkpoint)保存在当前 **`Python`** 进程的内存中。每次重启程序，内存中的数据就会丢失——这对于需要跨多次调用记住对话历史的多轮对话场景显然是不够的。

**`PostgresSaver`** 则将检查点(checkpoint)写入 **`PostgreSQL`** 数据库。与程序进程的生命周期解耦：
* 多次运行同一个 **`thread_id`** 的对话，历史消息会持续累积；
* 重启 **`Python`** 程序后，只要数据库中的记录还在，历史检查点(checkpoint)就能被恢复；
* 多个 **`Python`** 进程可以共享同一个 **`PostgreSQL`** 实例，实现跨进程的状态共享。

**`PostgresSaver`** 来自 **`langgraph-checkpoint-postgres`** 包，本节附录将完整介绍如何部署和配置它所需的 **`PostgreSQL`** 环境。

### A.1.3. 本文使用的连接信息

为与 **6.2.4节** 的代码保持一致，本附录统一使用以下连接信息：

| 配置项          | 值                                                                                                                                 |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **`Host`**      | **`localhost`**                                                                                                                    |
| **`Port`**      | **`5432`**（**`PostgreSQL`** 默认端口）                                                                                           |
| **`Database`**  | **`langgraph_db`**                                                                                                                 |
| **`User`**      | **`langgraph_user`**                                                                                                               |
| **`Password`**  | **`123456`**                                                                                                                       |
| **`SSL Mode`**  | **`disable`**（本地开发环境关闭 SSL）                                                                                              |
| **连接URL**     | **`postgresql://langgraph_user:123456@localhost:5432/langgraph_db?sslmode=disable`**                                               |

> **注意**：这里的 **`sslmode=disable`** 表示关闭 **`SSL`** 连接，仅适用于本地开发和测试环境。生产环境中建议启用 **`SSL`** 并修改为强密码。


## A.2. PostgreSQL的安装与部署

本节介绍两种安装方式：**`Docker`** 部署（推荐，简单快速）和 **`Windows`** 直接安装（无需额外安装 **`Docker`**）。

### A.2.1. 方式一：使用Docker部署（推荐）

**`Docker`** 部署的优势在于：无需修改系统环境，安装卸载干净，版本切换方便，一条命令即可启动。

> 如果你还没有安装 **`Docker`**，请先访问 **[Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)** 下载并安装。安装完成后确保 **`Docker Engine`** 处于运行状态（桌面右下角 **`Docker`** 图标为绿色）。

#### A.2.1.1. 拉取PostgreSQL镜像

打开 **`PowerShell`** 或 **`CMD`** 终端，执行：

```bash
docker pull postgres:16
```

这里选择 **`PostgreSQL 16`** 版本，是当前最新的稳定大版本，与 **`langgraph-checkpoint-postgres==3.0.5`** 兼容。

拉取完成后，可通过以下命令查看本地已有的镜像：

```bash
docker images | findstr postgres
```

#### A.2.1.2. 运行PostgreSQL容器

执行以下命令启动容器：

```bash
docker run -d `
  --name langgraph-postgres `
  -e POSTGRES_DB=langgraph_db `
  -e POSTGRES_USER=langgraph_user `
  -e POSTGRES_PASSWORD=123456 `
  -p 5432:5432 `
  postgres:16
```

> **Windows `CMD` 用户注意**：上述换行使用了 **`PowerShell`** 的反引号 `` ` ``。如果在 **`CMD`** 中执行，需要将 `` ` `` 替换为 `^`，或写成一整行：
> ```bash
> docker run -d --name langgraph-postgres -e POSTGRES_DB=langgraph_db -e POSTGRES_USER=langgraph_user -e POSTGRES_PASSWORD=123456 -p 5432:5432 postgres:16
> ```

各参数含义：

| 参数 | 含义 |
| ---- | ---- |
| **`-d`** | 后台运行（`daemon` 模式），终端关闭后容器不会停止 |
| **`--name langgraph-postgres`** | 容器名称，方便后续管理 |
| **`-e POSTGRES_DB=langgraph_db`** | 创建容器时自动创建名为 **`langgraph_db`** 的数据库 |
| **`-e POSTGRES_USER=langgraph_user`** | 自动创建用户 **`langgraph_user`** |
| **`-e POSTGRES_PASSWORD=123456`** | 设置该用户的密码为 **`123456`** |
| **`-p 5432:5432`** | 将容器的 5432 端口映射到宿主机的 5432 端口 |

> **注意**：通过环境变量自动创建的数据库和用户，其权限已自动配置好，无需再手动授权。如果使用手动安装方式，则需要额外执行授权步骤（见 **A.3.2节**）。

#### A.2.1.3. 验证容器运行状态

```bash
# 查看正在运行的容器
docker ps

# 如果看不到 langgraph-postgres，查看所有容器（包括已停止的）
docker ps -a

# 查看容器日志
docker logs langgraph-postgres
```

如果容器没有运行，可以通过以下命令启动：

```bash
docker start langgraph-postgres
```

> **常用容器管理命令**：
> ```bash
> docker stop langgraph-postgres    # 停止容器
> docker start langgraph-postgres   # 启动容器
> docker restart langgraph-postgres # 重启容器
> docker rm langgraph-postgres      # 删除容器（需要先停止）
> ```

#### A.2.1.4. Docker Compose方式（可选）

如果更习惯使用 **`Docker Compose`** 管理容器，可以创建 `docker-compose.yml` 文件：

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:16
    container_name: langgraph-postgres
    environment:
      POSTGRES_DB: langgraph_db
      POSTGRES_USER: langgraph_user
      POSTGRES_PASSWORD: 123456
    ports:
      - “5432:5432”
    volumes:
      - pgdata:/var/lib/postgresql/data  # 持久化数据，容器删除后数据不丢失

volumes:
  pgdata:
```

然后在 `docker-compose.yml` 所在目录执行：

```bash
docker compose up -d
```

**`volumes`** 配置将数据库文件映射到宿主机，**即使容器被删除，数据也不会丢失**——这对于保存检查点(checkpoint)数据非常重要。


### A.2.2. 方式二：Windows直接安装

如果你不想安装 **`Docker`**，也可以直接在 **`Windows`** 上安装 **`PostgreSQL`**。

#### A.2.2.1. 下载安装包

访问 **[PostgreSQL Windows 官方下载页](https://www.postgresql.org/download/windows/)**，或直接访问 **[EDB安装包下载页](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads)**，选择 **`PostgreSQL 16.x`** 版本、**`Windows x86-64`** 架构的安装包下载。

> 建议下载带有 **`pgAdmin`**（图形化管理工具）的完整安装包。

#### A.2.2.2. 安装步骤

1. **运行安装程序**：双击下载的 `.exe` 文件，点击 **`Next`**；
2. **选择安装目录**：建议保持默认路径 `C:\Program Files\PostgreSQL\16\`；
3. **选择安装组件**：勾选 **`PostgreSQL Server`**、**`pgAdmin 4`**（可选）、**`Command Line Tools`**（必选，需要 **`psql`** 命令行工具）；
4. **设置超级用户密码**：安装程序会要求为 **`postgres`** 超级用户设置密码（记好这个密码）；
5. **设置端口**：保持默认 **`5432`**；
6. **选择区域设置**：保持默认 **`Default locale`**；
7. **安装**：等待进度条完成。

#### A.2.2.3. 验证安装

安装完成后，打开 **`CMD`** 或 **`PowerShell`**：

```bash
# 检查 psql 是否可用
psql --version

# 使用超级用户连接数据库
psql -h localhost -p 5432 -U postgres
```

输入安装时设置的 **`postgres`** 密码，如果能进入 **`psql`** 交互界面（显示 `postgres=#` 提示符），说明安装成功。

#### A.2.2.4. Windows服务管理

**`PostgreSQL`** 安装后会自动注册为 **`Windows`** 服务（服务名：**`postgresql-x64-16`**），默认开机自启。

常用服务管理操作：

```bash
# 查看服务状态（PowerShell）
Get-Service postgresql-x64-16

# 手动启动服务
net start postgresql-x64-16

# 手动停止服务
net stop postgresql-x64-16
```

也可以通过 **`services.msc`**（服务管理器）图形界面管理。


### A.2.3. 安装Python依赖

无论选择哪一种安装方式，都需要安装 **`Python`** 侧的依赖包：

```bash
pip install langgraph-checkpoint-postgres==3.0.5
pip install psycopg[binary]
```

| 包名 | 作用 |
| ---- | ---- |
| **`langgraph-checkpoint-postgres`** | **`LangGraph`** 的 **`PostgreSQL`** 检查点(checkpoint)后端，提供 **`PostgresSaver`** |
| **`psycopg[binary]`** | **`Python`** 的 **`PostgreSQL`** 驱动，负责底层数据库通信 |

> **`langgraph-checkpoint-postgres`** 内部依赖 **`psycopg`**，但建议显式安装以确保版本正确。**`[binary]`** 后缀表示安装包含预编译二进制文件的版本，避免从源码编译的麻烦。


## A.3. PostgreSQL基本操作

本节介绍 **`PostgreSQL`** 的常用操作。掌握这些内容有助于理解 **`PostgresSaver`** 在数据库中做了什么，也方便日常排查问题。

### A.3.1. 连接PostgreSQL

#### A.3.1.1. 使用psql命令行工具

**`psql`** 是 **`PostgreSQL`** 自带的命令行客户端。

连接命令格式：

```bash
psql -h <主机> -p <端口> -U <用户名> -d <数据库名>
```

对应本附录的连接信息：

```bash
psql -h localhost -p 5432 -U langgraph_user -d langgraph_db
# 输入密码: 123456
```

连接成功后，终端提示符变为 `langgraph_db=>`，可以输入 **`SQL`** 命令。

> **Docker 用户**：如果主机上没有安装 **`psql`**，可以直接进入容器内执行：
> ```bash
> docker exec -it langgraph-postgres psql -U langgraph_user -d langgraph_db
> ```

常用 **`psql`** 元命令：

| 命令 | 说明 |
| ---- | ---- |
| **`\l`** | 列出所有数据库 |
| **`\dt`** | 列出当前数据库的所有表 |
| **`\d table_name`** | 查看表结构 |
| **`\q`** | 退出 **`psql`** |

#### A.3.1.2. 使用Python连接

在 **`Python`** 中使用 **`psycopg`** 连接：

```python
import psycopg

DB_URL = “postgresql://langgraph_user:123456@localhost:5432/langgraph_db?sslmode=disable”

conn = psycopg.connect(DB_URL)
cur = conn.cursor()

# 查看 PostgreSQL 版本信息
cur.execute(“SELECT version();”)
result = cur.fetchone()
print(f”PostgreSQL 版本：{result[0]}”)

cur.close()
conn.close()
```

**运行结果如下**

```
PostgreSQL 版本：PostgreSQL 16.x on x86_64-pc-linux-gnu, compiled by gcc ...
```

> **注意**：如果在 `pip install psycopg` 时没有加 `[binary]` 后缀，可能会因为缺少编译环境而报错。遇到这种情况执行 `pip install psycopg[binary]` 即可。


### A.3.2. 创建数据库和用户

> **Docker 用户请注意**：通过 **`docker run`** 的 **`-e`** 环境变量启动容器时，数据库和用户已经自动创建并配置好权限，**可以跳过本节**，直接阅读 **A.3.3节**。

如果你是**手动安装**的，启动时**不会**自动创建 `langgraph_db` 数据库和 `langgraph_user` 用户，需要手动创建。

#### A.3.2.1. 创建数据库

首先用超级用户 **`postgres`** 连接：

```bash
psql -h localhost -p 5432 -U postgres
```

然后创建数据库：

```sql
CREATE DATABASE langgraph_db;
```

查看是否创建成功：

```sql
\l
```

输出中如果能找到 **`langgraph_db`** 条目，则说明创建成功。

#### A.3.2.2. 创建用户并授权

在同一个 **`psql`** 会话中继续执行：

```sql
-- 创建用户
CREATE USER langgraph_user WITH PASSWORD '123456';

-- 赋予该用户对 langgraph_db 的所有权限
GRANT ALL PRIVILEGES ON DATABASE langgraph_db TO langgraph_user;
```

> **`PostgreSQL 15+`** 版本默认收紧了对 **`public`** 模式的权限管理。如果后续 **`PostgresSaver`** 在创建表时报权限错误，需要额外执行：
> ```sql
> \c langgraph_db
> GRANT ALL ON SCHEMA public TO langgraph_user;
> ```

退出 **`psql`**，用新用户测试连接：

```bash
psql -h localhost -p 5432 -U langgraph_user -d langgraph_db
# 输入密码: 123456
```

如果能成功连接，说明用户和数据库配置正确。


### A.3.3. 基本CRUD操作

了解基本的 **`SQL`** 增删改查有助于查看和理解 **`PostgresSaver`** 内部的数据存储。

#### A.3.3.1. 创建表

```sql
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

* **`SERIAL PRIMARY KEY`**：自增主键，**`PostgreSQL`** 会自动为新插入的行分配唯一的 **`id`**；
* **`TEXT`**：变长文本类型，适合存储聊天消息内容；
* **`TIMESTAMP DEFAULT CURRENT_TIMESTAMP`**：时间戳类型，默认为插入时的当前时间。

#### A.3.3.2. 插入数据

```sql
INSERT INTO messages (thread_id, role, content)
VALUES ('chapter_6_6.2.4', 'user', '你好，我是老王');

INSERT INTO messages (thread_id, role, content)
VALUES ('chapter_6_6.2.4', 'assistant', '老王你好！有什么可以帮你的？');
```

#### A.3.3.3. 查询数据

```sql
-- 查询所有记录
SELECT * FROM messages;

-- 按 thread_id 筛选（最常用，对应 LangGraph 中的 thread）
SELECT * FROM messages WHERE thread_id = 'chapter_6_6.2.4';

-- 按时间排序
SELECT * FROM messages WHERE thread_id = 'chapter_6_6.2.4' ORDER BY created_at;
```

#### A.3.3.4. 更新和删除数据

```sql
-- 更新某条记录的内容
UPDATE messages SET content = '你好，我是老王，今年30岁。'
WHERE id = 1;

-- 删除某条记录
DELETE FROM messages WHERE id = 2;

-- 删除整个表（注意：操作不可逆）
DROP TABLE IF EXISTS messages;
```


### A.3.4. 查看PostgresSaver创建的表结构

这是整个附录中最实用的部分——**连接回 6.2.4节**，看看 **`checkpointer.setup()`** 在 **`PostgreSQL`** 中到底创建了什么。

#### A.3.4.1. 运行setup()后查看表

首先运行一次 **6.2.4节** 的代码（确保 **`setup()`** 已执行），然后在 **`psql`** 中查看：

```sql
\c langgraph_db
\dt
```

**运行结果如下**

```
             List of relations
 Schema |        Name           | Type  |     Owner
--------+-----------------------+-------+----------------
 public | checkpoint_blobs      | 数据表 | langgraph_user
 public | checkpoint_migrations | 数据表 | langgraph_user
 public | checkpoint_writes     | 数据表 | langgraph_user
 public | checkpoints           | 数据表 | langgraph_user
```

可以看到 **`setup()`** 在数据库中创建了四张表：

| 表名 | 用途 |
| ---- | ---- |
| **`checkpoints`** | 存储完整的检查点(checkpoint)快照（状态、元数据等） |
| **`checkpoint_writes`** | 存储每个检查点(checkpoint)对应的通道写入记录 |
| **`checkpoint_blobs`** | 存储大型二进制数据（如图片、文件等，如果状态中包含的话） |
| **`checkpoint_migrations`** | 记录数据库迁移版本，**`LangGraph`** 内部使用，用于管理表结构变更 |

#### A.3.4.2. 检查点(checkpoint)表结构说明

使用 **`\d`** 命令查看每张表的实际结构。

**（1）checkpoints 表**

```sql
\d checkpoints
```

```
         栏位         | 类型  |  可空的  |    预设
----------------------+-------+----------+-------------
 thread_id            | text  | not null |
 checkpoint_ns        | text  | not null | ''::text
 checkpoint_id        | text  | not null |
 parent_checkpoint_id | text  |          |
 type                 | text  |          |
 checkpoint           | jsonb | not null |
 metadata             | jsonb | not null | '{}'::jsonb
索引：
    "checkpoints_pkey" PRIMARY KEY, btree (thread_id, checkpoint_ns, checkpoint_id)
    "checkpoints_thread_id_idx" btree (thread_id)
```

**关键字段说明**：

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| **`thread_id`** | **`text`** | 线程ID，对应 **`config`** 中的 **`thread_id`**，用于区分不同对话 |
| **`checkpoint_ns`** | **`text`** | 检查点(checkpoint)命名空间，默认空字符串，用于子图等场景隔离不同层级的检查点(checkpoint) |
| **`checkpoint_id`** | **`text`** | 检查点(checkpoint)唯一ID，由 **`LangGraph`** 内部生成 |
| **`parent_checkpoint_id`** | **`text`** | 父检查点(checkpoint)ID，构建检查点(checkpoint)链，用于 **`Time Travel`** 回退 |
| **`type`** | **`text`** | 检查点(checkpoint)类型 |
| **`checkpoint`** | **`jsonb`** | 检查点(checkpoint)核心数据（状态快照），以 **`JSON`** 格式存储 |
| **`metadata`** | **`jsonb`** | 检查点(checkpoint)元数据（创建时间、来源、步骤序号等），预设为 `'{}'::jsonb` |

* 主键为 **(thread_id, checkpoint_ns, checkpoint_id)** 联合主键。
* 另有 **`thread_id`** 单列索引，加速按线程筛选查询。

**（2）checkpoint_writes 表**

```sql
\d checkpoint_writes
```

```
     栏位      |  类型   |  可空的  |   预设
---------------+---------+----------+----------
 thread_id     | text    | not null |
 checkpoint_ns | text    | not null | ''::text
 checkpoint_id | text    | not null |
 task_id       | text    | not null |
 idx           | integer | not null |
 channel       | text    | not null |
 type          | text    |          |
 blob          | bytea   | not null |
 task_path     | text    | not null | ''::text
索引：
    "checkpoint_writes_pkey" PRIMARY KEY, btree (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
    "checkpoint_writes_thread_id_idx" btree (thread_id)
```

**关键字段说明**：

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| **`thread_id`** | **`text`** | 线程ID |
| **`checkpoint_ns`** | **`text`** | 命名空间，与 **`checkpoints`** 表对应 |
| **`checkpoint_id`** | **`text`** | 所属检查点(checkpoint)ID |
| **`task_id`** | **`text`** | 产生写入的节点任务ID |
| **`idx`** | **`integer`** | 写入顺序索引 |
| **`channel`** | **`text`** | 写入的目标通道名（如 **`messages`**、**`__pregel_tasks`**） |
| **`type`** | **`text`** | 写入数据类型 |
| **`blob`** | **`bytea`** | 写入的二进制数据 |
| **`task_path`** | **`text`** | 任务路径，用于追踪节点在子图中的层级关系 |

* 主键为 **(thread_id, checkpoint_ns, checkpoint_id, task_id, idx)** 联合主键，确保同一任务下的写入顺序唯一。

**（3）checkpoint_blobs 表**

```sql
\d checkpoint_blobs
```

```
     栏位      | 类型  |  可空的  |   预设
---------------+-------+----------+----------
 thread_id     | text  | not null |
 checkpoint_ns | text  | not null | ''::text
 channel       | text  | not null |
 version       | text  | not null |
 type          | text  | not null |
 blob          | bytea |          |
索引：
    "checkpoint_blobs_pkey" PRIMARY KEY, btree (thread_id, checkpoint_ns, channel, version)
    "checkpoint_blobs_thread_id_idx" btree (thread_id)
```

**关键字段说明**：

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| **`thread_id`** | **`text`** | 线程ID |
| **`checkpoint_ns`** | **`text`** | 命名空间 |
| **`channel`** | **`text`** | 通道名 |
| **`version`** | **`text`** | 数据版本标识，同一通道可以有多个版本 |
| **`type`** | **`text`** | 数据类型 |
| **`blob`** | **`bytea`** | 二进制数据本体 |

* 主键为 **(thread_id, checkpoint_ns, channel, version)** 联合主键。
* 与 **`checkpoint_writes`** 不同，**`checkpoint_blobs`** 以 **(channel, version)** 维度管理数据版本，不直接关联某个具体的 **`checkpoint_id`**，允许多个检查点(checkpoint)共享同一通道的大型二进制数据。

**（4）checkpoint_migrations 表**

```sql
\d checkpoint_migrations
```

```
 栏位 |  类型   |  可空的  | 预设
------+---------+----------+------
 v    | integer | not null |
索引：
    "checkpoint_migrations_pkey" PRIMARY KEY, btree (v)
```

**关键字段说明**：

| 字段 | 类型 | 说明 |
| ---- | ---- | ---- |
| **`v`** | **`integer`** | 当前数据库迁移版本号 |

* 单列主键，仅一条记录，**`setup()`** 执行时自动检查并更新。
* 该表由 **`LangGraph`** 内部管理，用户无需手动操作。未来 **`LangGraph`** 版本升级时，若表结构有变更，**`setup()`** 会根据此版本号自动执行对应的迁移 **`SQL`**。

> **四张表的关系**：
> * **`checkpoints`** 存储每次 **`graph.invoke()`** 产生的检查点(checkpoint)快照；
> * **`checkpoint_writes`** 存储每个检查点(checkpoint)下各节点任务的通道写入记录（一条检查点(checkpoint)通常对应多条写入）；
> * **`checkpoint_blobs`** 存储通道级别的大型二进制数据，以 **(thread_id, checkpoint_ns, channel, version)** 为维度，可跨检查点(checkpoint)复用；
> * **`checkpoint_migrations`** 仅用于数据库表结构版本管理。
>
> **`LangGraph`** 通过 **`thread_id`** + **`checkpoint_ns`** + **`checkpoint_id`** 联合定位和恢复状态。

#### A.3.4.3. 写入检查点(checkpoint)数据查询

运行 **6.2.4节** 的示例代码后，可以直接在数据库中查看写入的数据：

```sql
SET client_encoding = 'UTF8';
-- 查看指定线程的检查点(checkpoint)数量
SELECT thread_id, count(*)
FROM checkpoints
WHERE thread_id = 'chapter03-02'
GROUP BY thread_id;

-- 查看所有检查点(checkpoint)的基本信息（按创建时间排序）
SELECT checkpoint_id, parent_checkpoint_id, checkpoint_ns, metadata
FROM checkpoints
WHERE thread_id = 'chapter03-02'
ORDER BY metadata;

-- 查看最新检查点(checkpoint)的消息通道写入数据
SELECT c.checkpoint_id, c.checkpoint_ns, cw.channel, cw.task_id, cw.type
FROM checkpoints c
JOIN checkpoint_writes cw
  ON c.thread_id = cw.thread_id
 AND c.checkpoint_ns = cw.checkpoint_ns
 AND c.checkpoint_id = cw.checkpoint_id
WHERE c.thread_id = 'chapter03-02'
  AND cw.channel = 'messages'
ORDER BY cw.idx;

-- 查看最新检查点(checkpoint)的完整状态 JSON
SELECT checkpoint_id, checkpoint
FROM checkpoints
WHERE thread_id = 'chapter03-02'
ORDER BY metadata DESC
LIMIT 1;

```

通过这些查询，可以直观地理解 **`PostgresSaver`** 是如何将 **`LangGraph`** 的检查点(checkpoint)数据持久化到 **`PostgreSQL`** 中的。

> **Windows 终端编码问题**：
> 在中文 **`Windows`** 的 **`CMD`** 或 **`PowerShell`** 中使用 **`psql`** 查询 **`checkpoint`**（**`jsonb`**）字段时，如果数据中包含 emoji 等 4 字节 **`UTF-8`** 字符，可能会报错：
>
> ```
> 错误:  编码"UTF8"的字符0x0xf0 0x9f 0xa4 0x94在编码"GBK"没有相对应值
> ```
> 这是因为 **`Windows`** 终端默认编码为 **`GBK`**，无法处理 4 字节 **`UTF-8`** 字符。解决方案：
> ```sql
> -- 查询前先切换客户端编码
> SET client_encoding = 'UTF8';
> ```
> 或在 **`PowerShell`** 中先执行 **`chcp 65001`** 切换到 **`UTF-8`** 编码，再启动 **`psql`**。

#### 补充说明

* **`setup()`** 是幂等的——多次调用不会重复创建表，也不会清空已有数据；
* 实际项目中，建议将 **`setup()`** 作为独立的数据库初始化/迁移步骤执行，而不是每次启动应用时都调用（这也是 **6.2.4节** 代码注释中提到的建议）；
* 如果需要主动清理某个线程的检查点(checkpoint)数据，可以使用 **`checkpointer.delete_thread(thread_id)`** 方法，或者从数据库层面删除对应记录；
* 关于连接池：如果需要在生产环境中处理高并发请求，建议使用 **`psycopg_pool`** 提供的连接池功能，而不是每次创建新连接。

