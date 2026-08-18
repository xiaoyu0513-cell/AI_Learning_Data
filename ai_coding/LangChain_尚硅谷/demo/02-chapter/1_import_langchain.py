import importlib.metadata

# 需要检查版本的 LangChain 模块列表
packages = [
    "langchain",
    "langchain-core",
    "langchain-community",
    "langchain-text-splitters",
    "langchain-experimental",
]

print("=== langchain_env 环境版本检测 ===")
for pkg in packages:
    try:
        # 使用官方标准库获取安装的版本
        version = importlib.metadata.version(pkg)
        print(f"{pkg}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg}: ❌ 未安装")
        
