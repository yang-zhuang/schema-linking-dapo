"""
Configuration constants for evaluation system
"""

# vLLM service configuration
DEFAULT_API_URL = "http://localhost:8000/v1"
DEFAULT_API_KEY = "token-abc123"  # vLLM无需验证，仅满足SDK格式要求
DEFAULT_MODEL_NAME = "dapo-Qwen3-0.6B" # "Qwen3-0.6B"、"dapo-Qwen3-0.6B"

# Data path configuration
DEFAULT_INPUT_CSV_PATH = "../data/val_prompt2.csv"
DEFAULT_SAVE_PATH = f"../data/{DEFAULT_MODEL_NAME}/model_process_results.csv"
DEFAULT_METRICS_PATH = f"../data/{DEFAULT_MODEL_NAME}/eval_metrics.json"

# Concurrency configuration
DEFAULT_MAX_WORKERS = 10  # 并发工作线程数，根据服务器性能调整
DEFAULT_REQUEST_TIMEOUT = 60  # 单个请求超时时间（秒）

# System prompt
DEFAULT_SYSTEM_PROMPT = """你是一个模式链接助手。你的任务是根据<question>...</question>中的用户问题，从<database>...</database>的数据库中选取合适的表名及对应的列名。
请严格按照以下 JSON 格式输出结果：
```json
{
  "schema": [
    {
      "table_name": "表名1",
      "columns": ["列1", "列2", "列3"]
    },
    ...
  ]
}
```
如果没有匹配的表或列，请返回 {"schema": []}。不要包含任何额外字段、解释或文本。
"""