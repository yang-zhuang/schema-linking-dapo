import re
import json
import random
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from modelscope import AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 数据路径配置
INPUT_CSV_PATH = "./dataset/training_prompt2.csv"
OUTPUT_JSONL_PATH = "./dataset/train.jsonl"

# 并发配置
MAX_WORKERS = 5  # 并发工作线程数，根据服务器性能调整
REQUEST_TIMEOUT = 60  # 单个请求超时时间（秒）

# 系统提示词（业务规则集中存放）
SYSTEM_PROMPT = """你是一个模式链接助手。你的任务是根据<question>...</question>中的用户问题，从<database>...</database>的数据库中选取合适的表名及对应的列名。  
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


# ========================== 工具函数（提取重复逻辑，单一职责）==========================
def _extract_tables_and_columns(standard_str: str) -> tuple[set, set]:
    """
    通用工具函数：从标准化字符串中提取表名和列名（小写化，避免大小写误差）
    :param standard_str: 格式如 "###Tables: 表1,表2;\n###Columns: 表1.列1,表2.列2;"
    :return: (表名集合, 列名集合)
    """
    # 提取表名
    tables_match = re.search(r'Tables:\s*(.*?);', standard_str)
    tables = set(tables_match.group(1).split(', ')) if (tables_match and tables_match.group(1).strip()) else set()

    # 提取列名
    cols_match = re.search(r'Columns:\s*(.*?);', standard_str)
    columns = set(col.strip() for col in cols_match.group(1).split(', ')) if (
            cols_match and cols_match.group(1).strip()) else set()

    # 统一小写，消除大小写差异影响
    return {t.lower() for t in tables}, {c.lower() for c in columns}


def load_input_data(file_path: str) -> pd.DataFrame:
    """加载输入CSV数据（添加异常捕获，明确错误来源）"""
    try:
        df = pd.read_csv(file_path)
        # 校验必要列是否存在
        required_cols = ["question", "database_schema", "target_schema"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV缺少必要列，需包含：{required_cols}")
        return df
    except Exception as e:
        raise ValueError(f"数据加载失败：{str(e)}")


# ========================== 数据处理模块（真实标签、模型输入构建）==========================
def process_ground_truth(target_schema_raw: str) -> tuple[dict, str]:
    """
    处理真实标签schema：将原始字符串转为标准JSON和标准化字符串
    :param target_schema_raw: 原始标签字符串（如 "###Tables: singer;\n###Columns: singer.name;"）
    :return: (真实标签JSON, 真实标签标准化字符串)
    """
    # 提取表名和列名
    truth_tables, truth_cols = _extract_tables_and_columns(target_schema_raw)

    # 构建真实标签JSON（与模型输出格式对齐）
    ground_truth_json = {"schema": []}
    table_col_map = {table: [] for table in truth_tables}
    for col in truth_cols:
        try:
            table, col_name = col.split(".")
            if table in table_col_map:
                table_col_map[table].append(col_name)
        except ValueError:
            continue  # 跳过格式错误的列（如无表名的列）

    # 填充JSON结构
    for table, cols in table_col_map.items():
        ground_truth_json["schema"].append({"table_name": table, "columns": cols})

    # 生成标准化字符串（用于后续对比）
    tables_str = ', '.join(truth_tables)
    cols_str = ', '.join(truth_cols)
    ground_truth_str = f"###Tables: {tables_str};\n###Columns: {cols_str};"

    return ground_truth_json, ground_truth_str


def build_model_prompt(question: str, db_schema: str) -> str:
    """构建模型输入提示词（封装Prompt格式，便于后续调整）"""
    return f"<question>{question}</question>\n<database>{db_schema}</database>"


# ========================== 主逻辑 ==========================
def main():
    df = load_input_data(INPUT_CSV_PATH)
    training_samples = []
    seq_lengths = []

    print("✅ 正在构建训练样本...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="构建样本"):
        # 1. 处理 ground truth
        gt_json, gt_str = process_ground_truth(row["target_schema"])

        # 2. 构建对话消息（不含 assistant，用于训练输入）
        user_content = build_model_prompt(row["question"], row["database_schema"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        # 3. 计算完整对话（含 assistant 回复）的 token 长度，用于分析
        full_messages = messages + [{"role": "assistant", "content": json.dumps(gt_json, ensure_ascii=False, indent=None)}]
        full_text = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        input_ids = tokenizer([full_text], return_tensors="pt")["input_ids"]
        seq_len = input_ids.shape[1]
        seq_lengths.append(seq_len)

        # 4. 保存训练样本（仅含 prompt + ground truth，不含 full 对话）
        training_samples.append({
            "prompt": messages,
            "question": row["question"],
            "ground_truth": gt_json,
            "ground_truth_standard_str": gt_str,
            "seq_length": seq_len
        })

    # 随机打乱
    random.seed(42)
    random.shuffle(training_samples)

    # 输出统计信息
    max_len = max(seq_lengths) if seq_lengths else 0
    print(f"\n✅ 最大序列长度: {max_len}")  # 2258
    print(f"\n✅ 总样本数: {len(training_samples)}") # 8529

    # 保存为 JSONL
    output_path = Path(OUTPUT_JSONL_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"💾 正在写入 JSONL 文件: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in tqdm(training_samples, desc="写入文件"):
            # 移除 seq_length（若仅用于分析可保留；此处按需保留）
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("🎉 数据准备完成！")


if __name__ == "__main__":
    main()
