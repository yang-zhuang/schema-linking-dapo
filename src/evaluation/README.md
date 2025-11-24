# Qwen3-0.6B 模型评估系统

这是一个用于评估 Qwen3-0.6B 模型在模式链接任务上性能的模块化评估系统。

## 📋 数据集与评估标准

### 数据集信息
- **数据集来源**: SCHEMA-R1 论文评估集
- **论文**: [SCHEMA-R1: A REASONING TRAINING APPROACH FOR SCHEMA LINKING IN TEXT-TO-SQL TASK](https://arxiv.org/pdf/2506.11986)
- **数据集**: val_prompt2.csv
- **样本数量**: 1004 条

### 评估指标说明
系统采用与 SCHEMA-R1 论文一致的三项评估指标：

#### 1. Exact Match (EM) - 精确匹配
**含义**: 衡量预测内容是否与真实标签完全一致
**计算方式**:
```
EM = 1 (如果预测结果 == 真实标签) else 0
```

#### 2. Filtered Accuracy (FilteredAcc) - 过滤准确率
**含义**: 量化预测结果中正确内容所占的比例，即预测的准确性
**计算方式**:
```
FilteredAcc = |预测 ∩ 真实| / |预测|
```
这等价于标准的 **Precision**（精确率）

#### 3. Recall (Rec) - 召回率
**含义**: 评估模型预测的完整性，即真实标签中有多少被正确预测出来
**计算方式**:
```
Recall = |预测 ∩ 真实| / |真实|
```

### 指标计算映射
当前系统输出与论文标准的对应关系：
- `Average Strict Accuracy` → **EM (Exact Match)**
- `Average Precision` → **FilteredAcc** (过滤准确率)
- `Average Partial Accuracy` → **Recall** (召回率)

### 计算示例

#### Exact Match (EM) 示例
- 真实标签：`[table1.col1, table1.col2]`
- 预测结果：`[table1.col1, table1.col2]` → **EM = 1.0**
- 预测结果：`[table1.col1]` → **EM = 0.0**

#### Filtered Accuracy (Precision) 示例
- 真实标签：`[table1.col1, table2.col1]`
- 预测结果：`[table1.col1, table1.col2, table2.col1]`
- 重叠元素：`[table1.col1, table2.col1]` (2个)
- 预测结果总数：3个
- **FilteredAcc = Precision = 2/3 = 0.67**

#### Recall 示例
- 真实标签：`[table1.col1, table1.col2, table2.col1]`
- 预测结果：`[table1.col1, table2.col1]`
- 重叠元素：`[table1.col1, table2.col1]` (2个)
- 真实标签总数：3个
- **Recall = 2/3 = 0.67**

## 🚀 快速开始

### 1. 启动 vLLM 服务

首先需要启动 vLLM 服务来加载模型。编辑 `scripts/start_vllm.sh` 文件，将模型路径替换为你本地的路径：

```bash
# 编辑脚本中的模型路径
vim scripts/start_vllm.sh

# 启动服务
bash scripts/start_vllm.sh
```

**脚本配置说明：**
- `--model`: 基础模型路径（需要修改为你的本地路径）
- `--lora-modules`: LoRA 模块配置（格式：`模块名=LoRA路径`）
- `--served-model-name`: 服务中暴露的模型名称
- 其他参数：GPU内存利用率、最大长度、并发数等

### 2. 运行评估

启动 vLLM 服务后，运行评估：

```bash
# 评估未微调的基础模型
python scripts/run_evaluation.py --model_name "Qwen3-0.6B"

# 评估 LoRA 微调后的模型
python scripts/run_evaluation.py --model_name "dapo-Qwen3-0.6B"

# 调试模式（只处理少量样本）
python scripts/run_evaluation.py --debug --debug_samples 50
```

**注意：** `--model_name` 参数必须与 vLLM 脚本中的 `--served-model-name` 或 `--lora-modules` 中的模块名称保持一致。

## 📊 评估结果对比

### SCHEMA-R1 论文基线 (Qwen2.5-0.5B)
| 指标 | 表 (Table) | 列 (Column) |
|------|------------|-------------|
| **EM** | 55.38% | 13.24% |
| **FilteredAcc** | 75.60% | 44.02% |
| **Recall** | 85.34% | 64.86% |

### Qwen3-0.6B 评估结果

#### 微调前 (基础模型)
| 指标 | 表 (Table) | 列 (Column) |
|------|------------|-------------|
| **EM** | 42.23% | 19.02% |
| **FilteredAcc** | 92.32% | 74.92% |
| **Recall** | 55.68% | 44.76% |

#### DAPO+LoRA 微调后
| 指标 | 表 (Table) | 列 (Column) |
|------|------------|-------------|
| **EM** | 70.82% | 30.48% |
| **FilteredAcc** | 86.32% | 69.25% |
| **Recall** | 92.12% | 72.52% |

### 性能提升分析
- **表选择**:
  - EM 提升 28.59%
  - Recall 提升 36.44%
  - FilteredAcc 略有下降 (-6.00%)
- **列选择**:
  - EM 提升 11.46%
  - Recall 提升 27.76%
  - FilteredAcc 有所下降 (-5.67%)

**分析**: DAPO 微调显著提升了模型的召回率（更完整的预测），但略微降低了精确率，这是合理的权衡。

### 与 SCHEMA-R1 基线对比
- Qwen3-0.6B + DAPO 在 **表选择** 任务上表现显著优于 SCHEMA-R1 基线
- 在 **列选择** 任务上也有明显提升
- 模型展现出更好的完整性和准确性平衡

## 📁 项目结构

```
src/evaluation/
├── __init__.py                 # 模块导出
├── config.py                  # 配置常量
├── utils.py                   # 工具函数
├── data_loader.py             # 数据加载和预处理
├── model_client.py            # vLLM 模型交互
├── metrics_calculator.py      # 评估指标计算
├── file_handler.py            # 文件操作和结果保存
├── evaluate_qwen_model.py     # 主评估逻辑
└── README.md                  # 本文档

scripts/
├── start_vllm.sh              # vLLM 服务启动脚本
└── run_evaluation.py          # 评估执行脚本
```

## ⚙️ 配置选项

### 命令行参数 (evaluate_qwen_model.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--api_url` | `http://localhost:8000/v1` | vLLM API 服务地址 |
| `--api_key` | `token-abc123` | API 密钥 |
| `--model_name` | `Qwen3-0.6B` | 模型名称 |
| `--input_csv` | `../data/val_prompt2.csv` | 输入数据文件路径 |
| `--save_path` | `../data/model_process_results.csv` | 结果保存路径 |
| `--metrics_path` | `../data/eval_metrics.json` | 评估指标保存路径 |
| `--max_workers` | `5` | 并发工作线程数 |
| `--request_timeout` | `60` | 单个请求超时时间（秒） |
| `--system_prompt` | 默认提示词 | 系统提示词 |
| `--system_prompt_file` | `None` | 系统提示词文件路径 |
| `--debug` | `False` | 启用调试模式 |
| `--debug_samples` | `5` | 调试模式样本数量 |

### 配置文件 (config.py)

主要配置常量：
- API 服务配置
- 数据路径配置
- 并发配置
- 默认系统提示词

## 📝 输出结果

评估完成后会生成以下文件：

1. **model_process_results.csv**: 详细的处理结果
   - 问题原文
   - 模型原始响应
   - 解析后的 JSON 结果
   - 真实标签
   - 标准化后的预测和标签

2. **eval_metrics.json**: 评估指标汇总
   - 表名和列名的四项指标
   - 平均值统计

3. **控制台输出**: 实时显示评估进度和结果摘要

## 🔧 使用示例

### 评估基础模型
```bash
# 启动基础模型 vLLM 服务
# 修改 start_vllm.sh 中的 --lora-modules 部分来评估基础模型

# 运行评估
python scripts/run_evaluation.py --model_name "Qwen3-0.6B" --debug
```

### 评估 LoRA 微调模型
```bash
# 启动 LoRA 模型 vLLM 服务
bash scripts/start_vllm.sh

# 运行评估
python scripts/run_evaluation.py --model_name "dapo-Qwen3-0.6B"
```

### 自定义配置
```bash
python scripts/run_evaluation.py \
    --model_name "dapo-Qwen3-0.6B" \
    --input_csv data/test_data.csv \
    --max_workers 3 \
    --debug_samples 100
```

## 🐛 常见问题

1. **vLLM 启动失败**
   - 检查模型路径是否正确
   - 确认 GPU 内存是否充足
   - 安装必要的依赖：`pip install vllm`

2. **评估连接失败**
   - 确认 vLLM 服务已启动
   - 检查端口配置是否一致
   - 验证模型名称是否匹配

3. **内存不足**
   - 减少 `--max_workers` 数量
   - 降低 `--max_num_seqs` 参数
   - 调整 `--gpu-memory-utilization`

## 📚 参考文献

- **SCHEMA-R1 论文**: [A REASONING TRAINING APPROACH FOR SCHEMA LINKING IN TEXT-TO-SQL TASK](https://arxiv.org/pdf/2506.11986)
- **评估代码**: 基于 SCHEMA-R1 论文的评估方法实现
- **模型**: Qwen3-0.6B vs Qwen2.5-0.5B (Schema-R1) 性能对比