# Schema Linking DAPO 项目源码

这是 Schema Linking DAPO 项目的源代码目录。

## 📁 项目结构

```
src/
├── README.md                    # 本文档
├── __init__.py                  # 包初始化文件
├── training/                    # 训练模块
│   └── grpo.py                 # GRPO训练器实现
├── rewards/                     # 奖励函数模块
│   ├── schema_rewards.py        # 模式选择奖励
│   ├── format_rewards.py        # 格式验证奖励
│   └── base_rewards.py          # 基础工具奖励
├── data/                        # 数据处理模块
│   └── data_processor.py        # 训练数据准备和预处理
├── utils/                       # 工具函数模块
└── evaluation/                  # 模型评估系统
    ├── README.md                # 评估系统详细文档
    ├── __init__.py              # 评估模块导出
    ├── config.py                # 配置常量
    ├── utils.py                 # 工具函数
    ├── data_loader.py           # 数据加载和预处理
    ├── model_client.py          # vLLM模型交互
    ├── metrics_calculator.py    # 评估指标计算
    ├── file_handler.py          # 文件操作和结果保存
    └── evaluate_qwen_model.py   # 主评估逻辑
```

## 📊 训练数据集

### 数据集来源
- **数据集**: SCHEMA-R1 论文数据集
- **总样本数**: 8,539 条
- **论文链接**: [SCHEMA-R1: A REASONING TRAINING APPROACH FOR SCHEMA LINKING IN TEXT-TO-SQL TASK](https://arxiv.org/pdf/2506.11986)

### 数据使用差异

#### SCHEMA-R1 数据划分
- **SFT 数据**: 200 样本（用于 CoT 监督微调）
- **GRPO 训练数据**: 8,329 样本（用于强化学习训练）

#### 当前项目数据使用
- **直接 DAPO 训练**: 使用全部 8,539 样本
- **无 SFT 阶段**: 直接进行 DAPO 训练

### 输出格式对比

#### SCHEMA-R1 格式
```
###table: table1, table2
###columns: table1.col1, table2.col1
```

#### 当前项目格式
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

## 🏆 奖励函数体系设计与分析

### 1. 整体架构概述

当前项目与SCHEMA-R1论文采用了不同的奖励函数设计思路。下面将详细阐述两者的奖励函数体系，并进行对比分析。

### 2. 当前项目奖励函数体系

#### 2.1 基础奖励 (Base Rewards)

**软长度惩罚 (Soft Overlong Punishment)**
- **位置**: `src/rewards/base_rewards.py:get_soft_overlong_punishment_medium`
- **计算逻辑**:
  - 当生成内容长度 ≤ 896 tokens: 无惩罚 (0.0)
  - 当896 < 长度 ≤ 1024 tokens: 线性惩罚 `(896 - completion_length) / 128`
  - 当长度 > 1024 tokens: 最大惩罚 (-1.0)
- **设计目的**: 避免生成过长内容，平衡推理质量与计算效率

#### 2.2 格式奖励 (Format Rewards)

**思考标签验证 (Think Tag Validation)**
- **位置**: `src/rewards/format_rewards.py:think_tag_penalty`
- **计算逻辑**:
  - 包含完整`</think>`标签: 0.0 (无惩罚)
  - 缺少标签: -1.0 (最大惩罚)

**JSON格式验证 (Valid JSON Reward)**
- **位置**: `src/rewards/format_rewards.py:valid_json_reward`
- **计算逻辑**:
  - 能成功解析为字典: +1.0
  - 解析失败: -1.0

#### 2.3 模式链接奖励 (Schema Linking Rewards)

**表选择奖励与惩罚 (Table Selection)**
- **位置**: `src/rewards/schema_rewards.py`
- **奖励计算**: `len(生成表 ∩ 真实表) / len(真实表)` (范围: 0.0-1.0)
- **惩罚计算**: `-len(生成表 - 真实表) / len(生成表)` (范围: -1.0-0.0)
- **特殊情况处理**:
  - 双方都为空: 奖励=1.0, 惩罚=0.0
  - 仅生成表为空: 奖励=0.0, 惩罚=-1.0

**列选择奖励与惩罚 (Column Selection)**
- **计算逻辑**: 与表选择相同，但应用于列级别
- **权重设计**: 列选择依赖于表选择的正确性，但当前系统未明确体现这种依赖关系

### 3. SCHEMA-R1奖励函数体系详解

#### 3.1 格式奖励 (Format Reward)

SCHEMA-R1采用双层格式验证机制，确保输出符合特定结构要求：

```
Rf = 1, format_success
    0, format_fail
```

```
Rc = bool(count("###table:") == 1) + bool(count("###columns:") == 1)
```

- **第一层验证 (Rf)**: 确保模型在`<answer>...</answer>`标签内放置最终答案，在`<think>...</think>`标签内放置推理过程
- **第二层验证 (Rc)**: 
  - 验证表预测是否位于`###table:`标记之后
  - 验证列预测是否位于`###columns:`标记之后
  - 每项验证成功得1分，总分范围为0-2分
- **设计目的**: 在允许模型创造性探索的同时，维持严格的输出格式规范，避免RL训练过程中格式混乱

#### 3.2 推理长度奖励 (Reasoning Length Reward)

SCHEMA-R1设置了明确的长度控制边界，避免过短或过长的推理过程：

```
Rl = {
    0, len_response < Lower Length
    1, Lower Length <= len_response < Upper Length
    0, Upper Length <= len_response
}
```

- **参数定义**:
  - `len_response`: 响应的总token数量
  - `Lower Length`: 允许的最小推理长度
  - `Upper Length`: 允许的最大推理长度
- **设计依据**: 研究表明过长的推理会增加训练和推理时间但不一定提升性能，而过短的推理可能无法充分解决问题
- **实现机制**: 非连续的二元奖励(0或1)，只有在理想长度范围内才给予奖励

#### 3.3 模式链接奖励 (Schema Linking Reward) - 详细文字解释

SCHEMA-R1的模式链接奖励是其最具特色的部分，采用精细化设计来评估模型在数据库表和列选择上的准确度。这部分奖励分为表预测和列预测两部分，并明确认识到表预测的正确性对列预测有决定性影响。

##### 表预测奖励机制 (直观解释)

表预测奖励的设计思想非常直观：既要奖励正确预测，也要惩罚错误预测，且考虑预测集合的规模影响。

- **奖励计算逻辑**：
  1. 首先设定一个最大奖励值（例如Rt_max = 0.8）
  2. 将这个最大奖励平均分配给每个正确的表预测。例如，如果真实有4个表，且最大奖励是0.8，那么每个正确预测的表可获得0.8/4 = 0.2分
  3. 如果模型正确预测了3个表，那么总奖励为0.2 × 3 = 0.6分

- **惩罚计算逻辑**：
  1. 设定一个最大惩罚值（例如Pt_max = 0.8）
  2. 将最大惩罚平均分配给每个错误的表预测。例如，如果模型预测了5个表，其中2个是错误的，那么每个错误预测的惩罚为0.8/5 = 0.16分
  3. 总惩罚为0.16 × 2 = 0.32分

- **最终表预测奖励** = 总奖励 - 总惩罚 = 0.6 - 0.32 = 0.28分

**通俗理解**：这就像老师批改试卷，不仅看你答对了几题（给予分数），还要看你答错了几题（扣分）。如果你只答对了部分题目但没有乱猜，你可能得分更高；如果你猜了很多但错的也多，总分就会降低。

##### 列预测奖励机制 (直观解释)

列预测奖励的计算方式与表预测相同，但参数设置不同，体现了任务的层次结构：

- **参数设置差异**：
  - 列预测的最大奖励值（Rc_max）小于表预测的最大奖励值（Rt_max）
  - 列预测的最大惩罚值（Pc_max）小于表预测的最大惩罚值（Pt_max）
  
  例如：Rt_max = 0.8, Rc_max = 0.6, Pt_max = 0.8, Pc_max = 0.6

- **设计原因**：在数据库查询任务中，先选择正确的表是选择正确列的前提。如果表选错了，那么在错误表上选择的列再准确也无济于事。因此，系统对表预测赋予更高的权重。

**示例场景**：
- 假设一个查询需要访问`employees`和`departments`两个表
- 真实需要的列是：`employees.name`, `employees.salary`, `departments.name`
- 模型预测的表：`employees`（正确），`projects`（错误）
- 模型预测的列：`employees.name`（正确），`employees.salary`（正确），`projects.budget`（错误）

在这种情况下：
1. 表预测部分：1个正确，1个错误 → 中等奖励，中等惩罚
2. 列预测部分：2个正确，1个错误 → 但因为有一个表是错误的，这些列预测的意义大打折扣

通过设置Rt_max > Rc_max，系统明确告诉模型：先确保选对表，再考虑选对列。

### 奖励函数对比分析

| 特性 | SCHEMA-R1 | 当前项目 |
|------|-----------|----------|
| **数据格式** | 纯文本 (`###table:...`) | JSON结构化 |
| **格式检查** | 正则匹配 | JSON解析验证 |
| **长度控制** | 硬边界惩罚 | 软边界惩罚 |
| **表列权重** | `Rtmax > Rcmax` | 独立奖励/惩罚 |

## 🚀 核心模块说明

### 1. 训练模块 (`training/`)

#### `grpo.py` - GRPO训练器
- **功能**: 实现基于 GRPO（Generalized Relative Policy Optimization）的训练算法
- **特性**:
  - 支持多目标奖励函数
  - 集成vLLM推理加速
  - 支持LoRA参数高效微调
  - Token级别的重要性采样

### 2. 奖励函数模块 (`rewards/`)

#### `schema_rewards.py` - 模式选择奖励
- **功能**: 评估模型选择的表和列的准确性
- **实现**: 精确率和召回率的独立奖励/惩罚机制
- **指标**:
  - 表选择精确率和召回率
  - 列选择精确率和召回率
  - 支持空表/列的特殊情况处理

#### `format_rewards.py` - 格式验证奖励
- **功能**: 验证模型输出的JSON格式和结构
- **实现**: 思考标签检查和JSON解析验证
- **检查项**:
  - JSON格式有效性
  - 思考标签完整性

#### `base_rewards.py` - 基础工具奖励
- **功能**: 提供基础的长度控制和工具函数
- **实现**: 基于DAPO论文的软长度惩罚机制
- **包含**:
  - 多种长度预设 (short/medium/long/xlong)
  - 软边界惩罚计算

### 3. 数据处理模块 (`data/`)

#### `data_processor.py` - 数据处理器
- **功能**: 处理和预处理训练数据
- **特性**:
  - CSV到JSONL格式转换
  - 对话格式数据构建
  - 并发处理支持
  - Token长度分析

### 4. 评估系统 (`evaluation/`)

评估系统是一个完整的子项目，用于评估训练好的模型性能：

#### 主要功能
- **多指标评估**: EM、FilteredAcc、Recall
- **vLLM集成**: 支持本地模型服务和LoRA加载
- **模块化设计**: 清晰的代码组织和职责分离
- **详细报告**: 生成CSV和JSON格式的评估结果

#### 使用方式
```bash
# 启动vLLM服务
bash scripts/start_vllm.sh

# 运行评估
python scripts/run_evaluation.py --model_name "dapo-Qwen3-0.6B"
```

详细文档请参见: [src/evaluation/README.md](evaluation/README.md)

## 🛠️ 技术栈

### 核心框架
- **PyTorch**: 深度学习框架
- **TRL (Transformer Reinforcement Learning)**: RL训练框架
- **vLLM**: 高性能推理引擎
- **Transformers**: HuggingFace模型库

### 训练技术
- **DAPO**: 强化学习算法
- **GRPO**: 通用相对策略优化
- **LoRA (Low-Rank Adaptation)**: 参数高效微调
- **重要性采样**: 训练优化技术

### 数据处理
- **Pandas**: 数据处理
- **JSON**: 数据格式
- **并发处理**: 性能优化

## 📋 使用指南

### 1. 环境准备
```bash
# 安装依赖
pip install torch transformers trl vllm pandas numpy
```

### 2. 数据准备
```python
from src.data.data_processor import DataProcessor

# 处理训练数据
processor = DataProcessor()
processor.load_and_process_data(
    input_csv="data/training_prompt2.csv",
    output_jsonl="data/train.jsonl"
)
```

### 3. 模型训练
```bash
# 使用训练脚本
bash scripts/train_dapo_lora.sh
```

### 4. 模型评估
```bash
# 启动评估服务
bash scripts/start_vllm.sh

# 运行评估
python scripts/run_evaluation.py --model_name "dapo-Qwen3-0.6B"
```

## 📊 性能指标

基于SCHEMA-R1论文评估集的测试结果：

| 模型 | 表选择 EM | 列选择 EM | 表选择 Recall | 列选择 Recall |
|------|-----------|-----------|---------------|---------------|
| Qwen3-0.6B (基础) | 42.23% | 19.02% | 55.68% | 44.76% |
| Qwen3-0.6B + DAPO | 70.82% | 30.48% | 92.12% | 72.52% |

## 📚 参考文献

### 论文资料
- **DAPO论文**: [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- **SCHEMA-R1论文**: [A Reasoning Training Approach for Schema Linking in Text-to-SQL Task](https://arxiv.org/pdf/2506.11986)
- **GRPO论文**: [Generalized Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
