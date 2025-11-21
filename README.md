# Schema Linking with DAPO 算法

[![Python 版本](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![许可证](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

基于 **DAPO**（直接优势策略优化）算法的 Text-to-SQL 模式选择强化学习方法。本项目基于 SCHEMA-R1 论文，实现了改进的训练方法和自定义奖励函数。

## 🎯 项目概述

本项目专注于 Text-to-SQL 系统中的 **模式链接（schema linking）** 任务，即从数据库模式中选择最相关的表和列来回答自然语言问题。

### 核心特性：
- **DAPO 算法**：实现直接优势策略优化，提供更好的训练稳定性
- **自定义奖励函数**：多目标奖励系统用于模式选择
- **高效训练**：LoRA 参数高效微调配合 VLLM 加速
- **全面评估**：基于精确率/召回率的表和列选择评估指标

## 🏗️ 项目结构

```
schema-linking-dapo/
├── src/                          # 源代码
│   ├── training/                 # 训练模块
│   │   └── grpo.py               # GRPO 训练器
│   ├── rewards/                  # 奖励函数
│   │   ├── schema_rewards.py     # 模式选择奖励
│   │   ├── format_rewards.py     # 格式验证奖励
│   │   └── base_rewards.py       # 基础工具奖励
│   ├── data/                     # 数据处理
│   │   └── data_processor.py     # 训练数据准备
│   └── utils/                    # 工具函数
├── configs/                      # 配置文件
│   └── training_config.yaml      # 训练参数
├── scripts/                      # 执行脚本
│   └── train_dapo_lora.sh        # 训练脚本
├── data/                         # 数据集
│   ├── train.jsonl               # 训练数据
│   └── val.jsonl                 # 验证数据
├── outputs/                      # 模型输出
├── requirements.txt              # 依赖项
├── setup.py                      # 安装配置
└── README.md                     # 项目文档
```

## 🚀 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/schema-linking-dapo.git
cd schema-linking-dapo

# 安装依赖
pip install -r requirements.txt

# 开发模式安装项目
pip install -e .
```

### 2. 数据准备

下载 SCHEMA-R1 数据集：
```bash
# 下载训练和验证数据到 data/ 目录
# 来源：https://github.com/hongWin/Schema-R1/tree/main/src/GRPO_data
# 需要的文件：training_prompt2.csv, val_prompt2.csv

# 将下载的文件放到 data/ 目录：
# data/training_prompt2.csv
# data/val_prompt2.csv

# 处理原始数据，生成训练数据
python src/data/data_processor.py
```

### 3. 训练

```bash
# 给训练脚本添加执行权限
chmod +x scripts/train_dapo_lora.sh

# 运行训练（脚本会自动处理路径）
./scripts/train_dapo_lora.sh
```

或者自定义训练参数：
```bash
# 编辑配置文件
vim configs/training_config.yaml

# 然后运行训练
./scripts/train_dapo_lora.sh
```

**注意**：训练脚本会自动检测项目根目录并使用正确的路径，无需切换目录。

## 🧠 奖励函数

项目实现了全面的奖励系统：

### 模式奖励 (`src/rewards/schema_rewards.py`)
- **表奖励**：基于精确率的正确表选择奖励
- **表惩罚**：对错误表选择的惩罚
- **列奖励**：基于精确率的正确列选择奖励
- **列惩罚**：对错误列选择的惩罚

### 格式奖励 (`src/rewards/format_rewards.py`)
- **思考标签惩罚**：确保正确使用 `</think>` 标签
- **有效 JSON 奖励**：验证 JSON 输出格式

### 基础奖励 (`src/rewards/base_rewards.py`)
- **长度惩罚**：防止过长的输出

## ⚙️ 配置

训练配置通过 `configs/training_config.yaml` 管理：

```yaml
# 关键参数
training:
  num_iterations: 2
  learning_rate: 1e-5
  loss_type: "dapo"
  epsilon: 0.2
  epsilon_high: 0.28

model:
  base_model: "/mnt/d/modelscope/Qwen3-0.6B"
  use_peft: true
  load_in_4bit: true
```

## 📊 模型架构

- **基础模型**：Qwen3-0.6B（或兼容模型）
- **训练方法**：DAPO（直接优势策略优化）
- **高效微调**：LoRA（低秩适应）
- **推理加速**：VLLM 共置模式
- **量化支持**：4-bit 量化

## 🔧 技术细节

### DAPO 算法
项目实现了 DAPO 的以下关键组件：
- **Token 级重要性采样**
- **双 epsilon 裁剪（ε, ε_high）**
- **优势归一化**

### 训练流程
1. **数据处理**：将 CSV 转换为带适当提示的 JSONL 格式
2. **模型训练**：使用自定义奖励函数的 GRPO 训练器
3. **评估**：模式选择性能的多指标评估

## 📈 性能

模型在模式选择任务上取得了强大的性能：
- **高精确率**的表和列选择
- **稳定训练**的 DAPO 算法
- **高效推理**的 LoRA 和 VLLM

## 🛠️ 开发指南

### 添加新的奖励函数
```python
# 在 src/rewards/ 目录下创建新的奖励函数
# 在训练脚本中添加引用路径
--reward_funcs "src.rewards.your_rewards.your_reward_function"
```

### 调整训练参数
编辑 `configs/training_config.yaml` 文件来调整：
- 学习率和批大小
- DAPO 特定参数（epsilon, beta 等）
- 模型路径和量化设置

## 🤝 贡献

我们欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 引用

如果您使用此代码，请引用我们的工作和原始 SCHEMA-R1 论文：

```bibtex
@article{schema-r1,
  title={SCHEMA-R1: A Reasoning Training Approach for Schema Linking in Text-to-SQL Task},
  author={...},
  journal={...},
  year={2024}
}
```

## 📜 许可证

本项目采用 Apache License 2.0 许可证 - 详情请查看 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- SCHEMA-R1 论文作者提供的原始数据集和方法
- TRL（Transformers Reinforcement Learning）团队的训练框架
- VLLM 团队提供的高效推理加速

## 🔗 相关链接

- [SCHEMA-R1 原始项目](https://github.com/hongWin/Schema-R1)
- [TRL 文档](https://huggingface.co/docs/trl)
- [VLLM 项目](https://github.com/vllm-project/vllm)