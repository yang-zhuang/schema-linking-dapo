import re
from typing import Dict, List, Any, Optional
import json_repair


def _extract_tables_and_columns(schema: List[Dict[str, Any]]) -> tuple[set, set]:
    """从 schema 列表中提取所有表名和列名的集合，统一转换为小写"""
    tables = set()
    columns = set()
    for item in schema:
        table_name = item.get("table_name")
        if table_name:
            # 将表名转换为小写
            try:
                tables.add(table_name.lower())
            except Exception as e:
                pass

        cols = item.get("columns", [])
        if isinstance(cols, list):
            # 将所有列名转换为小写
            try:
                columns.update(col.lower() for col in cols)
            except Exception as e:
                pass
    return tables, columns


def _parse_completion_content(completion_item: list[dict[str, str]]) -> Optional[Dict[str, Any]]:
    """解析 completion 内容，提取 JSON 数据"""
    if isinstance(completion_item, str):
        generated_content = completion_item
    else:
        generated_content = completion_item[0]["content"]

    # 解析 content
    if '</think>' in generated_content:
        try:
            json_match = re.search("</think>(.*?)$", generated_content, re.DOTALL)
            if json_match:
                json_text = json_match.group(1).strip()
                json_content = json_repair.loads(json_text)
                # 验证是否为字典类型
                if isinstance(json_content, dict):
                    return json_content
        except Exception:
            pass
    return None


def table_reward(completions: list[list[dict[str, str]]], ground_truth, **kwargs) -> list[Optional[float]]:
    """
    表奖励：计算completion中正确表的比例（相对于ground truth）

    分数设计：
    - 范围：0.0 到 1.0
    - 计算公式：正确预测的表数量 / 真实表总数
    - 特殊情况：
        * 真实表为空且生成表为空：1.0（完全匹配）
        * 真实表为空但生成了表：0.0（错误生成）
        * 解析失败或格式错误：0.0

    示例：
    - 真实表：{table1, table2}，生成表：{table1, table3} → 得分：1/2 = 0.5
    - 真实表：{}，生成表：{} → 得分：1.0
    - 真实表：{}，生成表：{table1} → 得分：0.0
    """

    rewards = []
    for completion, truth_content in zip(completions, ground_truth):
        # 解析生成内容
        json_content = _parse_completion_content(completion)
        if json_content is None:
            rewards.append(0.0)
            continue

        # 提取生成结果和真实值中的表名
        generated_tables, _ = _extract_tables_and_columns(json_content.get("schema", []))
        truth_tables, _ = _extract_tables_and_columns(truth_content.get("schema", []))

        # 计算表匹配得分
        if not truth_tables and not generated_tables:
            rewards.append(float(1)) # 空对空，完全匹配
        elif not truth_tables and generated_tables:
            rewards.append(float(0)) # 有真实表但未生成任何表
        else:
            # 计算正确预测的表数量
            correct_tables = len(generated_tables & truth_tables)
            # 得分 = 正确表数 / 真实表总数
            score = correct_tables / len(truth_tables)
            rewards.append(float(score))

    return rewards


def table_penalty(completions: list[list[dict[str, str]]], ground_truth, **kwargs) -> list[Optional[float]]:
    """
    表惩罚：计算completion中错误表的比例（相对于completion中的表总数）

    分数设计：
    - 范围：-1.0 到 0.0
    - 计算公式：- (错误表数量 / 生成表总数)
    - 特殊情况：
        * 生成表为空且真实表为空：0.0（无惩罚）
        * 生成表为空但真实表不为空：-1.0（最大惩罚，应选但未选）
        * 生成表不为空但真实表为空：-1.0（最大惩罚，不应选但选了）
        * 解析失败或格式错误：0.0（无惩罚）

    示例：
    - 生成表：{table1, table2, table3}，真实表：{table1} → 惩罚：-2/3 ≈ -0.67
    - 生成表：{table1}，真实表：{table1, table2} → 惩罚：0.0（没有错误表）
    - 生成表：{}，真实表：{table1} → 惩罚：-1.0（最大惩罚，应选但未选）
    - 生成表：{table1}，真实表：{} → 惩罚：-1.0（最大惩罚，不应选但选了）
    - 生成表：{}，真实表：{} → 惩罚：0.0
    """
    penalties = []

    for completion, truth_content in zip(completions, ground_truth):
        # 解析生成内容
        json_content = _parse_completion_content(completion)
        if json_content is None:
            penalties.append(0.0)  # 解析失败，无惩罚
            continue

        # 提取生成结果和真实值中的表名
        comp_tables, _ = _extract_tables_and_columns(json_content.get("schema", []))
        gt_tables, _ = _extract_tables_and_columns(truth_content.get("schema", []))

        if not comp_tables:
            # 生成表为空的情况
            if gt_tables:
                penalties.append(-1.0)  # 应选但未选，最大惩罚
            else:
                penalties.append(0.0)  # 双方都为空，无惩罚
        else:
            # 计算错误表的数量
            incorrect_tables = len(comp_tables - gt_tables)
            # 惩罚 = - (错误表数 / 生成表总数)
            penalty = - (incorrect_tables / len(comp_tables))
            penalties.append(float(penalty))

    return penalties


def column_reward(completions: list[list[dict[str, str]]], ground_truth, **kwargs) -> list[Optional[float]]:
    """
    列奖励：计算completion中正确列的比例（相对于ground truth）

    分数设计：
    - 范围：0.0 到 1.0
    - 计算公式：正确预测的列数量 / 真实列总数
    - 特殊情况：
        * 真实列为空且生成列为空：1.0（完全匹配）
        * 真实列为空但生成了列：0.0（错误生成）
        * 解析失败或格式错误：0.0

    示例：
    - 真实列：{col1, col2, col3}，生成列：{col1, col2, col4} → 得分：2/3 ≈ 0.67
    - 真实列：{}，生成列：{} → 得分：1.0
    - 真实列：{}，生成列：{col1} → 得分：0.0
    """
    rewards = []

    for completion, truth_content in zip(completions, ground_truth):
        # 解析生成内容
        json_content = _parse_completion_content(completion)
        if json_content is None:
            rewards.append(0.0)  # 解析失败，无奖励
            continue

        # 提取生成结果和真实值中的列名
        _, comp_columns = _extract_tables_and_columns(json_content.get("schema", []))
        _, gt_columns = _extract_tables_and_columns(truth_content.get("schema", []))

        # 计算列匹配得分
        if not gt_columns and not comp_columns:
            rewards.append(float(1))  # 空对空，完全匹配
        elif not gt_columns and comp_columns:
            rewards.append(float(0))  # 真实无列但生成了列
        else:
            # 计算正确预测的列数量
            correct_columns = len(comp_columns & gt_columns)
            # 得分 = 正确列数 / 真实列总数
            score = correct_columns / len(gt_columns)
            rewards.append(float(score))

    return rewards


def column_penalty(completions: list[list[dict[str, str]]], ground_truth, **kwargs) -> list[Optional[float]]:
    """
    列惩罚：计算completion中错误列的比例（相对于completion中的列总数）

    分数设计：
    - 范围：-1.0 到 0.0
    - 计算公式：- (错误列数量 / 生成列总数)
    - 特殊情况：
        * 生成列为空且真实列为空：0.0（无惩罚）
        * 生成列为空但真实列不为空：-1.0（最大惩罚，应选但未选）
        * 生成列不为空但真实列为空：-1.0（最大惩罚，不应选但选了）
        * 解析失败或格式错误：0.0（无惩罚）

    示例：
    - 生成列：{col1, col2, col3, col4}，真实列：{col1, col2} → 惩罚：-2/4 = -0.5
    - 生成列：{col1, col2}，真实列：{col1, col2, col3} → 惩罚：0.0（没有错误列）
    - 生成列：{}，真实列：{col1} → 惩罚：-1.0（最大惩罚，应选但未选）
    - 生成列：{col1}，真实列：{} → 惩罚：-1.0（最大惩罚，不应选但选了）
    - 生成列：{}，真实列：{} → 惩罚：0.0
    """
    penalties = []

    for completion, truth_content in zip(completions, ground_truth):
        # 解析生成内容
        json_content = _parse_completion_content(completion)
        if json_content is None:
            penalties.append(0.0)  # 解析失败，无惩罚
            continue

        # 提取生成结果和真实值中的列名
        _, comp_columns = _extract_tables_and_columns(json_content.get("schema", []))
        _, gt_columns = _extract_tables_and_columns(truth_content.get("schema", []))

        if not comp_columns:
            # 生成为空的情况
            if gt_columns:
                penalties.append(-1.0)  # 应选但未选，最大惩罚
            else:
                penalties.append(0.0)  # 双方都为空，无惩罚
        else:
            # 计算错误列的数量
            incorrect_columns = len(comp_columns - gt_columns)
            # 惩罚 = - (错误列数 / 生成列总数)
            penalty = - (incorrect_columns / len(comp_columns))
            penalties.append(float(penalty))

    return penalties