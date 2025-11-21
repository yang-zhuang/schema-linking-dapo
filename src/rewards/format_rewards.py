import re
import json_repair


def think_tag_penalty(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    """
    思考标签惩罚：检查生成内容中是否包含</think>标签

    分数设计：
    - 包含</think>标签：0.0（无惩罚）
    - 不包含</think>标签：-1.0（有惩罚）

    示例：
    - 内容："<think>...</think>..." → 惩罚：0.0
    - 内容："直接回答" → 惩罚：-1.0
    """
    if isinstance(completions[0], str):
        completion_contents = [completion for completion in completions]
    else:
        completion_contents = [completion[0]["content"] for completion in completions]

    penalties = []
    for content in completion_contents:
        # 检查是否包含</think>标签
        if "</think>" in content:
            penalties.append(0.0)  # 包含标签，无惩罚
        else:
            penalties.append(-1.0)  # 不包含标签，有惩罚

    return penalties


def valid_json_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    """
    有效JSON奖励：检查是否能从内容中解析出有效的字典

    分数设计：
    - 能解析成字典：1.0（有奖励）
    - 不能解析成字典：-1.0（无奖励）

    示例：
    - 内容："</think>{\"key\": \"value\"}" → 奖励：1.0
    - 内容："</think>invalid json" → 奖励：-1.0
    - 内容："直接回答" → 奖励：-1.0
    """
    if isinstance(completions[0], str):
        completion_contents = [completion for completion in completions]
    else:
        completion_contents = [completion[0]["content"] for completion in completions]

    rewards = []
    for content in completion_contents:
        # 尝试解析JSON内容
        json_content = None
        if '</think>' in content:
            try:
                json_match = re.search("</think>(.*?)$", content, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1).strip()
                    json_content = json_repair.loads(json_text)
            except Exception:
                pass

        # 检查是否成功解析为字典
        if json_content is not None and isinstance(json_content, dict):
            rewards.append(1.0)  # 成功解析为字典，有奖励
        else:
            rewards.append(-1.0)  # 未能解析为字典，无奖励

    return rewards
