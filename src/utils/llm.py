"""
LLM调用模块 
Multi Model Support - 根据任务类型自动切换模型（推理模型、代码模型、基础模型）
"""
import json
import httpx
from typing import List, Dict, Any, Optional, Callable
from src.config import (
    SILICONFLOW_API_BASE,
    SILICONFLOW_API_KEY,
    LLM_MODEL,
    LLM_MODEL_ADVANCED,
    TASK_MODEL_MAPPING
)


class LLMClient:

    def __init__(
        self,
        api_base: str = SILICONFLOW_API_BASE,
        api_key: str = SILICONFLOW_API_KEY,
        model: str = LLM_MODEL,
        default_task_type: str = "general"
    ):
        self.api_base = api_base
        self.api_key = api_key
        self.default_model = model
        self.model = model
        self.default_task_type = default_task_type
        self.client = httpx.Client(timeout=600.0)  # 统一为600秒，与chat方法中的超时一致

    def set_task_type(self, task_type: str):
        """
        根据任务类型自动切换模型

        Args:
            task_type: 任务类型（code_generation, script_generation, data_visualization等）
        """
        self.model = TASK_MODEL_MAPPING.get(task_type, self.default_model)
        print(f"[LLM] 切换到模型: {self.model} (任务类型: {task_type})")

    def set_model(self, model: str):
        """手动设置模型"""
        self.model = model
        print(f"[LLM] 手动切换到模型: {self.model}")

    def reset_model(self):
        """重置为默认模型"""
        self.model = self.default_model

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        发送聊天请求

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            tools: 工具定义列表
            tool_choice: 工具选择策略

        Returns:
            API响应
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice

        try:
            response = self.client.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=600.0  # 增加到600秒（10分钟），给复杂计划生成更多时间
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            # 捕获HTTP错误并提供详细信息
            error_detail = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
            print(f"[LLM错误] {error_detail}")

            # 检查是否是token长度问题
            if e.response.status_code == 400:
                if "token" in e.response.text.lower() or "length" in e.response.text.lower():
                    print(f"[LLM错误] 可能是消息过长，尝试截断...")
                    # 如果是token长度问题，截断消息重试一次
                    if len(messages) > 1:
                        truncated_messages = messages[:1] + messages[-3:]  # 保留第一条和最后3条
                        payload["messages"] = truncated_messages
                        try:
                            response = self.client.post(
                                f"{self.api_base}/chat/completions",
                                headers=headers,
                                json=payload,
                                timeout=600.0  # 重试时也使用600秒超时
                            )
                            response.raise_for_status()
                            print(f"[LLM错误] 截断消息后重试成功")
                            return response.json()
                        except:
                            pass
            raise RuntimeError(f"LLM API调用失败: {error_detail}")
        except Exception as e:
            print(f"[LLM错误] 请求异常: {str(e)}")
            raise

    def chat_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict],
        tool_executor: Callable[[str, Dict], Any],
        max_iterations: int = 10
    ) -> str:
        """
        带工具调用的对话

        Args:
            messages: 初始消息
            tools: 工具定义
            tool_executor: 工具执行函数
            max_iterations: 最大迭代次数

        Returns:
            最终回复
        """
        current_messages = messages.copy()

        for _ in range(max_iterations):
            response = self.chat(current_messages, tools=tools, tool_choice="auto")

            choice = response["choices"][0]
            message = choice["message"]

            # 检查是否有工具调用
            if "tool_calls" in message and message["tool_calls"]:
                current_messages.append(message)

                for tool_call in message["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    func_args = json.loads(tool_call["function"]["arguments"])

                    # 执行工具
                    result = tool_executor(func_name, func_args)

                    # 添加工具结果
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, ensure_ascii=False)
                    })
            else:
                # 没有工具调用，返回最终结果
                return message.get("content", "")

        return "达到最大迭代次数"

    def generate_json(
        self,
        prompt_or_messages,
        schema_hint: str = "",
        system_prompt: str = "",
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict:
        """
        生成JSON格式输出

        Args:
            prompt_or_messages: 用户提示(str) 或 消息列表(List[Dict])
            schema_hint: JSON结构提示
            system_prompt: 系统提示
            temperature: 温度参数
            max_tokens: 最大输出token数（默认4096，规划任务需要更多）

        Returns:
            解析后的JSON
        """
        # 支持两种输入格式
        if isinstance(prompt_or_messages, list):
            # messages 列表格式
            messages = prompt_or_messages
        else:
            # 字符串 prompt 格式（向后兼容）
            prompt = prompt_or_messages
            if not system_prompt:
                system_prompt = "你是一个专业的科研助手，请严格按照JSON格式输出。"

            if schema_hint:
                prompt = f"{prompt}\n\n请按照以下JSON格式输出：\n{schema_hint}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

        response = self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        content = response["choices"][0]["message"]["content"]

        # 提取JSON - 使用多种策略
        import re

        # 策略1: 直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 策略2: 从markdown代码块中提取
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 策略2.5: 处理没有结束```的代码块（可能被截断）
        if content.strip().startswith('```'):
            # 移除开头的```json或```
            stripped = re.sub(r'^```(?:json)?\s*', '', content.strip())
            # 如果有结束```也移除
            stripped = re.sub(r'```\s*$', '', stripped)
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass

        # 策略3: 查找第一个{或[到最后一个}或]
        json_start = content.find('{') if '{' in content else content.find('[')
        json_end = content.rfind('}') if '}' in content else content.rfind(']')

        if json_start != -1 and json_end != -1 and json_end > json_start:
            try:
                json_str = content[json_start:json_end + 1]
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # 策略4: 尝试修复常见JSON错误
        try:
            # 移除多余的逗号
            fixed_content = re.sub(r',(\s*[}\]])', r'\1', content)
            # 移除注释
            fixed_content = re.sub(r'//.*?\n', '\n', fixed_content)
            fixed_content = re.sub(r'/\*.*?\*/', '', fixed_content, flags=re.DOTALL)
            return json.loads(fixed_content)
        except json.JSONDecodeError:
            pass

        # 策略5: 修复截断的JSON（补全缺失的括号）
        try:
            # 从markdown代码块或纯JSON中提取
            json_content = content
            if '```' in content:
                json_content = re.sub(r'^```(?:json)?\s*', '', content.strip())
                json_content = re.sub(r'```\s*$', '', json_content)

            # 找到JSON开始位置
            start_idx = json_content.find('{')
            if start_idx == -1:
                start_idx = json_content.find('[')
            if start_idx != -1:
                json_content = json_content[start_idx:]

            # 计算缺失的括号
            open_braces = json_content.count('{') - json_content.count('}')
            open_brackets = json_content.count('[') - json_content.count(']')

            # 移除末尾不完整的内容（如截断的字符串）
            # 找到最后一个完整的值结束位置
            last_complete = max(
                json_content.rfind('",'),
                json_content.rfind('"},'),
                json_content.rfind('"]'),
                json_content.rfind('}]'),
                json_content.rfind('"}'),
                json_content.rfind('true'),
                json_content.rfind('false'),
                json_content.rfind('null')
            )

            if last_complete > 0 and open_braces > 0:
                # 截断到最后一个完整的值
                json_content = json_content[:last_complete + 1]
                # 移除末尾可能的逗号
                json_content = json_content.rstrip().rstrip(',')
                # 重新计算缺失的括号
                open_braces = json_content.count('{') - json_content.count('}')
                open_brackets = json_content.count('[') - json_content.count(']')

            # 补全缺失的括号
            json_content = json_content + (']' * open_brackets) + ('}' * open_braces)

            result = json.loads(json_content)
            print(f"[JSON修复] 成功修复截断的JSON（补全了 {open_braces} 个 '}}' 和 {open_brackets} 个 ']'）")
            return result
        except json.JSONDecodeError:
            pass

        # 策略6: 更激进的截断修复 - 找到最后一个有效的属性值对
        try:
            json_content = content
            if '```' in content:
                json_content = re.sub(r'^```(?:json)?\s*', '', content.strip())
                json_content = re.sub(r'```\s*$', '', json_content)

            start_idx = json_content.find('{')
            if start_idx != -1:
                json_content = json_content[start_idx:]

                # 找最后一个完整的"key": value模式
                # 尝试多个截断点
                for pattern in [r'"[^"]*"\s*:\s*"[^"]*"', r'"[^"]*"\s*:\s*\[[^\]]*\]', r'"[^"]*"\s*:\s*\{[^}]*\}']:
                    matches = list(re.finditer(pattern, json_content))
                    if matches:
                        last_match = matches[-1]
                        truncated = json_content[:last_match.end()]

                        # 补全括号
                        open_b = truncated.count('{') - truncated.count('}')
                        open_br = truncated.count('[') - truncated.count(']')
                        truncated = truncated.rstrip().rstrip(',') + (']' * open_br) + ('}' * open_b)

                        try:
                            result = json.loads(truncated)
                            print(f"[JSON修复] 使用激进截断修复成功")
                            return result
                        except:
                            continue
        except:
            pass

        # 所有策略都失败，记录详细错误并抛出
        print(f"[JSON解析失败] LLM返回内容:\n{content[:500]}...")
        raise ValueError(f"无法解析JSON输出。内容前500字符: {content[:500]}")

    def close(self):
        """关闭客户端"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 全局客户端实例
_llm_client = None

def get_llm_client() -> LLMClient:
    """获取全局LLM客户端"""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
