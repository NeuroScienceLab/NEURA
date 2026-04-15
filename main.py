"""
NEURA
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.config import OUTPUT_DIR
from src.agent.research_graph import create_agent, ResearchAgentLangGraph


def post_completion_chat(state: dict):
    """完成后对话模式 - 基于已有 state 回答问题，支持多轮对话记忆"""
    from src.utils.llm import get_llm_client
    llm = get_llm_client()

    # 构建丰富上下文：工具结果摘要
    tool_summaries = []
    for r in state.get("tool_results", [])[-10:]:
        tool_summaries.append(f"- {r.get('tool_name','?')}: {r.get('status','?')} | {str(r.get('output',''))[:200]}")

    system_context = (
        f"你是一个神经影像分析助手。以下是本次分析的完整结果:\n"
        f"研究问题: {state.get('question', '')}\n"
        f"分析阶段: {state.get('phase')}\n"
        f"质量分: {state.get('scientific_quality_score', 0)}/10\n"
        f"迭代次数: {state.get('iteration_count', 0)}\n"
        f"评估反馈: {state.get('iteration_feedback', '')}\n"
        f"报告路径: {state.get('report_path', '无')}\n"
        f"\n工具执行结果:\n" + "\n".join(tool_summaries) + "\n"
        f"\n报告摘要:\n{state.get('report', '')[:2000]}"
    )

    conversation_history = [{"role": "system", "content": system_context}]

    print("\n" + "="*60)
    print("分析完成。进入对话模式（输入 'exit' 退出）")
    print("="*60)

    while True:
        try:
            q = input("\n你的问题 (exit退出): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ['exit', 'quit', 'q', '']:
            break

        conversation_history.append({"role": "user", "content": q})

        # 对话历史防溢出：超过15轮时保留system + 最近10轮
        if len(conversation_history) > 31:
            conversation_history = [conversation_history[0]] + conversation_history[-20:]

        resp = llm.chat(conversation_history, max_tokens=2048)
        answer = resp["choices"][0]["message"]["content"]
        conversation_history.append({"role": "assistant", "content": answer})
        print(f"\n{answer}")


def run_agent(question: str, thread_id: str = None, auto_continue: bool = False):
    """
    运行LangGraph Agent

    Args:
        question: 研究问题
        thread_id: 线程ID（用于恢复）
    """
    print("=" * 70)
    print("科研Agent (LangGraph版本)")
    print("=" * 70)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"研究问题: {question}\n")

    # 创建并运行Agent
    agent = create_agent()
    result = agent.run(question, thread_id, auto_continue=auto_continue)

    # 输出结果
    print(f"\n阶段: {result.get('phase', 'unknown')}")

    if result.get("report_path"):
        print(f"报告路径: {result['report_path']}")

    if result.get("last_error"):
        print(f"最后错误: {result['last_error']}")

    # 完成后进入对话模式
    if not auto_continue:
        post_completion_chat(result)

    return result


def show_graph():
    """显示图结构"""
    agent = create_agent()
    mermaid = agent.get_graph_image()
    print("LangGraph 图结构 (Mermaid格式):\n")
    print(mermaid)


def interactive_mode():
    """交互模式"""
    print("=" * 70)
    print("科研Agent (LangGraph版本) - 交互模式")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'graph' 查看图结构")
    print("=" * 70)

    agent = create_agent()

    while True:
        print("\n")
        question = input("请输入研究问题: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("再见!")
            break

        if question.lower() == 'graph':
            print(agent.get_graph_image())
            continue

        if not question:
            print("请输入有效的问题")
            continue

        try:
            result = agent.run(question)
        except Exception as e:
            print(f"执行出错: {e}")
            import traceback
            traceback.print_exc()


def list_runs():
    """列出所有运行"""
    runs_dir = OUTPUT_DIR / "runs"
    if not runs_dir.exists():
        print("没有运行记录")
        return

    print("=" * 70)
    print("运行历史")
    print("=" * 70)

    for run_dir in sorted(runs_dir.iterdir(), reverse=True)[:10]:
        if not run_dir.is_dir():
            continue

        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"\n[{meta.get('run_id')}] {meta.get('status')}")
            print(f"  问题: {meta.get('question', '')[:50]}...")
            print(f"  时间: {meta.get('created_at', '')}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="科研Agent (LangGraph版本)")
    parser.add_argument(
        "question",
        nargs="?",
        help="研究问题"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="交互模式"
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="显示图结构"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出运行历史"
    )
    parser.add_argument(
        "--thread",
        type=str,
        help="指定线程ID（用于恢复）"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="自动模式，跳过所有人工审查暂停"
    )

    args = parser.parse_args()

    if args.graph:
        show_graph()
        return

    if args.list:
        list_runs()
        return

    if args.interactive:
        interactive_mode()
        return

    if args.question:
        run_agent(args.question, args.thread, auto_continue=args.auto)
    else:
        # 默认示例
        default_question = "比较SCA3患者组与健康对照组的脑结构差异"
        print(f"使用默认问题: {default_question}")
        run_agent(default_question, auto_continue=args.auto)


if __name__ == "__main__":
    main()
