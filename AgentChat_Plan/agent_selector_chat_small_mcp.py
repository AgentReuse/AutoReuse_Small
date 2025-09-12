import autogen
import asyncio
# from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

from autogen_agentchat.agents import AssistantAgent,UserProxyAgent
from autogen_agentchat.ui import Console
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.teams import SelectorGroupChat

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

import os
from Response_reuse import SemanticCache


model_client = OpenAIChatCompletionClient(
    model="qwen2.5:14b",
    base_url="http://localhost:11434/v1",
    api_key="NULL",
    model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
)



# STDIO MCPServer服务的配置
calculator_mcp_server = StdioServerParams(
    command="python",
    args=["mcp/calculator_server.py"]
)

semantic_cache = SemanticCache(
    embedding_model_path="./m3e-small",
    cache_path="./semantic_cache"
)

# ------------------ 终止检测（各 Agent 自己用）------------------
def agent_is_term(msg):
    try:
        return "TERMINATE" in str(msg.get("content", "")).upper()
    except Exception:
        return False

# ------------------ 自定义选手选择器（全局终止）------------------
def stop_on_terminate_selector(last_speaker, groupchat):
    # 若上一条消息触发终止，返回 None -> 立刻结束群聊
    if groupchat.messages:
        last = groupchat.messages[-1]
        content = str(last.get("content", "")).upper()
        if "TERMINATE" in content:
            print("[selector] TERMINATE detected -> stop chat.")
            return None

    # 否则简单轮询 user_proxy -> coder -> user_proxy ...
    # 你也可以实现更复杂的选择逻辑
    agents = groupchat.agents
    if last_speaker is None:
        return agents[0]
    idx = agents.index(last_speaker)
    return agents[(idx + 1) % len(agents)]

async def run_agent(task: str):
    #先加载mcp工具
    mcp_tool_calculator = await mcp_server_tools(calculator_mcp_server)

    embedding = semantic_cache.get_embedding(task)             #向量化
    similar_question, score, cached_data = semantic_cache.search_similar_query(embedding)   #相似性搜索

    isReuse = 1  # 0为不复用，1为计划复用，2为响应复用
    if score < 0.75:
        isReuse = 0
    elif 0.75 <= score < 0.95:
        isReuse = 1
    else:
        isReuse = 2

    #Test
    isReuse = 0

    if isReuse == 0:
        # ================== 定义 Agents ==================
        output_summarizer = AssistantAgent(
            name="OutputSummarizer",
            model_client=model_client,
            system_message=(
                "You will receive the full chat history after the group chat ends. "
                "Read it and produce a clear, step-by-step EXECUTION PLAN with:"
                " objectives & scope; tasks/milestones; owners (by agent names); "
                "deliverables & acceptance criteria; rough timeline; risks & mitigations. "
                "End with TERMINATE if complete, else CONTINUE."
            ),
        )

        coder = AssistantAgent(
            name="Coder",
            model_client=model_client,
            system_message=(
                "You will receive the full chat history after the group chat ends. "
                "Read it and produce a clear, step-by-step EXECUTION PLAN with:"
                " objectives & scope; tasks/milestones; owners (by agent names); "
                "deliverables & acceptance criteria; rough timeline; risks & mitigations. "
                "End with TERMINATE if complete, else CONTINUE."
            ),
            #code_execution_config={"work_dir": "output/coding", "use_docker": False},
        )


        general_agent = AssistantAgent(
            name="General_agent",
            model_client=model_client,
            system_message="You are General_agent, a versatile assistant responsible for handling tasks when no specialized agent is available. You should read the conversation context carefully and provide helpful, coherent, and logically consistent outputs. Your role is to fill in gaps, perform general reasoning, answer questions, or provide basic coding or documentation support as needed. Do not attempt to take over specialized responsibilities that belong to domain-specific agents unless explicitly required. Always ensure clarity, conciseness, and accuracy in your responses. Reply TERMINATE if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or explain the reason why the task is not solved yet.",
        )

        # user_proxy = autogen.UserProxyAgent(
        #     name="user_proxy",
        #     human_input_mode="NEVER",
        #     max_consecutive_auto_reply=10,
        #     llm_config=llm_config_codellama,
        #     is_termination_msg=agent_is_term,
        #     code_execution_config={"work_dir": "coding", "use_docker": False},
        #     system_message=(
        #         "Reply TERMINATE at the end of your response if the task has been solved at full satisfaction. "
        #         "Otherwise, reply CONTINUE, or explain why not solved yet."
        #     ),
        # )

        selector_prompt = """
            Now select the agent to execute the task.
            {roles}
            Current conversation context:
            {history}
            Read the above conversation and then choose one agent from {participants} to perform the next task.
            Ensure that the chosen agent is a professional agent with skills suitable for this conversation; if no such specialized agent is available, select General_agent.
            Select only one agent.
        """

        # 终止条件
        text_termination = TextMentionTermination("TERMINATE")
        max_message_termination = MaxMessageTermination(max_messages=25)
        termination = text_termination | max_message_termination

        # 创建团队
        team = SelectorGroupChat(
            [coder, output_summarizer, general_agent],
            model_client=model_client,
            termination_condition=termination,
            selector_prompt=selector_prompt,
            allow_repeated_speaker=True,  # 允许代理连续多轮发言。
        )

        # await Console(team.run_stream(task=task))
        history = []  # 用来保存所有消息
        async for event in team.run_stream(task=task):
            # event 可以是 BaseChatMessage, BaseAgentEvent, 或 TaskResult（最后一个）
            # 根据类型处理
            # 以下假设 event 有属性 `source` 或 `agent_name` 或类似
            # 并且如果是聊天消息，也有 content/text 属性

            # 将 event 转为字符串
            try:
                # 如果是聊天消息
                content = None
                agent = None
                if hasattr(event, "chat_message"):
                    # TaskResult 或者其他复杂类型可能有 chat_message 属性
                    cm = event.chat_message
                    content = getattr(cm, "content", repr(cm))
                    agent = getattr(cm, "source", None) or getattr(cm, "sender", None)
                elif hasattr(event, "content"):
                    content = event.content
                    agent = getattr(event, "source", None) or getattr(event, "agent", None)
                else:
                    # fallback
                    content = repr(event)
                    agent = None
            except Exception as e:
                content = repr(event)
                agent = None

            # 保存到 history
            history.append({
                "agent": agent,
                "event": content,
                "raw": event,
            })

        # `event` 完了之后，最后一个 item 应该是 TaskResult
        # 如果你想把 TaskResult 中的历史也拿出来：
        # TaskResult 的类型在 docs 里叫 TaskResult, 包含 .messages 属性
        # 可以这样：
        # 假设最后一个 event 被捕获为 task_result
        task_result = history[-1]["raw"]

        # =============== 提取 Coder 的输出 ===============
        coder_msgs = [m["content"] for m in history if m.get("name") == "Coder" and m.get("content")]

        # 拼接成一个字符串（如果你只需要合并结果）
        coder_output_text = "\n\n".join(coder_msgs)

        print("\n=== Coder Output ===\n", coder_output_text)

        # ================== 群聊结束后交给 OutputSummarizer 生成计划 ==================
        post_instruction = {
            "role": "user",
            "name": "Orchestrator",
            "content": (
                "Please read the entire chat history above and produce a structured EXECUTION PLAN as instructed."
            ),
        }

        sum_reply = output_summarizer.generate_reply(messages=history + [post_instruction])
        plan_text = sum_reply.get("content", "") if isinstance(sum_reply, dict) else str(sum_reply)

        semantic_cache.save_to_cache(task,coder_output_text,plan_text)  #存储响应和计划

        print("\n=== OutputSummarizer PLAN ===\n", plan_text)

    # elif isReuse == 1:
    #     plan_text=cached_data["plan"]
    #
    #     plan_proxy = autogen.UserProxyAgent(
    #         name="user_proxy",
    #         human_input_mode="NEVER",
    #         max_consecutive_auto_reply=10,
    #         llm_config=llm_config_codellama,
    #         is_termination_msg=agent_is_term,
    #         code_execution_config={"work_dir": "coding", "use_docker": False},
    #         system_message=plan_text,
    #     )
    #
    #     coder = autogen.AssistantAgent(
    #         name="Coder",
    #         llm_config=llm_config_codellama,
    #         is_termination_msg=agent_is_term,
    #         code_execution_config={"work_dir": "output/coding", "use_docker": False},
    #     )
    #
    #     groupchat = GroupChat(
    #         agents=[plan_proxy, coder],
    #         messages=[],
    #         max_round=12,
    #         speaker_selection_method=stop_on_terminate_selector,  # ★ 关键：全局终止控制
    #     )
    #     manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config_codellama)
    #
    #     # 开始群聊
    #     plan_proxy.initiate_chat(manager, message=task)
    #
    #     history = manager.groupchat.messages  # [{"role":..., "name":..., "content":...}, ...]
    #
    #     # =============== 提取 Coder 的输出 ===============
    #     coder_msgs = [m["content"] for m in history if m.get("name") == "Coder" and m.get("content")]
    #
    #     # 拼接成一个字符串（如果你只需要合并结果）
    #     coder_output_text = "\n\n".join(coder_msgs)
    #
    #     print("\n=== Coder Output ===\n", coder_output_text)


    elif isReuse == 2:
        response=cached_data["response"]

if __name__ == '__main__':
    task = "Write a python script to perform a quick sort."
    asyncio.run(run_agent(task=task))