import autogen
import asyncio
import argparse
# from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

from autogen_agentchat.agents import AssistantAgent,UserProxyAgent
from autogen_agentchat.ui import Console
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools, SseServerParams
from autogen_agentchat.teams import SelectorGroupChat,RoundRobinGroupChat

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.base import TaskResult

import time

import os
from Response_reuse import SemanticCache
from llm_mcp_agent import build_agents, default_terminations, SelectorGroupChat, Console, create_model_client


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

async def run_agent(task: str,enable_reuse: bool):
    start = time.time()

    embedding = semantic_cache.get_embedding(task)             #向量化
    similar_question, score, cached_data = semantic_cache.search_similar_query(embedding)   #相似性搜索

    isReuse = 1  # 0为不复用，1为计划复用，2为响应复用
    if score < 0.75:
        isReuse = 0
    elif 0.75 <= score < 0.95:
        isReuse = 1
    else:
        isReuse = 2

    print(f"\n====isReuse====\n\n isReuse:{isReuse} score:{score} \n---------")

    if enable_reuse and isReuse == 2:
        response=cached_data["response"]
        print(f"\n====Execution result====\n\n {response} \n---------")
    else:
        agents = await build_agents(enable_tools=True)
        general_agent = agents["general_agent"]
        coder = agents["coder"]
        web_search_agent = agents["web_search_agent"]
        web_fetch_agent = agents["web_fetch_agent"]
        navigation_agent = agents["navigation_agent"]
        reviewer = agents["reviewer"]
        plan_generator = agents["plan_generator"]

        model_client = create_model_client()

        if isReuse == 0 or not enable_reuse:
            selector_prompt = """
                    Now select the agent to execute the task.
                    {roles}
                    Current conversation context:
                    {history}
                    Read the above conversation and then choose one agent from {participants} to perform the next task.
                    Ensure that the chosen agent is a professional agent with skills suitable for this conversation; if no such specialized agent is available, select General_agent.
                    If any agent has mentioned REVIEW, then select Reviewer.
                    Select only one agent.
                """
            # 终止条件
            text_termination = TextMentionTermination("TERMINATE")
            max_message_termination = MaxMessageTermination(max_messages=25)
            termination = text_termination | max_message_termination

            # 创建团队
            team = SelectorGroupChat(
                [reviewer, coder, general_agent, plan_generator, navigation_agent, web_search_agent, web_fetch_agent], # 可以考虑加一个reviewer之类的，手动增加来回试错,reuse过程中不进行review）
                model_client=model_client,
                termination_condition=termination,
                selector_prompt=selector_prompt,
                allow_repeated_speaker=True,  # 允许代理连续多轮发言。
            )

            stream = team.run_stream(task=task)
            history=[]

            async for message in stream:
                if isinstance(message, TaskResult):
                    print("停止原因：", message.stop_reason)
                else:
                    json_message=message.dump()
                    print(f"Speaker: {json_message["source"]}")
                    print(f"{json_message["content"]}\n--------------\n\n")
                    #print(json_message)
                    history.append(json_message)

            #print("-------------History:--------------")
            #print(history)

            # # =============== 提取最后一个对话作为输出 ===============
            exec_result = history[-1]["content"]
            print(f"\n====Execution result====\n\n {exec_result} \n---------")

            team_state = await team.save_state()

            # ================== 群聊结束后交给 OutputSummarizer 生成计划 ==================
            print("\n=== Generating Plan ===\n")

            team_plan = RoundRobinGroupChat(
                [plan_generator],
                # 可以考虑加一个reviewer之类的，手动增加来回试错,reuse过程中不进行review）
                model_client=model_client,
                termination_condition=termination
            )

            await team_plan.load_state(team_state)

            stream = team_plan.run_stream(task="The task has been completed. Based on the conversation history, use the PlanGenerator to "
                                          "summarize only the essential step-by-step execution plan that can "
                                          "be directly followed by the multi-agent system to reproduce the solution "
                                          "for the same or similar request in the future. Exclude any analysis or unnecessary dialogue. "
                                          "Provide only the core actionable plan.")

            history_plan = []

            async for message in stream:
                if isinstance(message, TaskResult):
                    print("停止原因：", message.stop_reason)
                else:
                    json_message = message.dump()
                    print(f"Speaker: {json_message["source"]}")
                    print(f"{json_message["content"]}\n--------------\n\n")
                    # print(json_message)
                    history_plan.append(json_message)

            # sum_reply = plan_generator.generate_reply(messages=history)
            plan_text = str([m["content"] for m in history_plan if m.get("source") == "PlanGenerator" and m.get("content")])
            print("\n=== OutputSummarizer PLAN ===\n", plan_text)

            if enable_reuse:
                semantic_cache.save_to_cache(task,exec_result,plan_text)  #存储响应和计划

            print("\n=== OutputSummarizer PLAN ===\n", plan_text)

        elif isReuse == 1:
            plan_text=cached_data["plan"]

            selector_prompt = """
                            Strictly follow the plan and select the agent to execute the task.
                            {roles}
                            Current conversation context:
                            {history}
                            Read the above conversation and then choose one agent from {participants} to perform the next task.
                            Ensure that the chosen agent is a professional agent with skills suitable for this conversation; if no such specialized agent is available, select General_agent.
                            If any agent has mentioned REVIEW, then select ReviewerAgent.
                            Select only one agent.
                        """
            # 终止条件
            text_termination = TextMentionTermination("TERMINATE")
            max_message_termination = MaxMessageTermination(max_messages=25)
            termination = text_termination | max_message_termination

            # 创建团队
            team = SelectorGroupChat(
                [reviewer,coder, general_agent, navigation_agent, web_fetch_agent, web_search_agent],  # 可以考虑加一个reviewer之类的，手动增加来回试错,reuse过程中不进行review）
                model_client=model_client,
                termination_condition=termination,
                selector_prompt=selector_prompt,
                allow_repeated_speaker=True,  # 允许代理连续多轮发言。
            )

            stream = team.run_stream(task=task
                                          +"/// The following is an execution plan for a similar request. This plan was derived from a previous successful completion of a similar task by the system. Strictly follow this plan to accomplish the current request: "
                                          + plan_text)
            history = []

            async for message in stream:
                if isinstance(message, TaskResult):
                    print("停止原因：", message.stop_reason)
                else:
                    json_message = message.dump()
                    print(f"Speaker: {json_message["source"]}")
                    print(f"{json_message["content"]}\n--------------\n\n")
                    # print(json_message)
                    history.append(json_message)

            # print("-------------History:--------------")
            # print(history)

            # # =============== 提取最后一个对话作为输出 ===============
            exec_result = history[-1]["content"]
            print(f"\n====Execution result====\n\n {exec_result} \n---------")

    end = time.time()

    print(f"代码运行耗时: {end - start:.2f} 秒")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AutoReuse runner")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The task description for the agent"
    )
    parser.add_argument(
        "--enable_reuse",
        action="store_true",
        help="Enable reuse flag (default: False)"
    )
    args = parser.parse_args()

    asyncio.run(run_agent(task=args.task, enable_reuse=args.enable_reuse))