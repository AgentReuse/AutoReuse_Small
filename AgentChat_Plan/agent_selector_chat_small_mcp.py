import autogen
import asyncio
import argparse
# from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

from autogen_agentchat.agents import AssistantAgent,UserProxyAgent
from autogen_agentchat.ui import Console
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools, SseServerParams
from autogen_agentchat.teams import SelectorGroupChat

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.base import TaskResult

import time

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

web_search_mcp_server = StdioServerParams(
    command="python",
    args=["mcp/web_search_server.py"]
)

web_fetch_mcp_server = StdioServerParams(
    command="python",
    args=["mcp/web_fetch_server.py"]
)

# 高德地图SSE
gaode_server_params = SseServerParams(
    url="https://mcp.amap.com/sse?key=c46c2b1b530d3d92eb4e3dfb3da32a60"
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

async def run_agent(task: str,enable_reuse: bool):
    start = time.time()

    #先加载mcp工具
    mcp_tool_calculator = await mcp_server_tools(calculator_mcp_server)
    mcp_tool_web_search = await mcp_server_tools(web_search_mcp_server)
    mcp_tool_web_fetch = await mcp_server_tools(web_fetch_mcp_server)
    mcp_tool_gaodemap = await mcp_server_tools(gaode_server_params)

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
    #isReuse = 0

    # ================== 定义 Agents ==================
    reviewer = AssistantAgent(
        name="Reviewer",
        model_client=model_client,
        system_message=(
            """You are ReviewerAgent. After another agent completes a task, check whether the output correctly 
            fulfills the user’s request. Do not question the professional quality—only verify that the output 
            matches the requirements. If the output is satisfactory, organize and present that output in full 
            as the final result of this request and reply TERMINATE. If not, list the specific issues or mismatches."""
        ),
    )

    plan_generator = AssistantAgent(
        name="PlanGenerator",
        model_client=model_client,
        system_message=(
            """
            You are a planning agent. Your job is to analyze the full chat history after the group chat ends, 
            extract only the steps where real progress was made toward solving the task, and rewrite them as a 
            reproducible plan for this multi-agent system. Break down the completed task into smaller actionable 
            subtasks and assign them to the appropriate agents. Use the following format for task assignment:
            <agent> : <task>
            <agent> : <task>
            …
            Ensure the plan is minimal, precise, and directly usable to reproduce the solution in future runs. 
            Do not add unnecessary content such as risks, mitigations, timelines, or commentary. 
            End with TERMINATE if complete, else CONTINUE.
            """
        ),
    )

    coder = AssistantAgent(
        name="Coder",
        model_client=model_client,
        system_message=(
            """You are a highly skilled coder agent responsible for writing, checking, 
            and improving code based on the user’s requests. You must produce correct, 
            efficient, and well-documented code, verify syntax and logic, and point out 
            or fix potential bugs or improvements when necessary. Ensure that your 
            responses are precise, concise, and directly actionable. 
            Always provide complete solutions unless explicitly asked for partial output. 
            After completing the task, output REVIEW to indicate that the result should be checked by ReviewerAgent."""
        ),
        #code_execution_config={"work_dir": "output/coding", "use_docker": False},
    )

    general_agent = AssistantAgent(
        name="General_agent",
        model_client=model_client,
        system_message=
        """You are General_agent, a versatile assistant responsible for handling tasks when no specialized agent is available. 
        You should read the conversation context carefully and provide helpful, coherent, 
        and logically consistent outputs. Your role is to fill in gaps, perform general reasoning, 
        answer questions, or provide basic coding or documentation support as needed. 
        Do not attempt to take over specialized responsibilities that belong to domain-specific 
        agents unless explicitly required. Always ensure clarity, conciseness, and accuracy in your responses. 
        After completing the task, output REVIEW to indicate that the result should be checked by ReviewerAgent.""",
    )

    navigation_agent = AssistantAgent(
        name="navigation_agent",
        model_client=model_client,
        tools=mcp_tool_gaodemap,
        reflect_on_tool_use=True,
        system_message="""
                You are NavigationAgent, a specialized agent with access to a navigation tool that can plan routes, 
                retrieve map data, compute distances, directions, and waypoints. 
                When given a navigation request, you must use the provided navigation tool to produce a plan or set 
                of directions that are accurate, efficient, and safe. 
                Always verify that the map data / waypoints used are valid. 
                Your responses should include:
                1. A clear route plan or sequence of waypoints.
                2. Estimated distances or times if possible.
                
                After completing the task, output REVIEW to indicate that the result should be checked by ReviewerAgent.
            """,
    )

    web_search_agent = AssistantAgent(
        name="web_search_agent",
        model_client=model_client,
        tools=mcp_tool_web_search,
        reflect_on_tool_use=True,
        system_message="""
                You are web_search_agent. Your task is to handle web search queries using the `web_search` tool. 
                Whenever the user provides a query, you must call this tool and return its complete raw JSON response without summarizing, truncating, or altering any fields. 
                Do not invoke other tools such as `fetch_page`, `follow_links`, or `fetch_dynamic_page`; only the `web_search` tool should be used. 
                If the search results include multiple items, you should preserve their order and output them exactly as they appear in the tool’s response. 
                Always wrap the JSON output in a fenced code block marked with ```json for clarity. 
                After completing the task, output REVIEW to indicate that the result should be checked by ReviewerAgent.
            """,
    )

    web_fetch_agent = AssistantAgent(
        name="web_fetch_agent",
        model_client=model_client,
        tools=mcp_tool_web_fetch,
        reflect_on_tool_use=True,
        system_message="""
                You are web_fetch_agent. Your role is to retrieve and explore web pages using the tools `fetch_page`, `follow_links`, and `fetch_dynamic_page`. 
                When the user asks to extract text or metadata from a page, you should rely on `fetch_page`. 
                If the request is about listing or exploring links, then the proper choice is `follow_links`. 
                For situations where the content is dynamically generated through JavaScript and requires rendering, you must use `fetch_dynamic_page`. 
                In all cases, you are expected to return the complete raw JSON response exactly as provided by the tool, without truncating, summarizing, or altering any fields. 
                The JSON output must always be wrapped in a fenced code block marked with ```json for clarity, and you should never attempt to call tools outside of those assigned to this agent.
                After completing the task, output REVIEW to indicate that the result should be checked by ReviewerAgent.
            """,
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

    print(f"\n====isReuse====\n\n {isReuse} \n---------")

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

        # ================== 群聊结束后交给 OutputSummarizer 生成计划 ==================
        print("\n=== Generating Plan ===\n")

        selector_prompt_plan = """
                    The task is already complete. Now select the PlanGenerator agent to summarize the history.
                    """

        stream = team.run_stream(task="The task has been completed. Based on the conversation history, use the PlanGenerator to "
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


    elif isReuse == 2:
        response=cached_data["response"]
        print(f"\n====Execution result====\n\n {response} \n---------")

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