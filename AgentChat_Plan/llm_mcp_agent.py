# agents.py
import os
import asyncio
from typing import Dict, Optional

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.ui import Console
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.base import TaskResult

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import (
    StdioServerParams,
    mcp_server_tools,
    SseServerParams
)

# --------- 可选：从环境变量读取模型/服务端点，便于生产配置 ---------
DEFAULT_MODEL = "qwen2.5:14b" #os.getenv("AGENT_MODEL", "qwen2.5:14b")
DEFAULT_BASE_URL = "http://localhost:11434/v1" #os.getenv("AGENT_BASE_URL", "http://localhost:11434/v1")
DEFAULT_API_KEY = "NULL" #os.getenv("AGENT_API_KEY", "NULL")

def create_model_client(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> OpenAIChatCompletionClient:
    """创建并返回模型客户端；允许外部覆盖默认配置。"""
    return OpenAIChatCompletionClient(
        model=model or DEFAULT_MODEL,
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key or DEFAULT_API_KEY,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
    )

# --------- MCP 工具的惰性加载：封装为函数，避免顶层 await ---------
async def build_mcp_tools():
    """异步创建并返回所需的 MCP 工具字典。"""
    # STDIO MCP servers
    calculator_mcp_server = StdioServerParams(
        command="python", args=["mcp/calculator_server.py"]
    )
    web_search_mcp_server = StdioServerParams(
        command="python", args=["mcp/web_search_server.py"]
    )
    web_fetch_mcp_server = StdioServerParams(
        command="python", args=["mcp/web_fetch_server.py"]
    )
    # 高德地图 SSE
    gaode_server_params = SseServerParams(
        url="https://mcp.amap.com/sse?key=c46c2b1b530d3d92eb4e3dfb3da32a60"
    )

    # 实际加载（注意这里需要 await）
    mcp_tool_calculator = await mcp_server_tools(calculator_mcp_server)
    mcp_tool_web_search = await mcp_server_tools(web_search_mcp_server)
    mcp_tool_web_fetch  = await mcp_server_tools(web_fetch_mcp_server)
    try:
        mcp_tool_gaodemap   = await mcp_server_tools(gaode_server_params)
    except Exception as e:
        mcp_tool_gaodemap = None
        print("!!!!!!!!MCP:GaoDe Error!!!!!!!!")
        print(e)

    return {
        "calculator": mcp_tool_calculator,
        "web_search": mcp_tool_web_search,
        "web_fetch": mcp_tool_web_fetch,
        "gaode": mcp_tool_gaodemap,
    }

# --------- Agent 工厂：对外暴露，其他文件直接调用 ---------
async def build_agents(
    model_client: Optional[OpenAIChatCompletionClient] = None,
    enable_tools: bool = True,
) -> Dict[str, AssistantAgent]:
    """
    创建并返回你定义的所有 Agents（字典）。无副作用，供外部 import 使用。
    - model_client: 可传入外部自定义的模型客户端；不传则使用默认。
    - enable_tools: 是否加载并注入 MCP 工具（生产可按需关闭）。
    """
    model_client = model_client or create_model_client()

    tools = {"calculator": None, "web_search": None, "web_fetch": None, "gaode": None}
    if enable_tools:
        tools = await build_mcp_tools()

    reviewer = AssistantAgent(
        name="Reviewer",
        model_client=model_client,
        system_message=(
            """You are ReviewerAgent. After another agent completes a task, check whether the output correctly  
                fulfills the user’s request. Do not question the professional quality. Only verify that the output 
                matches the requirements. If the output is satisfactory, you must organize and present that output 
                in full, without omission or truncation, as the final result of this request and reply TERMINATE.  
                If not, list the specific issues or mismatches.
                """
        ),
    )

    plan_generator = AssistantAgent(
        name="PlanGenerator",
        model_client=model_client,
        system_message=(
            """
                You are a planning agent. After the group chat ends, read the full conversation history and extract only 
                the steps where real progress was made. Generate a reproducible plan that the multi-agent system can follow 
                for similar future requests. Always determine the correct agent names from the history (sources) instead of 
                inventing new ones. When assigning tasks, do not include specific parameters or detailed values—abstract 
                them so the same plan can be reused. Break down the task into smaller subtasks and assign them clearly in the following format:
                <agent> : <abstract task description>
                <agent> : <abstract task description>
                …
                Ensure the plan is concise, executable, and contains only the essential steps needed for reproduction. Exclude risks, timelines, commentary, or extra details. End with TERMINATE if complete, else CONTINUE.
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
    )

    general_agent = AssistantAgent(
        name="General_agent",
        model_client=model_client,
        system_message=(
            """You are General_agent, a versatile assistant responsible for handling tasks when no specialized agent is available. 
            You should read the conversation context carefully and provide helpful, coherent, 
            and logically consistent outputs. Your role is to fill in gaps, perform general reasoning, 
            answer questions, or provide basic coding or documentation support as needed. 
            Do not attempt to take over specialized responsibilities that belong to domain-specific 
            agents unless explicitly required. Always ensure clarity, conciseness, and accuracy in your responses. 
            After completing the task, output REVIEW to indicate that the result should be checked by ReviewerAgent."""
        ),
    )

    navigation_agent = AssistantAgent(
        name="navigation_agent",
        model_client=model_client,
        tools=tools["gaode"],
        reflect_on_tool_use=True,
        system_message=(
            """
                    You are NavigationAgent, a specialized agent with access to a navigation tool that can plan routes, 
                    retrieve map data, compute distances, directions, and waypoints. 
                    When given a navigation request, you must use the provided navigation tool to produce a plan or set 
                    of directions that are accurate, efficient, and safe. 
                    Always verify that the map data / waypoints used are valid. 
                    Your responses should include:
                    1. A clear route plan or sequence of waypoints.
                    2. Estimated distances or times if possible.
                    
                    After completing the task, output REVIEW to indicate that the result should be checked by ReviewerAgent.
                """
        ),
    )

    web_search_agent = AssistantAgent(
        name="web_search_agent",
        model_client=model_client,
        tools=tools["web_search"],
        reflect_on_tool_use=True,
        system_message=(
            """
                    You are web_search_agent. Your task is to handle web search queries using the `web_search` tool. 
                    Whenever the user provides a query, you must call this tool and return its complete raw JSON response without summarizing, truncating, or altering any fields. 
                    Do not invoke other tools such as `fetch_page`, `follow_links`, or `fetch_dynamic_page`; only the `web_search` tool should be used. 
                    If the search results include multiple items, you should preserve their order and output them exactly as they appear in the tool’s response. 
                    Always wrap the JSON output in a fenced code block marked with ```json for clarity. 
                    After completing the task, output REVIEW to indicate that the result should be checked by ReviewerAgent.
                """
        ),
    )

    web_fetch_agent = AssistantAgent(
        name="web_fetch_agent",
        model_client=model_client,
        tools=tools["web_fetch"],
        reflect_on_tool_use=True,
        system_message=(
            """
                    You are web_fetch_agent. Your role is to retrieve and explore web pages using the tools `fetch_page`, `follow_links`, and `fetch_dynamic_page`. 
                    When the user asks to extract text or metadata from a page, you should rely on `fetch_page`. 
                    If the request is about listing or exploring links, then the proper choice is `follow_links`. 
                    For situations where the content is dynamically generated through JavaScript and requires rendering, you must use `fetch_dynamic_page`. 
                    In all cases, you are expected to return the complete raw JSON response exactly as provided by the tool, without truncating, summarizing, or altering any fields. 
                    The JSON output must always be wrapped in a fenced code block marked with ```json for clarity, and you should never attempt to call tools outside of those assigned to this agent.
                    After completing the task, output REVIEW to indicate that the result should be checked by ReviewerAgent.
                """
        ),
    )

    return {
        "reviewer": reviewer,
        "plan_generator": plan_generator,
        "coder": coder,
        "general_agent": general_agent,
        "navigation_agent": navigation_agent,
        "web_search_agent": web_search_agent,
        "web_fetch_agent": web_fetch_agent,
    }

# （可选）提供一个便捷函数，直接给出“常用团队/终止条件”等构造
def default_terminations(max_turns: int = 12):
    return [
        TextMentionTermination("TERMINATE"),
        MaxMessageTermination(max_turns),
    ]

__all__ = [
    "create_model_client",
    "build_agents",
    "default_terminations",
    # 下面这些导出便于外部拼装团队/控制台
    "SelectorGroupChat",
    "Console",
    "TaskResult",
]
