import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core.models import ModelInfo
from autogen_ext.tools.mcp import SseServerParams, mcp_server_tools

from autogen_ext.models.openai import OpenAIChatCompletionClient



model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash-lite",
        model_info=ModelInfo(
            vision=True,
            function_calling=True,
            json_output=True,
            family="unknown",
            structured_output=True,
        ),
        api_key="AIzaSyB45FYaepy06XwOuk6xd6KoeKrwZsV9qOE",
    )

# 高德地图SSE
gaode_server_params = SseServerParams(
    url="https://mcp.amap.com/sse?key=c46c2b1b530d3d92eb4e3dfb3da32a60"
)


# 运行智能体
async def run_agent(task: str):
    # 获取MCPServer提供的工具
    tools = await mcp_server_tools(gaode_server_params)

    # 创建智能体
    agent = AssistantAgent(
        name="navigation_agent",
        model_client=model_client,
        tools=tools,
        reflect_on_tool_use=True,
        system_message="""
            You are an LBS map expert who can use tools to complete some functions related to maps and geography.
        """,
    )

    """使用 Console 实时打印所有消息。"""
    await Console(
        agent.run_stream(task=task),
        output_stats=True,  # 启用统计信息打印。
    )


if __name__ == '__main__':
    task = ("公共交通出行规划："
            "从浙江省宁波市学府路5号浙江大学宁波科创中心到浙江省杭州市浙江大学紫荆港校区")
    asyncio.run(run_agent(task=task))