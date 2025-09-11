import asyncio

import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen_agentchat.ui import Console
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

# ================== 基础配置 ==================
config_list_codellama = [
    {
        "base_url": "http://localhost:11434/v1",
        "api_key": "sk-111111111111",
        "model": "llama2:13b"
    }
]
llm_config_codellama = {"config_list": config_list_codellama}


# STDIO MCPServer服务的配置
calculator_mcp_server = StdioServerParams(
    command="python",
    args=["calculator_server.py"]
)


# 运行智能体
async def run_agent(task: str):
    # 获取MCPServer提供的工具
    tools = await mcp_server_tools(calculator_mcp_server)

    # 创建智能体
    agent = autogen.AssistantAgent(
        name="my_agent",
        llm_config=llm_config_codellama,
        tools=tools,
        reflect_on_tool_use=True,
        system_message="你是一个算术大师，使用提供的工具进行算术运算",
    )

    """使用 Console 实时打印所有消息。"""
    await Console(
        agent.run_stream(task=task),
        output_stats=True,  # 启用统计信息打印。
    )


if __name__ == '__main__':
    task = "1 + 1 等于？"
    asyncio.run(run_agent(task=task))