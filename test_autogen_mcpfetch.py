# 代码1
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core import CancellationToken
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

import autogen
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json, GroupChat, GroupChatManager
import os
from autogen import GroupChat
import json
from autogen.retrieve_utils import TEXT_FORMATS


######################################################################
config_list_mistral = [
    {
        "base_url": "http://localhost:11434/v1",
        "api_key": "sk-111111111111",
        "model": "llama2:13b"
    }
]

config_list_codellama = [
    {
        "base_url": "http://localhost:11434/v1",
        "api_key": "sk-111111111111",
        "model": "llama2:13b"
    }
]
######################################################################
llm_config_mistral={
    "config_list": config_list_mistral,
}

llm_config_codellama={
    "config_list": config_list_codellama,
}
######################################################################
llm_config_mistral = llm_config_mistral
llm_config_codellama = llm_config_codellama
######################################################################

async def main():
    # Setup the MCP fetch server parameters
    fetch_mcp_server = StdioServerParams(command="node", args=["mcp-server-fetch"])

    # Get the fetch tool from the MCP server
    tools = await mcp_server_tools(fetch_mcp_server)

    # Create fetch agent with the MCP fetch tool
    # fetch_agent = AssistantAgent(
    #     name="content_fetcher",
    #     model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
    #     tools=tools,  # The MCP fetch tool will be included here
    #     system_message="你是一个网页内容获取助手。使用fetch工具获取网页内容。"
    # )

    fetch_agent = autogen.AssistantAgent(
        name="content_fetcher",
        llm_config=llm_config_codellama,
        tools=tools,
        system_message="你是一个网页内容获取助手。使用fetch工具获取网页内容。"
    )

    # Create rewriter Agent (unchanged)
    rewriter_agent = autogen.AssistantAgent(
        name="content_rewriter",
        llm_config=llm_config_codellama,
        tools=tools,
        system_message="""你是一个内容改写专家。将提供给你的网页内容改写为科技资讯风格的文章。
                科技资讯风格特点：
                1. 标题简洁醒目
                2. 开头直接点明主题
                3. 内容客观准确但生动有趣
                4. 使用专业术语但解释清晰
                5. 段落简短，重点突出

                当你完成改写后，在响应的最后回复TERMINATE。"""
    )

    # Set up termination condition and team (unchanged)
    termination = TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat([fetch_agent, rewriter_agent], termination_condition=termination)

    # Run the workflow (unchanged)
    result = await team.run(
        task="获取https://www.aivi.fyi/llms/introduce-Claude-3.7-Sonnet的内容，然后将其改写为科技资讯风格的文章",
        cancellation_token=CancellationToken()
    )

    print("\n最终改写结果：\n")
    print(result.messages[-1].content)
    return result


# This is the correct way to run async code in a Python script
if __name__ == "__main__":
    asyncio.run(main())
