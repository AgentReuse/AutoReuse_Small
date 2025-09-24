from typing import List, cast

import chainlit as cl
import yaml

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ModelClientStreamingChunkEvent, TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ChatCompletionClient


# —— 工具函数 —— #
@cl.step(type="tool")
async def query_train_ticket(date: str, origin: str, destination: str) -> str:
    return f"✅ 已为您购买 {date} 从 {origin} → {destination} 的火车票，车次 G1234，座位 5A。"


@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    with open("model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    model_client = ChatCompletionClient.load_component(model_cfg)

    plan_agent = AssistantAgent(
        name="PlanAgent",
        system_message=(
            "你是一个计划生产者。"
            "收到用户请求后，请输出一个详细的“行动计划”，按序号列出每一步该做什么。"
        ),
        model_client=model_client,
        model_client_stream=True,         # 启用流式输出
        reflect_on_tool_use=False,        # 计划阶段不需要调用工具
    )

    exec_agent = AssistantAgent(
        name="ExecutorAgent",
        system_message=(
            "你是一个计划执行者。"
            "收到 PlanAgent 输出的行动计划后，"
            "请按步骤调用相应工具（query_train_ticket）执行计划，"
            "并在每一步后附上工具返回的结果。"
        ),
        tools=[query_train_ticket],
        model_client=model_client,
        model_client_stream=True,
        reflect_on_tool_use=True,         # 执行时要反思工具输出
    )

    cl.user_session.set("plan_agent", plan_agent)  # type: ignore
    cl.user_session.set("exec_agent", exec_agent)  # type: ignore


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="买火车票",
            message="帮我买一张明天从杭州到宁波的火车票",
        ),
        cl.Starter(
            label="查天气",
            message="帮我查一下明天上海闵行区的天气怎么样",
        ),
    ]


@cl.on_message  # type: ignore
async def chat(message: cl.Message) -> None:
    user_text = message.content

    plan_agent = cast(AssistantAgent, cl.user_session.get("plan_agent"))  # type: ignore
    exec_agent = cast(AssistantAgent, cl.user_session.get("exec_agent"))  # type: ignore

    plan_msg = cl.Message(content="")
    plan_content = ""
    async for evt in plan_agent.on_messages_stream(
        messages=[TextMessage(content=user_text, source="user")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(evt, ModelClientStreamingChunkEvent):
            await plan_msg.stream_token(evt.content)
            plan_content += evt.content
        elif isinstance(evt, Response):
            await plan_msg.send()

    exec_msg = cl.Message(content="")
    async for evt in exec_agent.on_messages_stream(
        messages=[TextMessage(content=plan_content, source="assistant")],
        cancellation_token=CancellationToken(),
    ):
        if isinstance(evt, ModelClientStreamingChunkEvent):
            await exec_msg.stream_token(evt.content)
        elif isinstance(evt, Response):
            await exec_msg.send()
