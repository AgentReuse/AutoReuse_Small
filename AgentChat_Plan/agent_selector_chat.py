from typing import List, Sequence, cast
import time
import chainlit as cl
import yaml
from Response_reuse import SemanticCache
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.messages import TextMessage, ModelClientStreamingChunkEvent, BaseAgentEvent, BaseChatMessage
from autogen_core.models import ChatCompletionClient
from autogen_core import CancellationToken
import re
import json
from typing import Union, Dict

# Example usage in another script:
from transit_intent import load_models, predict

#初始化
semantic_cache = SemanticCache(
    embedding_model_path="./m3e-small",
    cache_path="./semantic_cache"
)

import os

os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

@cl.step(type="tool")
async def search_web(query: str) -> str:
    return f"🌐 检索结果：'{query}' 的最新网页摘要如下……"

@cl.step(type="tool")
async def analyze_data(data: str) -> str:
    return f"📊 针对数据'{data}'的初步分析结果：……"


def selector_func(messages: Sequence[BaseAgentEvent | BaseChatMessage]) -> str | None:
    MAX_TURNS = 6
    print("message_len")
    print(len(messages))
    if len(messages) == 1:
        return "PlanProvider"
    if len(messages) == MAX_TURNS - 1:
        return "OutputSummarizer"
    return None


def fill_plan_keep_placeholders(template: str,
                                entities: Union[str, Dict[str, str], None]) -> str:
    """
    用 entities 替换模板中形如 {key: default} 的槽位，并保留 {key: value} 格式
    :param template: 原计划字符串，例如 "Check the {transport_mode: train} status ..."
    :param entities: dict 或 JSON 字符串，例如 {"transport_mode":"plane", ...}
    :return: 替换后的字符串
    """
    # 兼容 JSON 字符串
    if isinstance(entities, str):
        try:
            entities = json.loads(entities)
        except json.JSONDecodeError as e:
            raise ValueError(f"entities 不是有效的 JSON：{e}")
    entities = entities or {}

    # 匹配 { key : default }
    pattern = re.compile(r"\{\s*(?P<key>[A-Za-z0-9_]+)\s*:\s*(?P<default>[^}]*)\}")

    def _replacer(m: re.Match) -> str:
        key = m.group("key")
        default_val = m.group("default").strip()
        new_val = entities.get(key, default_val)
        return f"{{{key}: {new_val}}}"

    return pattern.sub(_replacer, template)


def clean_braces(s):
    def replacer(match):
        content = match.group(1).strip()
        colon_pos = content.find(':')
        if colon_pos == -1:
            # 没有冒号，仅去除首尾引号
            content = re.sub(r"^['\"‘“]+|['\"’”]+$", '', content)
            return content.strip()
        # 有冒号，删除冒号及其前所有内容（包括冒号和首尾引号）
        after_colon = content[colon_pos+1:].strip()
        after_colon = re.sub(r"^['\"‘“]+|['\"’”]+$", '', after_colon)
        return after_colon.strip()
    # 替换所有 {...}
    return re.sub(r"\{([^{}]+)\}", replacer, s)


@cl.on_chat_start  # type: ignore
async def start_chat() -> None:
    with open("model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    model_client = ChatCompletionClient.load_component(model_cfg)

    plan_provider = AssistantAgent(
        name="PlanProvider",
        system_message="You are good at condensing user input into concise, structured, and information-dense task descriptions. Note: Your responses should be highly summarized, typically no more than 30 words. Your generated plan should include a keyword, which is replaceable. By substituting the original keyword with another one, the new plan should remain reusable. At the same time, the originally generated plan itself should also have a high degree of reusability. In the task description you generate, the keywords clearly stated in the input must be included and enclosed in curly braces ({}). When mentioning an entity value in your output sentence, wrap it with curly braces in the format {entity_type:entity_value}. For example, if the entity is {'transport_mode': 'train', 'source': 'jfk airport', 'destination': 'san francisco', 'date': 'next monday'}, you must refer to san francisco as {'destination': 'san francisco'} in your response.",
        model_client=model_client,
        model_client_stream=True,
        reflect_on_tool_use=False,
    )

    info_retriever = AssistantAgent(
        name="InfoRetriever",
        system_message="You are good at retrieving knowledge, examples and data related to the task. When necessary, you can call the search_web tool. The above plan is a proven and feasible plan. You only need to follow it step by step, without overthinking, without engaging in divergent thinking, without additional discussion, and simply execute the plan.",
        tools=[search_web],
        model_client=model_client,
        model_client_stream=True,
        reflect_on_tool_use=True,
    )

    analyst = AssistantAgent(
        name="Analyst",
        system_message="You are good at conducting clear and organized analyses of given tasks or information, and can call on the analyze_data tool to assist in making judgments. The above plan is a proven and feasible plan. You only need to follow it step by step, without overthinking, without engaging in divergent thinking, without additional discussion, and simply execute the plan.",
        tools=[analyze_data],
        model_client=model_client,
        model_client_stream=True,
        reflect_on_tool_use=True,
    )

    output_summarizer = AssistantAgent(
        name="OutputSummarizer",
        system_message="You do not directly engage in communication with other agents. You only need to make a systematic summary of the outputs given by other team members in the current context, which should be organized and easy to understand. ",
        model_client=model_client,


        model_client_stream=True,
        reflect_on_tool_use=False,
    )

    team = SelectorGroupChat(
        [plan_provider],
        model_client=model_client,
        # selector_func=selector_func,  # 首尾定序，中间自由
        max_turns=6,
    )

    cl.user_session.set(plan_provider, plan_provider)
    cl.user_session.set("team", team)  # type: ignore


@cl.set_starters  # type: ignore
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="Ticket",
            message="Is the train from JFK Airport to San Francisco running next Monday?"
        ),
    ]


@cl.on_message
async def chat(message: cl.Message) -> None:
    user_text = message.content
    start_time = time.time()
    embedding = semantic_cache.get_embedding(user_text)             #向量化
    similar_question, score, cached_data = semantic_cache.search_similar_query(embedding)   #相似性搜索
    cached_intent = json.loads(cached_data["intent"]) if cached_data is not None else None
    print(score)
    plan_provider = cl.user_session.get("plan_provider")
    new_plan_with_braces = ""

    load_models(intent_dir="transit_intent/bert_intent_model",
                slot_dir="transit_intent/bert_slot_model")
    intent = predict(user_text)
    semantic_cache.save_to_cache(user_text, None, None, intent)
    print(intent)

    input = str(user_text) + str(intent)

    team: SelectorGroupChat = cl.user_session.get("team")
    msg = cl.Message(content="")
    team = cast(SelectorGroupChat, cl.user_session.get("team"))

    # isReuse = 1  # 0为不复用，1为计划复用，2为响应复用
    #
    # if score < 0.75:
    #     isReuse = 0
    # elif 0.75 <= score < 0.95:
    #     if intent["intent"]["label"] == cached_intent["intent"]["label"]:
    #         isReuse = 1
    #     else:
    #         isReuse = 0
    # else:
    #     isReuse = 2

    isReuse = 0

    if isReuse == 0:
        async for evt in plan_provider.on_messages_stream(
                messages=[TextMessage(content=input, source="input")],
                cancellation_token=CancellationToken(),
        ):
            if isinstance(evt, ModelClientStreamingChunkEvent):
                new_plan_with_braces += evt.content

        new_plan_without_braces = clean_braces(new_plan_with_braces)

        async for evt in team.run_stream(
                task=new_plan_without_braces,
                cancellation_token=CancellationToken(),
        ):
            semantic_cache.save_to_cache(user_text, None, new_plan_with_braces)
            agent_name = getattr(evt, "source", None) or getattr(getattr(evt, "chat_message", None), "source", None)

            if agent_name == "OutputSummarizer":
                if msg is None:
                    msg = cl.Message(author="OutputSummarizer", content="")
                if hasattr(evt, "content") and isinstance(evt.content, str):
                    await msg.stream_token(evt.content)
                elif hasattr(evt, "content"):
                    await msg.send()
                semantic_cache.save_to_cache(user_text, evt.content, None)
        end_time_0 = time.time()


    elif isReuse == 1:
        plan = clean_braces(fill_plan_keep_placeholders(cached_data["plan"], intent["entities"]))
        async for evt in team.run_stream(
                task=plan,
                cancellation_token=CancellationToken(),
        ):
            agent_name = getattr(evt, "source", None) or getattr(getattr(evt, "chat_message", None), "source", None)

            if agent_name == "OutputSummarizer":
                if msg is None:
                    msg = cl.Message(author="OutputSummarizer", content="")
                if hasattr(evt, "content") and isinstance(evt.content, str):
                    await msg.stream_token(evt.content)
                elif hasattr(evt, "content"):
                    await msg.send()
        end_time_1 = time.time()

    elif isReuse == 2:
        response = cached_data["response"]  # 读取响应
        end_time_2 = time.time()
        print(response)


    if isReuse == 0:
        delay0 = end_time_0-start_time
        print(f"delay0 is {delay0}")
    elif isReuse == 1:
        delay1 = end_time_1 - start_time
        print(f"delay1 is {delay1}")
    else:
        delay2 = end_time_2 - start_time
        print(f"delay2 is {delay2}")



