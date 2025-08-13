import os
import time
import json
import re
import yaml
from typing import List, Sequence, Union, Dict
import chainlit as cl
from Response_reuse import SemanticCache
import autogen
from transit_intent import load_models, predict

# ========== 配置代理（如有） ==========
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'

# ========== 初始化向量缓存 ==========
semantic_cache = SemanticCache(
    embedding_model_path="./m3e-small",
    cache_path="./semantic_cache"
)

# ========== 工具函数 ==========
async def search_web(query: str) -> str:
    return f"🌐 检索结果：'{query}' 的最新网页摘要如下……"

async def analyze_data(data: str) -> str:
    return f"📊 针对数据'{data}'的初步分析结果：……"

def fill_plan_keep_placeholders(template: str, entities: Union[str, Dict[str, str], None]) -> str:
    if isinstance(entities, str):
        try:
            entities = json.loads(entities)
        except json.JSONDecodeError as e:
            raise ValueError(f"entities 不是有效的 JSON：{e}")
    entities = entities or {}
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
            content = re.sub(r"^['\"‘“]+|['\"’”]+$", '', content)
            return content.strip()
        after_colon = content[colon_pos+1:].strip()
        after_colon = re.sub(r"^['\"‘“]+|['\"’”]+$", '', after_colon)
        return after_colon.strip()
    return re.sub(r"\{([^{}]+)\}", replacer, s)

# ========== LLM 配置 ==========
llm_config = {
    "config_list": [
        {
            "base_url": "http://localhost:11434/v1",
            "api_key": "sk-111111111111",
            "model": "llama2:13b"
        }
    ]
}

@cl.on_chat_start
async def start_chat():
    plan_provider = autogen.AssistantAgent(
        name="PlanProvider",
        system_message="You are good at condensing user input into concise, structured, and information-dense task descriptions...",
        llm_config=llm_config
    )

    info_retriever = autogen.AssistantAgent(
        name="InfoRetriever",
        system_message="You are good at retrieving knowledge...",
        llm_config=llm_config,
        tools=[search_web]
    )

    analyst = autogen.AssistantAgent(
        name="Analyst",
        system_message="You are good at conducting clear and organized analyses...",
        llm_config=llm_config,
        tools=[analyze_data]
    )

    output_summarizer = autogen.AssistantAgent(
        name="OutputSummarizer",
        system_message="You do not directly engage in communication with other agents...",
        llm_config=llm_config
    )

    groupchat = autogen.GroupChat(
        agents=[plan_provider, info_retriever, analyst, output_summarizer],
        messages=[],
        max_round=6
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

    cl.user_session.set("plan_provider", plan_provider)
    cl.user_session.set("manager", manager)
    cl.user_session.set("output_summarizer", output_summarizer)

@cl.on_message
async def chat(message: cl.Message):
    user_text = message.content
    start_time = time.time()
    embedding = semantic_cache.get_embedding(user_text)
    similar_question, score, cached_data = semantic_cache.search_similar_query(embedding)
    cached_intent = json.loads(cached_data["intent"]) if cached_data is not None else None

    load_models(intent_dir="transit_intent/bert_intent_model",
                slot_dir="transit_intent/bert_slot_model")
    intent = predict(user_text)
    semantic_cache.save_to_cache(user_text, None, None, intent)

    isReuse = 0  # 你原来的复用逻辑可以替换这里的值

    plan_provider: autogen.AssistantAgent = cl.user_session.get("plan_provider")
    manager: autogen.GroupChatManager = cl.user_session.get("manager")

    if isReuse == 0:
        plan_resp = plan_provider.generate_reply(messages=[{"content": str(user_text) + str(intent)}])
        new_plan_with_braces = plan_resp["content"]
        new_plan_without_braces = clean_braces(new_plan_with_braces)
        semantic_cache.save_to_cache(user_text, None, new_plan_with_braces)
        manager.groupchat.messages.append({"role": "user", "content": new_plan_without_braces})
        manager.run_chat()
        end_time = time.time()

    elif isReuse == 1:
        plan = clean_braces(fill_plan_keep_placeholders(cached_data["plan"], intent["entities"]))
        manager.groupchat.messages.append({"role": "user", "content": plan})
        manager.run_chat()
        end_time = time.time()

    elif isReuse == 2:
        response = cached_data["response"]
        print(response)
        end_time = time.time()

    print(f"delay is {end_time - start_time:.2f}s")
