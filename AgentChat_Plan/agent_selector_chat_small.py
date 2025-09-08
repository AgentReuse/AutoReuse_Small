import time
import json
import re
import autogen
from Response_reuse import SemanticCache
from transit_intent import load_models, predict

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

def fill_plan_keep_placeholders(template, entities):
    if isinstance(entities, str):
        try:
            entities = json.loads(entities)
        except json.JSONDecodeError:
            entities = {}
    entities = entities or {}
    pattern = re.compile(r"\{\s*(?P<key>\w+)\s*:\s*(?P<default>[^}]*)\}")
    def _replacer(m):
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
            return content.strip().strip("\"'")
        return content[colon_pos+1:].strip().strip("\"'")
    return re.sub(r"\{([^{}]+)\}", replacer, s)

# ========== LLM 配置 ==========
llm_config = {
    "config_list": [
        {
            "base_url": "http://localhost:11434/v1",
            "api_key": "sk-111111111111",  # dummy key
            "model": "llama2:13b"
        }
    ]
}

# ========== 构建 Agents ==========
plan_provider = autogen.AssistantAgent(
    name="PlanProvider",
    system_message="You are good at condensing user input into concise, structured, and information-dense task descriptions. Note: Your responses should be highly summarized, typically no more than 30 words. Your generated plan should include a keyword, which is replaceable. By substituting the original keyword with another one, the new plan should remain reusable. At the same time, the originally generated plan itself should also have a high degree of reusability. In the task description you generate, the keywords clearly stated in the input must be included and enclosed in curly braces ({}). When mentioning an entity value in your output sentence, wrap it with curly braces in the format {entity_type:entity_value}. For example, if the entity is {'transport_mode': 'train', 'source': 'jfk airport', 'destination': 'san francisco', 'date': 'next monday'}, you must refer to san francisco as {'destination': 'san francisco'} in your response.",
    llm_config=llm_config
)

info_retriever = autogen.AssistantAgent(
    name="InfoRetriever",
    system_message="You are good at retrieving knowledge, examples and data related to the task. When necessary, you can call the search_web tool. The above plan is a proven and feasible plan. You only need to follow it step by step, without overthinking, without engaging in divergent thinking, without additional discussion, and simply execute the plan.",
    llm_config=llm_config,
    tools=[search_web]
)

analyst = autogen.AssistantAgent(
    name="Analyst",
    system_message="You are good at conducting clear and organized analyses of given tasks or information, and can call on the analyze_data tool to assist in making judgments. The above plan is a proven and feasible plan. You only need to follow it step by step, without overthinking, without engaging in divergent thinking, without additional discussion, and simply execute the plan.",
    llm_config=llm_config,
    tools=[analyze_data]
)

output_summarizer = autogen.AssistantAgent(
    name="OutputSummarizer",
    system_message="You do not directly engage in communication with other agents. You only need to make a systematic summary of the outputs given by other team members in the current context, which should be organized and easy to understand. ",
    llm_config=llm_config
)

groupchat = autogen.GroupChat(
    agents=[plan_provider, info_retriever, analyst, output_summarizer],
    messages=[],
    max_round=6
)
manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

# ========== 主流程（替代 Chainlit） ==========
def run_chat(user_text: str):
    start_time = time.time()

    # intent 识别
    embedding = semantic_cache.get_embedding(user_text)
    similar_question, score, cached_data = semantic_cache.search_similar_query(embedding)
    cached_intent = json.loads(cached_data["intent"]) if cached_data else None

    load_models(intent_dir="transit_intent/bert_intent_model",
                slot_dir="transit_intent/bert_slot_model")
    intent = predict(user_text)
    semantic_cache.save_to_cache(user_text, None, None, intent)

    # 这里保留你的复用逻辑
    isReuse = 0  # 可以改成1/2测试

    if isReuse == 0:
        plan_resp = plan_provider.generate_reply(messages=[{"content": str(user_text) + str(intent)}])
        new_plan_with_braces = plan_resp["content"]
        new_plan_without_braces = clean_braces(new_plan_with_braces)
        semantic_cache.save_to_cache(user_text, None, new_plan_with_braces)

        print("\n[Generated Plan]:", new_plan_without_braces)
        manager.groupchat.messages.append({"role": "user", "content": new_plan_without_braces})
        manager.run_chat()

    elif isReuse == 1:
        plan = clean_braces(fill_plan_keep_placeholders(cached_data["plan"], intent["entities"]))
        print("\n[Reused Plan]:", plan)
        manager.groupchat.messages.append({"role": "user", "content": plan})
        manager.run_chat()

    elif isReuse == 2:
        response = cached_data["response"]
        print("\n[Reused Response]:", response)

    print(f"\n⏱ Delay = {time.time()-start_time:.2f}s")

# ========== 示例运行 ==========
if __name__ == "__main__":
    user_input = "Is the train from JFK Airport to San Francisco running next Monday?"
    run_chat(user_input)
