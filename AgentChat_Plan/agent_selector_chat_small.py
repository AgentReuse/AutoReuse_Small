import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
from Response_reuse import SemanticCache

semantic_cache = SemanticCache(
    embedding_model_path="./m3e-small",
    cache_path="./semantic_cache"
)

# ================== 基础配置 ==================
config_list_codellama = [
    {
        "base_url": "http://localhost:11434/v1",
        "api_key": "sk-111111111111",
        "model": "llama2:13b"
    }
]
llm_config_codellama = {"config_list": config_list_codellama}

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


task = "Write a python script to perform a quick sort."

embedding = semantic_cache.get_embedding(task)             #向量化
similar_question, score, cached_data = semantic_cache.search_similar_query(embedding)   #相似性搜索

isReuse = 1  # 0为不复用，1为计划复用，2为响应复用
if score < 0.75:
    isReuse = 0
elif 0.75 <= score < 0.95:
    isReuse = 1
else:
    isReuse = 2

if isReuse == 0:
    # ================== 定义 Agents ==================
    output_summarizer = autogen.AssistantAgent(
        name="OutputSummarizer",
        llm_config=llm_config_codellama,
        system_message=(
            "You will receive the full chat history after the group chat ends. "
            "Read it and produce a clear, step-by-step EXECUTION PLAN with:"
            " objectives & scope; tasks/milestones; owners (by agent names); "
            "deliverables & acceptance criteria; rough timeline; risks & mitigations. "
            "End with TERMINATE if complete, else CONTINUE."
        ),
    )

    coder = autogen.AssistantAgent(
        name="Coder",
        llm_config=llm_config_codellama,
        is_termination_msg=agent_is_term,
        code_execution_config={"work_dir": "output/coding", "use_docker": False},
    )

    user_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        llm_config=llm_config_codellama,
        is_termination_msg=agent_is_term,
        code_execution_config={"work_dir": "coding", "use_docker": False},
        system_message=(
            "Reply TERMINATE at the end of your response if the task has been solved at full satisfaction. "
            "Otherwise, reply CONTINUE, or explain why not solved yet."
        ),
    )

    groupchat = GroupChat(
        agents=[user_proxy, coder],
        messages=[],
        max_round=12,
        speaker_selection_method=stop_on_terminate_selector,  # ★ 关键：全局终止控制
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config_codellama)

    # 开始群聊
    user_proxy.initiate_chat(manager, message=task)

    history = manager.groupchat.messages  # [{"role":..., "name":..., "content":...}, ...]

    # =============== 提取 Coder 的输出 ===============
    coder_msgs = [m["content"] for m in history if m.get("name") == "Coder" and m.get("content")]

    # 拼接成一个字符串（如果你只需要合并结果）
    coder_output_text = "\n\n".join(coder_msgs)

    print("\n=== Coder Output ===\n", coder_output_text)

    # ================== 群聊结束后交给 OutputSummarizer 生成计划 ==================
    post_instruction = {
        "role": "user",
        "name": "Orchestrator",
        "content": (
            "Please read the entire chat history above and produce a structured EXECUTION PLAN as instructed."
        ),
    }

    sum_reply = output_summarizer.generate_reply(messages=history + [post_instruction])
    plan_text = sum_reply.get("content", "") if isinstance(sum_reply, dict) else str(sum_reply)

    semantic_cache.save_to_cache(task,coder_output_text,plan_text)  #存储响应和计划

    print("\n=== OutputSummarizer PLAN ===\n", plan_text)

elif isReuse == 1:
    plan_text=cached_data["plan"]

    plan_proxy = autogen.UserProxyAgent(
        name="user_proxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        llm_config=llm_config_codellama,
        is_termination_msg=agent_is_term,
        code_execution_config={"work_dir": "coding", "use_docker": False},
        system_message=plan_text,
    )

    coder = autogen.AssistantAgent(
        name="Coder",
        llm_config=llm_config_codellama,
        is_termination_msg=agent_is_term,
        code_execution_config={"work_dir": "output/coding", "use_docker": False},
    )

    groupchat = GroupChat(
        agents=[plan_proxy, coder],
        messages=[],
        max_round=12,
        speaker_selection_method=stop_on_terminate_selector,  # ★ 关键：全局终止控制
    )
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config_codellama)

    # 开始群聊
    plan_proxy.initiate_chat(manager, message=task)

    history = manager.groupchat.messages  # [{"role":..., "name":..., "content":...}, ...]

    # =============== 提取 Coder 的输出 ===============
    coder_msgs = [m["content"] for m in history if m.get("name") == "Coder" and m.get("content")]

    # 拼接成一个字符串（如果你只需要合并结果）
    coder_output_text = "\n\n".join(coder_msgs)

    print("\n=== Coder Output ===\n", coder_output_text)


elif isReuse == 2:
    response=cached_data["response"]
    print("----------------------RESPONSE-----------------------------")
    print(response)
