import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os

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
    code_execution_config={"work_dir":"output/coding", "use_docker":False},
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    llm_config=llm_config_codellama,
    is_termination_msg=agent_is_term,
    code_execution_config={"work_dir":"coding", "use_docker":False},
    system_message=(
        "Reply TERMINATE at the end of your response if the task has been solved at full satisfaction. "
        "Otherwise, reply CONTINUE, or explain why not solved yet."
    ),
)

# ================== 仅让 user_proxy 和 coder 参加群聊 ==================
task = "Write a python script to perform a quick sort."

groupchat = GroupChat(
    agents=[user_proxy, coder],
    messages=[],
    max_round=12,
    speaker_selection_method=stop_on_terminate_selector,  # ★ 关键：全局终止控制
)
manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config_codellama)

# 开始群聊
user_proxy.initiate_chat(manager, message=task)

# ================== 群聊结束后交给 OutputSummarizer 生成计划 ==================
history = manager.groupchat.messages  # [{"role":..., "name":..., "content":...}, ...]

post_instruction = {
    "role": "user",
    "name": "Orchestrator",
    "content": (
        "Please read the entire chat history above and produce a structured EXECUTION PLAN as instructed."
    ),
}

sum_reply = output_summarizer.generate_reply(messages=history + [post_instruction])
plan_text = sum_reply.get("content", "") if isinstance(sum_reply, dict) else str(sum_reply)

print("\n=== OutputSummarizer PLAN ===\n", plan_text)

os.makedirs("output", exist_ok=True)
with open("output/plan_summary.txt", "w", encoding="utf-8") as f:
    f.write(plan_text)
