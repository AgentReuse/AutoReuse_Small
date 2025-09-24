import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
import yaml
from autogen_core.models import ChatCompletionClient



async def main():
    with open("model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    model_client = ChatCompletionClient.load_component(model_cfg)
    agent_a = AssistantAgent(
        "A",
        model_client=model_client,
        system_message="You are a helpful assistant.",
    )
    agent_b = AssistantAgent(
        "B",
        model_client=model_client,
        system_message="Provide feedback on the input, if your feedback has been addressed, "
        "say 'APPROVE', otherwise provide a reason for rejection.",
    )
    agent_c = AssistantAgent(
        "C", model_client=model_client, system_message="Translate the final product to Korean."
    )

    # Create a loop graph with conditional exit: A -> B -> C ("APPROVE"), B -> A (otherwise).
    builder = DiGraphBuilder()
    builder.add_node(agent_a).add_node(agent_b).add_node(agent_c)
    builder.add_edge(agent_a, agent_b)

    # Create conditional edges using strings
    builder.add_edge(agent_b, agent_c, condition=lambda msg: "APPROVE" in msg.to_model_text())
    builder.add_edge(agent_b, agent_a, condition=lambda msg: "APPROVE" not in msg.to_model_text())

    builder.set_entry_point(agent_a)
    graph = builder.build()

    # Create a GraphFlow team with the directed graph.
    team = GraphFlow(
        participants=[agent_a, agent_b, agent_c],
        graph=graph,
        termination_condition=MaxMessageTermination(20),  # Max 20 messages to avoid infinite loop.
    )

    # Run the team and print the events.
    async for event in team.run_stream(task="Write a short poem about AI Agents."):
        print(event)


asyncio.run(main())