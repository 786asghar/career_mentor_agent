
import os
import chainlit as cl
from typing import cast
from dotenv import load_dotenv
from agents import Agent, Runner, RunConfig, RunContextWrapper, AsyncOpenAI, OpenAIChatCompletionsModel

load_dotenv()

def setup_config():
    external_client = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-1.5-flash",
        openai_client=external_client,
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )

    # Setup Agents
    career_agent = Agent(
        name="career_agent",
        instructions="You career the user's suggests fields",
        handoff_description="A career the sugggests field",
        model=model
    )

    skill_agent = Agent(
        name="skill_agent",
        instructions="You shows skill-building plans",
        handoff_description="A skill-building plans",
        model=model
    )

    jobs_agent = Agent(
        name="jobs_agent",
        instructions="You shares real-world job roles",
        handoff_description="A shares real-world job roles",
        model=model
    )

    # triage Agent
    triage_agent = Agent(
        name="triage_agent",
        instructions=(
            "You are a Guide students through career exploration. You use the tools given to you to career."
            "If asked for multiple agents support, you call the relevant tools in order."
            "You never studenet career on your own, you always use the provided tools."
        ),
        tools=[
            career_agent.as_tool(
                tool_name="career_exploration",
                tool_description="Guide students through career exploration",
            ),
            skill_agent.as_tool(
                tool_name="skill_building_plans",
                tool_description="Show skill-building plans for students",
            ),
            jobs_agent.as_tool(
                tool_name="real_world_job_roles",
                tool_description="Share real-world job roles and responsibilities",
            ),
        ],
        model=model
    )
    
    return triage_agent, config


@cl.on_chat_start
async def start():
    triage_agent, config = setup_config()
    cl.user_session.set("triage_agent", triage_agent)
    cl.user_session.set("config", config)
    cl.user_session.set("chat_history", [])
    await cl.Message(content="Wellcome to Students Career Mentor").send()


@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""
    # Send a thinking message
    msg = cl.Message(content="Thinking...")
    await msg.send()
    
    triage_agent = cast(Agent, cl.user_session.get("triage_agent"))
    config = cast(RunConfig, cl.user_session.get("config"))

    # Retrieve the chat history from the session.
    history = cl.user_session.get("chat_history") or []

    # Append the user's message to the history.
    history.append({"role": "user", "content": message.content})

    result = await Runner.run(triage_agent, history, run_config=config)

    response_content = result.final_output

    # Update the thinking message with the actual response
    msg.content = response_content
    await msg.update()

    history.append({"role": "assistant", "content": response_content})

    cl.user_session.set("chat_history", history)

    print(f"History: {history}")