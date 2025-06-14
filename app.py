
import os
import json
from typing import cast, Union, Any
import chainlit as cl
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)
from tools import get_movie_details, poetry_agent

# Load API keys from .env file
load_dotenv()

# --- Configure OpenRouter Client ---
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Disable the agent's built-in tracing
set_tracing_disabled(True)

# Global configuration for chat profiles
PROFILES_CONFIG = [
    {
        "name": "DeepSeek Assistant",
        "markdown_description": "A general purpose AI powered by DeepSeek Chat. Can handle conversational requests and tool calls. **Model: DeepSeek Chat (Tool-capable)**.",
        "model_id": "deepseek/deepseek-chat-v3-0324:free",
        "icon": "public/deepseek.svg",
    },
    {
        "name": "Creative Assistant",
        "markdown_description": "A friendly and creative AI that can write poetry and fetch movie details. **Model: Google Gemini Flash (May not support tool use)**.",
        "model_id": "google/gemini-2.0-flash-exp:free",
        "icon": "public/creative.svg",
    },
    {
        "name": "Technical Analyst",
        "markdown_description": "A more direct and technical AI for analysis and information retrieval. **Model: Mistral Small (May not support tool use)**.",
        "model_id": "mistralai/mistral-small-24b-instruct-2501:free",
        "icon": "public/technical.svg",
    },
]

# --- Chainlit Chat Profiles ---
@cl.set_chat_profiles
async def chat_profiles():
    # These profiles define the selectable models and their descriptions in the UI
    return [
        cl.ChatProfile(
            name=p["name"],
            markdown_description=p["markdown_description"],
            model=p["model_id"], # Pass the model_id as the model argument
            icon=p["icon"],
        )
        for p in PROFILES_CONFIG
    ]

# Helper function to initialize/update the agent and send welcome messages
async def setup_agent_and_welcome(profile_name: str):
    print(f"DEBUG: setup_agent_and_welcome called with requested profile_name: {profile_name}")
    
    current_model_id = None
    current_profile_name = profile_name # Start with the requested name

    # Lookup the model_id from our PROFILES_CONFIG based on profile_name
    for p_config in PROFILES_CONFIG:
        if p_config["name"] == profile_name:
            current_model_id = p_config["model_id"]
            current_profile_name = p_config["name"] # Ensure it's the exact name from config
            print(f"DEBUG: Found profile in config: {current_profile_name}, Model ID: {current_model_id}")
            break

    if current_model_id is None: # If the profile wasn't found in our config
        # Default to the first one in PROFILES_CONFIG if selected profile not found
        current_model_id = PROFILES_CONFIG[0]["model_id"]
        current_profile_name = PROFILES_CONFIG[0]["name"]
        print(f"DEBUG: Profile '{profile_name}' not found. Falling back to default: {current_profile_name}, Model ID: {current_model_id}")
        await cl.Message(
            content=f"**Warning**: Could not precisely match selected chat profile '{profile_name}'. Defaulting to '{current_profile_name}' powered by `{current_model_id}`."
        ).send()

    # DEBUG: Print the model that will actually be used for agent initialization
    print(f"DEBUG: Agent being initialized with model: {current_model_id} for profile: {current_profile_name}")

    welcome_message = f"Welcome! You are now chatting with the **{current_profile_name}** powered by `{current_model_id}`.\n\n"
    
    # --- Universal Starter Prompts (regardless of model) ---
    # These will be displayed for every profile.
    starter_prompts = [
        cl.Action(name="generic_hello", value="Hello there!", label="Say Hello", payload={"content": "Hello there!"}),
        cl.Action(name="generic_movie", value="Tell me about a popular movie.", label="Any Movie Info", payload={"content": "Tell me about a popular movie."}),
        cl.Action(name="creative_poem_action", value="Write a poem about a rainy day.", label="Write a poem", payload={"content": "Write a poem about a rainy day."}),
    ]

    # Send the welcome message and universal actions
    await cl.Message(
        content=welcome_message,
        actions=starter_prompts
    ).send()

    # Re-initialize the manager agent with the newly selected model
    manager_agent = Agent(
        name="manager_agent",
        instructions=(
            "You are a helpful assistant. Your main goal is to route user requests to the most appropriate tool or respond directly if no tool is suitable. "
            "**Always provide a meaningful, clear, and comprehensive response to the user's query.** "
            "**Always use the `get_movie_details` tool when the user asks for information about a specific movie, film, or cinematic work. If the user asks for a *specific detail* (e.g., release date, genres, overview), use the `detail_requested` argument of the `get_movie_details` tool to get only that information.** "
            "**Always use the `poetry_agent` tool when the user explicitly asks you to 'write a poem', 'compose verse', 'create a poem', or if their query explicitly mentions 'poem' or 'poet'. Do NOT answer these specific queries directly; always delegate them to the `poetry_agent` tool.** "
            "For all other general questions or conversational requests, answer directly without using any tools. "
            "**After using a tool, always summarize its output in a natural, conversational way before responding to the user. Do not just print the raw tool output. Ensure your final response is clear and directly answers the user's original query.**"
            "**IMPORTANT: If the user asks for a specific detail about a movie (e.g., 'budget', 'director', 'cast', 'runtime') and the `get_movie_details` tool cannot provide that information, you must clearly state that you do not have that particular piece of information available or cannot provide it from the current data. Do not invent or guess details.**"
            "For example, if the tool provides title and overview, but the user asked for 'budget', you should respond: 'I found details for [Movie Title], but I do not have information about its budget at the moment.'"
        ),
        model=OpenAIChatCompletionsModel(
            model=current_model_id, # Use the model of the newly selected profile
            openai_client=openrouter_client,
        ),
        tools=[
            get_movie_details,
            poetry_agent.as_tool(
                tool_name="poetry_agent",
                tool_description="Use this tool to write a poem about a topic or generate poetic verse. Always call this tool when the user requests a poem."
            ),
        ],
    )
    cl.user_session.set("agent", manager_agent)


# --- Chat Lifecycle Events ---
@cl.on_chat_start
async def on_chat_start():
    # Initialize chat history for the session
    cl.user_session.set("chat_history", [])

    # Get the initially selected profile name from the user session.
    # If not explicitly set by Chainlit yet, default to the name of the first profile in PROFILES_CONFIG.
    initial_profile_name = cl.user_session.get("chat_profile_name")
    if not initial_profile_name:
        if PROFILES_CONFIG:
            initial_profile_name = PROFILES_CONFIG[0]["name"] # Use the name of the first profile as default
        else:
            initial_profile_name = "DeepSeek Assistant" # Absolute fallback if PROFILES_CONFIG is empty
    
    # DEBUG: Print the initial profile name being processed
    print(f"DEBUG: Initial chat started with profile: {initial_profile_name}")

    # Call the helper function to set up the initial agent and welcome message
    await setup_agent_and_welcome(initial_profile_name)

@cl.on_settings_update
async def on_settings_update(settings: dict):
    # This is called when the user changes settings, including the chat profile.
    new_profile_name = settings.get("chatProfile")
    if new_profile_name:
        # DEBUG: Print the new profile name received from settings update
        print(f"DEBUG: on_settings_update received new profile: {new_profile_name}")
        await cl.Message(content=f"Chat profile changed to: **{new_profile_name}**").send()
        await setup_agent_and_welcome(new_profile_name) # Re-setup agent with new profile

# Corrected action handler using @cl.action_callback
# Each @cl.action_callback decorator needs to specify the 'name' of the cl.Action it handles
# Removed specific profile actions and kept only the generic ones used universally
@cl.action_callback("generic_hello")
@cl.action_callback("generic_movie")
@cl.action_callback("creative_poem_action") # Renamed for clarity and re-use
async def handle_starter_actions(action: cl.Action): # Renamed function for clarity
    """
    Handles actions triggered by clicking on starter prompts.
    Simulates a user message and passes it to the on_message handler.
    """
    if action.payload and "content" in action.payload:
        user_message_content = action.payload["content"]

        # Send a message to the UI as if the user typed it
        user_message_element = cl.Message(
            author="User",
            content=user_message_content,
        )
        await user_message_element.send()

        # Create a Message object that can be passed to on_message
        message_for_agent = cl.Message(content=user_message_content)

        # Pass this simulated message to the on_message handler for processing
        await on_message(message_for_agent)
    else:
        await cl.Message(content=f"Action '{action.name}' triggered with no content payload.").send()


@cl.on_message
async def on_message(message: cl.Message):
    agent = cast(Agent, cl.user_session.get("agent"))
    chat_history = cl.user_session.get("chat_history")

    chat_history.append({"role": "user", "content": message.content})

    thinking_msg = cl.Message(content="Thinking...")
    await thinking_msg.send()

    response_message = cl.Message(content="") # Initialize the response message
    full_response = "" # Accumulate the full response content

    # Flag to track if the main response message has been sent to Chainlit UI
    response_message_sent = False
    event_count = 0 # Initialize event_count

    try:
        print("\n--- Starting stream_events loop ---")
        response_stream = Runner.run_streamed(
            starting_agent=agent,
            input=chat_history,
        )

        async for event in response_stream.stream_events():
            event_count += 1 # Increment event_count for each event
            # print(f"Received outer event type: {event.type}") # For detailed debugging

            if event.type == "raw_response_event":
                inner_event = event.data
                inner_event_type = getattr(inner_event, 'type', None)

                if inner_event_type == "response.output_text.delta":
                    if not response_message_sent:
                        await thinking_msg.remove()
                        await response_message.send() # Send the message to get its ID
                        response_message_sent = True

                    delta_content = getattr(inner_event, 'delta', '')
                    if delta_content:
                        full_response += delta_content
                        await response_message.stream_token(delta_content)

                elif inner_event_type == "response.output_item.added":
                    item = getattr(inner_event, 'item', None)
                    if item and getattr(item, 'type', None) == 'function_call':
                        # Tool call initiation
                        if not response_message_sent:
                            await thinking_msg.remove()
                            # Do not send response_message yet, as tool call is not the final text response
                            response_message_sent = True

                        tool_name = getattr(item, 'name', 'Unknown Tool')
                        tool_arguments_str = getattr(item, 'arguments', '{}')

                        try:
                            tool_arguments = json.loads(tool_arguments_str)
                            tool_input_display = ", ".join([f"{k}={v}" for k, v in tool_arguments.items()])
                        except json.JSONDecodeError:
                            tool_input_display = tool_arguments_str

                        await cl.Message(
                            content=f"🧠 Calling Tool: `{tool_name}` with input: `{tool_input_display}`",
                            author="Tool Call",
                            parent_id=response_message.id if response_message.id else None,
                        ).send()
                    elif getattr(item, 'type', None) == 'tool_output':
                        # This event means a tool's output is available. The manager agent
                        # should process this and then provide a conversational summary.
                        # We do NOT append this raw output to full_response.
                        # However, for debugging or visibility, you could send a separate Chainlit message.
                        output_content = getattr(inner_event, 'output', '')
                        try:
                            if isinstance(output_content, str):
                                json_parsed = json.loads(output_content)
                                display_content = f"```json\n{json.dumps(json_parsed, indent=2)}\n```"
                            else:
                                display_content = f"```json\n{json.dumps(output_content, indent=2)}\n```"
                        except (json.JSONDecodeError, TypeError):
                            display_content = str(output_content)

                        await cl.Message(
                            content=f"✅ Tool Output: \n{display_content}",
                            author="Tool Output",
                            parent_id=response_message.id if response_message.id else None,
                        ).send()
                        pass # Let the agent process this.

                    elif getattr(item, 'type', None) == 'message':
                        # This is a direct message from a nested agent (like poetry_agent)
                        if not response_message_sent:
                            await thinking_msg.remove()
                            await response_message.send() # Send the message to get its ID
                            response_message_sent = True

                        message_content_parts = getattr(item, 'content', [])
                        if message_content_parts and hasattr(message_content_parts[0], 'text'):
                            text_content = message_content_parts[0].text
                            if text_content:
                                full_response += text_content
                                await response_message.stream_token(text_content)
                        elif isinstance(message_content_parts, str): # Fallback if content is a direct string
                            full_response += message_content_parts
                            await response_message.stream_token(message_content_parts)
                    else:
                        print(f"Unknown item type in response.output_item.added: {item}")


                elif inner_event_type == "response.function_call_arguments.delta":
                    pass # Handled by response.output_item.added for display

                elif inner_event_type in ["response.content_part.done", "response.output_item.done",
                                         "response.created", "response.completed", "response.content_part.added"]:
                    pass

                else:
                    print(f"Unhandled inner event type under raw_response_event: {inner_event_type}, inner data: {inner_event}")

            elif hasattr(event, 'type') and event.type == 'run_item_stream_event':
                inner_item = getattr(event, 'item', None)
                if inner_item:
                    inner_item_type = getattr(inner_item, 'type', 'Unknown RunItemStreamEvent type')

                    if inner_item_type == 'tool_call_item':
                        if not response_message_sent:
                            await thinking_msg.remove()
                            response_message_sent = True # Mark as sent to ensure thinking_msg is gone

                        tool_name = getattr(inner_item.raw_item, 'name', 'Unknown Tool')
                        tool_arguments_str = getattr(inner_item.raw_item, 'arguments', '{}')

                        try:
                            tool_arguments = json.loads(tool_arguments_str)
                            tool_input_display = ", ".join([f"{k}={v}" for k, v in tool_arguments.items()])
                        except json.JSONDecodeError:
                            tool_input_display = tool_arguments_str

                        await cl.Message(
                            content=f"🧠 Calling Tool: `{tool_name}` with input: `{tool_input_display}`",
                            author="Tool Call",
                            parent_id=response_message.id if response_message.id else None,
                        ).send()

                    elif inner_item_type == 'tool_call_output_item':
                        if not response_message_sent:
                            await thinking_msg.remove()
                            response_message_sent = True # Mark as sent to ensure thinking_msg is gone

                        output_content = getattr(inner_item, 'output', '')

                        try:
                            if isinstance(output_content, str):
                                json_parsed = json.loads(output_content)
                                display_content = f"```json\n{json.dumps(json_parsed, indent=2)}\n```"
                            else:
                                display_content = f"```json\n{json.dumps(output_content, indent=2)}\n```"
                        except (json.JSONDecodeError, TypeError):
                            display_content = str(output_content)

                        await cl.Message(
                            content=f"✅ Tool Output: \n{display_content}",
                            author="Tool Output",
                            parent_id=response_message.id if response_message.id else None,
                        ).send()

                    elif inner_item_type == 'message_output_item':
                        # This is a direct message from an agent, likely the final summary
                        if not response_message_sent:
                            await thinking_msg.remove()
                            await response_message.send() # Send the message to get its ID
                            response_message_sent = True

                        raw_message_content = getattr(inner_item.raw_item, 'content', [])

                        if raw_message_content and hasattr(raw_message_content[0], 'text'):
                            text_content = raw_message_content[0].text
                            if text_content:
                                full_response += text_content
                                await response_message.stream_token(text_content)
                        elif isinstance(raw_message_content, str):
                            full_response += raw_message_content
                            await response_message.stream_token(raw_message_content)

                    else:
                        print(f"Unhandled RunItemStreamEvent inner item type: {inner_item_type}, item data: {inner_item}")
                else:
                    print(f"RunItemStreamEvent with no 'item' attribute: {event}")

            # Explicitly ignore known internal event types from agents SDK
            elif event.type in ["agent_updated_stream_event", "turn_started", "turn_finished", "agent_invoked"]:
                pass
            else:
                print(f"Unhandled TOP-LEVEL event type: {event.type}, data: {getattr(event, 'data', event)}")

        print(f"--- Finished stream_events loop. Total events: {event_count} ---")

    except Exception as e:
        print(f"An error occurred during streaming: {e}")
        # Always remove thinking message on error
        await thinking_msg.remove()

        # If the main response message was never sent, send an error message as the main response
        if not response_message_sent:
            error_cl_message = cl.Message(content=f"An error occurred: {e}. Please try again.")
            await error_cl_message.send()
            full_response = error_cl_message.content
        else:
            # If the message was already sent, stream the error as an additional token
            await response_message.stream_token(f"\n\nAn error occurred: {e}. Please try again.")
            full_response += f"\n\nAn error occurred: {e}. Please try again."

    # Finalize the main message content if it was streamed but not fully updated
    # This ensures full_response content is reflected in the message after streaming
    if response_message_sent and response_message.content != full_response:
        response_message.content = full_response
        await response_message.update()
    elif not response_message_sent and full_response.strip():
        # If content accumulated but message was never explicitly sent (e.g., error before first delta)
        response_message.content = full_response
        await response_message.send()
    elif not response_message_sent and not full_response.strip():
        # If no content accumulated and message was never sent, provide a generic message
        response_message.content = "No direct textual response from model. This might indicate an issue with the model's output generation after a tool call, or that the tool output was not processed into a conversational response. Please try again or refine your query."
        await cl.Message(content=full_response).send()


    # Add the final full_response to chat history
    if full_response.strip():
        chat_history.append({"role": "assistant", "content": full_response})
    cl.user_session.set("chat_history", chat_history)


@cl.on_chat_end
def on_chat_end():
    chat_history = cl.user_session.get("chat_history", [])
    if not chat_history:
        print("Chat history is empty.")
        return

    with open("chat_history.json", "w") as f:
        json.dump(chat_history, f, indent=4)
    print("Chat history has been saved to chat_history.json")