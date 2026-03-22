"""
agent.py

Connects to mcp_server.py at startup, fetches tool schemas dynamically,
runs the agent loop. No tool definitions live here.
"""

import asyncio
import json
import os
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
api_key = os.environ.get('OPENROUTER_API_KEY')

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "openai/gpt-5.4-nano"


# ─────────────────────────────────────────
# CONVERT MCP TOOL FORMAT → OPENAI FORMAT
# ─────────────────────────────────────────

def mcp_tool_to_openai(mcp_tool) -> dict:
    """
    MCP and OpenAI use slightly different tool schema formats.
    This converts between them.

    MCP format:
      { name, description, inputSchema: { type, properties, required } }

    OpenAI format:
      { type: "function", function: { name, description, parameters: {...} } }
    """
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.inputSchema   # MCP's inputSchema IS a JSON Schema
        }
    }


# ─────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────

async def chat(user_message: str, history: list, session: ClientSession, tools: list):
    history.append({"role": "user", "content": user_message})
    print(f"\nUser: {user_message}")

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=tools,
            max_tokens=500
        )

        choice = response.choices[0]
        message = choice.message
        history.append(message)

        # ── LLM wants to call a tool ──
        if choice.finish_reason == "tool_calls":
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                print(f"\n  [tool called]  {name}({args})")

                # ask MCP server to execute the tool
                # key difference from before — we don't run Python functions
                # directly, we ask the server to run them
                result = await session.call_tool(name, args)

                # result.content is a list of content blocks
                # for simple tools it's just one text block
                tool_output = result.content[0].text
                print(f"  [tool result]  {tool_output}")

                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output
                })

        # ── LLM is done ──
        elif choice.finish_reason == "stop":
            reply = message.content
            print(f"\nAssistant: {reply}")
            return reply


# ─────────────────────────────────────────
# MAIN — connect to MCP server, then chat
# ─────────────────────────────────────────

async def main():
    # point to the mcp server script
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )

    # spawn the server process and open a session
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # initialize the connection
            await session.initialize()

            # fetch all tools from the server dynamically
            # if you add a new tool to mcp_server.py, it appears here
            # automatically — agent.py never needs to change
            mcp_tools = await session.list_tools()
            tools = [mcp_tool_to_openai(t) for t in mcp_tools.tools]

            print("Tools loaded from MCP server:")
            for t in mcp_tools.tools:
                print(f"  - {t.name}: {t.description}")

            # shared history across turns
            history = []

            await chat("What is 128 multiplied by 37?", history, session, tools)
            await chat("What time is it right now?", history, session, tools)
            await chat("Reverse the word 'hello'", history, session, tools)


if __name__ == "__main__":
    asyncio.run(main())