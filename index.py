from openai import OpenAI
import os
from datetime import datetime
import json

api_key = os.environ.get('OPENROUTER_API_KEY')

client = OpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "openai/gpt-5.4-nano"

# ─────────────────────────────────────────
# 1. DEFINE YOUR TOOLS (JSON schema)
# ─────────────────────────────────────────

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a basic math operation: add, subtract, multiply, divide",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The math operation to perform"
                    },
                    "a": {
                        "type": "number",
                        "description": "First number"
                    },
                    "b": {
                        "type": "number",
                        "description": "Second number"
                    }
                },
                "required": ["operation", "a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# ─────────────────────────────────────────
# 2. ACTUAL PYTHON FUNCTIONS THAT RUN
# ─────────────────────────────────────────

def calculate(operation: str, a: float, b: float) -> str:
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return "Error: cannot divide by zero"
        result = a / b
    return f"{a} {operation} {b} = {result}"


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────────────────────
# 3. DISPATCHER — maps tool name → function
# ─────────────────────────────────────────

def run_tool(name: str, args: dict) -> str:
    print(f"\n  [tool called]  {name}({args})")        # so you can see it happen
    if name == "calculate":
        return calculate(**args)
    elif name == "get_current_time":
        return get_current_time()
    else:
        return f"Unknown tool: {name}"


# ─────────────────────────────────────────
# 4. THE AGENT LOOP
# ─────────────────────────────────────────

def chat(user_message: str, history: list) -> str:
    # add user message to history
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

        # always add the assistant's response to history
        history.append(message)

        # ── case 1: LLM wants to call a tool ──
        if choice.finish_reason == "tool_calls":
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                result = run_tool(name, args)
                print(f"  [tool result]  {result}")

                # feed result back into history
                history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
            # loop again — LLM will now form a final answer

        # ── case 2: LLM is done, gives final answer ──
        elif choice.finish_reason == "stop":
            reply = message.content
            print(f"\nAssistant: {reply}")
            return reply


# ─────────────────────────────────────────
# 5. RUN IT
# ─────────────────────────────────────────

if __name__ == "__main__":
    history = []   # shared across turns — this IS the memory

    chat("What is 128 multiplied by 37?", history)
    chat("What time is it right now?", history)
    chat("Divide that first result by 4", history)  # uses memory of previous answer