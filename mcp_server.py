"""
mcp_server.py

Run this as a standalone process. The agent connects to it.
To add a new tool: add a function with @mcp.tool() decorator. That's it.
"""

from mcp.server.fastmcp import FastMCP
from datetime import datetime

mcp = FastMCP("my-tools")

# ─────────────────────────────────────────
# TOOLS — add new ones here anytime
# ─────────────────────────────────────────

@mcp.tool()
def calculate(operation: str, a: float, b: float) -> str:
    """
    Perform a basic math operation.
    operation: one of 'add', 'subtract', 'multiply', 'divide'
    a: first number
    b: second number
    """
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
    else:
        return f"Unknown operation: {operation}"
    return f"{a} {operation} {b} = {result}"


@mcp.tool()
def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@mcp.tool()
def reverse_text(text: str) -> str:
    """Reverses any given string of text."""
    return text[::-1]


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")   # agent will spawn this process