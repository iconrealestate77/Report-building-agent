import ast
import operator
import json
import os
from datetime import datetime, timezone
from langchain_core.tools import tool

os.makedirs("logs", exist_ok=True)

_ALLOWED_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")

def _log_tool_call(expression: str, result: str, success: bool):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "calculator",
        "expression": expression,
        "result": result,
        "success": success,
    }
    with open("logs/tool_calls.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely and return the result as a string.
    Supports +, -, *, /, **, %, and parentheses. Example: '(120 + 45) * 3'."""
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval(parsed.body)
        result_str = str(result)
        _log_tool_call(expression, result_str, success=True)
        return result_str
    except Exception as e:
        error_msg = f"Error evaluating expression: {e}"
        _log_tool_call(expression, error_msg, success=False)
        return error_msg
