

import os
import sys
import json
import traceback
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    api_key=os.environ.get("AIPIPE_TOKEN", ""),
    base_url="https://aipipe.org/openai/v1",
)


class CodeRequest(BaseModel):
    code: str


class ErrorAnalysis(BaseModel):
    error_lines: List[int]


def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    captured = StringIO()
    sys.stdout = captured

    try:
        exec(compile(code, "<string>", "exec"), {})
        return {"success": True, "output": captured.getvalue()}
    except Exception:
        return {"success": False, "output": traceback.format_exc()}
    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Find the exact 1-based line number where the "
                    "exception occurred in the submitted Python code. "
                    "Return JSON only: {\"error_lines\": [3]}. "
                    "Do not invent line numbers."
                ),
            },
            {
                "role": "user",
                "content": f"CODE:\n{code}\n\nTRACEBACK:\n{error_traceback}",
            },
        ],
        response_format={"type": "json_object"},
    )

    data = json.loads(response.choices[0].message.content)
    result = ErrorAnalysis.model_validate(data)
    return sorted(set(
        n for n in result.error_lines
        if 1 <= n <= len(code.splitlines())
    ))


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {"error": [], "result": execution["output"]}

    try:
        lines = analyze_error_with_ai(request.code, execution["output"])
    except Exception:
        lines = []

    return {"error": lines, "result": execution["output"]}


@app.get("/")
def home():
    return {"status": "ok"}
