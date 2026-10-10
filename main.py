
import os
import sys
import json
import re
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
    captured_stdout = StringIO()
    captured_stderr = StringIO()

    sys.stdout = captured_stdout
    sys.stderr = captured_stderr

    try:
        exec(compile(code, "<string>", "exec"), {})
        output = captured_stdout.getvalue()
        return {"success": True, "output": output}

    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}

    finally:
        sys.stdout = old_stdout
        sys.stderr = sys.__stderr__


def analyze_error_with_ai(
    code: str, error_traceback: str
) -> List[int]:
    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Analyze the Python code and traceback. "
                    "Return JSON with an error_lines array of integers. "
                    "Use 1-based line numbers from the submitted code. "
                    "Identify the line where the exception was raised. "
                    "Do not invent line numbers."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CODE:\n{code}\n\n"
                    f"TRACEBACK:\n{error_traceback}\n\n"
                    'Return JSON: {"error_lines": [3]}'
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    result = ErrorAnalysis.model_validate(data)

    return sorted(set(
        line for line in result.error_lines
        if 1 <= line <= len(code.splitlines())
    ))


def extract_error_lines(code: str, error_traceback: str) -> List[int]:
    matches = re.findall(
        r'File "<string>", line (\d+)',
        error_traceback,
    )

    if not matches:
        return []

    line = int(matches[-1])

    if 1 <= line <= len(code.splitlines()):
        return [line]

    return []


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    # AI analyzes the traceback only when execution fails.
    try:
        error_lines = analyze_error_with_ai(
            request.code,
            execution["output"],
        )
    except Exception:
        error_lines = []

    # Use the traceback to correct or recover the line number.
    traceback_lines = extract_error_lines(
        request.code,
        execution["output"],
    )

    if traceback_lines:
        error_lines = traceback_lines

    return {
        "error": error_lines,
        "result": execution["output"],
    }


@app.get("/")
def home():
    return {"status": "ok"}
