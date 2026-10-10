
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
    old_stderr = sys.stderr
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()

    try:
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        exec(compile(code, "<string>", "exec"), {})

        return {
            "success": True,
            "output": stdout_buffer.getvalue(),
        }

    except Exception:
        return {
            "success": False,
            "output": traceback.format_exc(),
        }

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def extract_error_lines(code: str, error_traceback: str) -> List[int]:
    # Only consider traceback frames referring to submitted code.
    matches = re.findall(
        r'File "<string>", line (\d+)',
        error_traceback,
    )

    if not matches:
        return []

    line_number = int(matches[-1])

    if 1 <= line_number <= len(code.splitlines()):
        return [line_number]

    return []


def analyze_error_with_ai(
    code: str,
    error_traceback: str,
) -> List[int]:
    response = client.chat.completions.create(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Identify the exact 1-based line number in the "
                    "submitted Python code where the exception occurs. "
                    'Return JSON only: {"error_lines": [3]}. '
                    "Do not invent line numbers."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CODE:\n{code}\n\n"
                    f"TRACEBACK:\n{error_traceback}"
                ),
            },
        ],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    parsed = ErrorAnalysis.model_validate(data)

    return sorted({
        n for n in parsed.error_lines
        if 1 <= n <= len(code.splitlines())
    })


@app.post("/code-interpreter")
def code_interpreter(request: CodeRequest):
    execution = execute_python_code(request.code)

    if execution["success"]:
        return {
            "error": [],
            "result": execution["output"],
        }

    # Analyze errors with AI only when execution fails.
    try:
        ai_lines = analyze_error_with_ai(
            request.code,
            execution["output"],
        )
    except Exception:
        ai_lines = []

    # The actual traceback is authoritative when it identifies
    # a line in the submitted code.
    traceback_lines = extract_error_lines(
        request.code,
        execution["output"],
    )

    error_lines = traceback_lines or ai_lines

    return {
        "error": error_lines,
        "result": execution["output"],
    }


@app.get("/")
def home():
    return {"status": "ok"}
