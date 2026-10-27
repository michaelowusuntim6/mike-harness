#!/usr/bin/env python3
"""Deterministic Responses-API server used by the Mike end-to-end smoke test.

No DeepSeek login or network access is required. It walks the model through:

  1. apply_patch  -> create mike_e2e.txt
  2. exec_command -> cat mike_e2e.txt
  3. final assistant text

The point is to exercise Codex's real tool execution and SSE parsing, not the
language model.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = 8123
PATCH = (
    "*** Begin Patch\n"
    "*** Add File: mike_e2e.txt\n"
    "+hello from mike e2e\n"
    "*** End Patch"
)


def sse(event: dict) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()


def message_events(text: str) -> list[bytes]:
    item_id = "msg_mock"
    return [
        sse(
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "id": item_id,
                    "content": [{"type": "output_text", "text": ""}],
                },
            }
        ),
        sse(
            {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "delta": text,
                "content_index": 0,
            }
        ),
        sse(
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "id": item_id,
                    "content": [{"type": "output_text", "text": text}],
                },
            }
        ),
    ]


def completed() -> bytes:
    return sse(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_mock",
                "usage": {
                    "input_tokens": 1,
                    "input_tokens_details": None,
                    "output_tokens": 1,
                    "output_tokens_details": None,
                    "total_tokens": 2,
                },
            },
        }
    )


def tool_output_count(body: dict) -> int:
    return sum(
        1
        for item in body.get("input", [])
        if item.get("type") in ("function_call_output", "custom_tool_call_output")
    )


def response_events(stage: int) -> list[bytes]:
    events = [sse({"type": "response.created", "response": {"id": "resp_mock"}})]
    if stage == 0:
        events += message_events("I'll create the file with a patch.")
        events.append(
            sse(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "custom_tool_call",
                        "call_id": "call_patch",
                        "name": "apply_patch",
                        "input": PATCH,
                    },
                }
            )
        )
    elif stage == 1:
        events += message_events("Now I'll read it back.")
        events.append(
            sse(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_exec",
                        "name": "exec_command",
                        "arguments": json.dumps({"cmd": "cat mike_e2e.txt"}),
                    },
                }
            )
        )
    else:
        events += message_events("Done: created mike_e2e.txt and verified its contents.")
    events.append(completed())
    return events


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("mock-bridge: " + fmt % args + "\n")

    def do_GET(self) -> None:
        if self.path.startswith("/v1/models"):
            body = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "deepseek-v4.1-flash",
                            "object": "model",
                            "owned_by": "mock",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length) or b"{}")
        if self.path.rstrip("/") not in ("/v1/responses", "/responses"):
            self.send_error(404)
            return
        stage = tool_output_count(body)
        events = response_events(stage)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in events:
            self.wfile.write(event)
            self.wfile.flush()
        self.close_connection = True


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"mock bridge listening on http://127.0.0.1:{PORT}/v1", flush=True)
    server.serve_forever()
