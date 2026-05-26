"""
Codex 代理 - 多个模式可切换
链路：Codex → ccswitch → 本代理(:4001) → LiteLLM(:4000) → 大模型

模式 passthrough:    只透传 + 记录日志，不做任何过滤和转换
模式 drop-params:    只过滤非标准参数（client_metadata 等），不过滤 tool 类型
模式 filter:         过滤非标准参数 + 过滤非标准 tool + 降级转换
"""

import json
import logging
import os
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error

# ========== 命令行参数 ==========

parser = argparse.ArgumentParser(description="Codex Proxy")
parser.add_argument("--mode", choices=["passthrough", "drop-params", "filter"], default="passthrough",
                    help="passthrough: 只透传+日志; drop-params: 只过滤参数不过滤tool; filter: 过滤参数+tool+降级 (默认: passthrough)")
parser.add_argument("--litellm", default="http://localhost:4000",
                    help="LiteLLM 地址 (默认: http://localhost:4000)")
parser.add_argument("--port", type=int, default=4001,
                    help="代理监听端口 (默认: 4001)")
parser.add_argument("--log-dir", default="logs",
                    help="日志目录 (默认: logs)")
args = parser.parse_args()

LITELLM_URL = args.litellm
PORT = args.port
LOG_DIR = args.log_dir
MODE = args.mode

# 需要从请求中删除的参数（Codex /responses 专属）
PARAMS_TO_DROP = [
    "client_metadata",
    "vector_store_ids",
    "vector_store_request_metadata",
    "prompt_cache_key",
    "store",
    "include",
    "reasoning",
    "parallel_tool_calls",
]

# 需要从 tools 数组中删除的类型（仅在 filter 模式生效）
TOOL_TYPES_TO_DROP = [
    "file_search",
    "web_search",
    "namespace",
]

# ========== 日志 ==========

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(LOG_DIR, "proxy.log"), encoding="utf-8"),
    ],
)
log = logging.getLogger("codex-proxy")


def save_log(filename, data):
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        if isinstance(data, bytes):
            f.write(data.decode("utf-8", errors="replace"))
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return filepath


# ========== 请求过滤 ==========

def drop_params(body):
    """只过滤非标准参数，不过滤 tool 类型"""
    if not isinstance(body, dict):
        return body

    filtered = dict(body)

    for key in PARAMS_TO_DROP:
        if key in filtered:
            log.info(f"  [drop-params] Dropped param: {key}")
            del filtered[key]

    return filtered


def filter_request(body):
    """过滤非标准参数 + 过滤非标准 tool 类型"""
    if not isinstance(body, dict):
        return body

    filtered = dict(body)

    for key in PARAMS_TO_DROP:
        if key in filtered:
            log.info(f"  [filter] Dropped param: {key}")
            del filtered[key]

    if "tools" in filtered and isinstance(filtered["tools"], list):
        original_tools = body.get("tools", [])
        filtered["tools"] = [
            t for t in filtered["tools"]
            if t.get("type") not in TOOL_TYPES_TO_DROP
        ]
        dropped_tools = [
            f"{t.get('type')}:{t.get('name', t.get('type'))}"
            for t in original_tools
            if t.get("type") in TOOL_TYPES_TO_DROP
        ]
        if dropped_tools:
            log.info(f"  [filter] Dropped tools: {dropped_tools}")
        if len(filtered["tools"]) == 0:
            log.info(f"  [filter] No tools left, removing tools field")
            del filtered["tools"]
            if "tool_choice" in filtered:
                del filtered["tool_choice"]

    return filtered


# ========== /responses → /chat/completions 转换 ==========

def responses_to_chat(body):
    messages = []
    tools = []

    if "instructions" in body and body["instructions"]:
        messages.append({"role": "system", "content": body["instructions"]})

    input_data = body.get("input")
    if isinstance(input_data, str):
        messages.append({"role": "user", "content": input_data})
    elif isinstance(input_data, list):
        for item in input_data:
            if item.get("type") == "message":
                role = item.get("role", "user")
                content_parts = item.get("content", [])
                if isinstance(content_parts, list):
                    text_parts = []
                    for part in content_parts:
                        if isinstance(part, str):
                            text_parts.append(part)
                        elif isinstance(part, dict) and part.get("type") == "input_text":
                            text_parts.append(part.get("text", ""))
                    content = "\n".join(text_parts)
                elif isinstance(content_parts, str):
                    content = content_parts
                else:
                    content = str(content_parts)
                if role == "developer":
                    role = "system"
                messages.append({"role": role, "content": content})
            elif item.get("type") == "function_call":
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": item.get("call_id", f"call_{len(messages)}"),
                        "type": "function",
                        "function": {
                            "name": item.get("name"),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }],
                })
            elif item.get("type") == "function_call_output":
                messages.append({
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": item.get("output", ""),
                })

    if "tools" in body and isinstance(body["tools"], list):
        for tool in body["tools"]:
            if tool.get("type") == "function" and tool.get("function"):
                tools.append({
                    "type": "function",
                    "function": tool["function"],
                })

    result = {
        "model": body.get("model"),
        "messages": messages,
        "stream": body.get("stream", False),
    }

    if tools:
        result["tools"] = tools
        if "tool_choice" in body:
            result["tool_choice"] = body["tool_choice"]
    if "temperature" in body:
        result["temperature"] = body["temperature"]
    if "max_output_tokens" in body:
        result["max_tokens"] = body["max_output_tokens"]
    if "top_p" in body:
        result["top_p"] = body["top_p"]

    return result


def chat_to_responses(data, original_model):
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    tool_calls = message.get("tool_calls", [])

    output_content = []

    if content:
        output_content.append({"type": "output_text", "text": content})

    for tc in tool_calls or []:
        output_content.append({
            "type": "function_call",
            "name": tc.get("function", {}).get("name"),
            "call_id": tc.get("id"),
            "arguments": tc.get("function", {}).get("arguments"),
        })

    return {
        "id": data.get("id", f"resp-{int(time.time())}"),
        "object": "response",
        "model": original_model,
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": output_content,
        }],
        "usage": {
            "input_tokens": data.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": data.get("usage", {}).get("completion_tokens", 0),
        },
        "status": "completed",
    }


# ========== 代理 ==========

class ProxyHandler(BaseHTTPRequestHandler):

    def forward(self, url, headers, body_bytes):
        req = urllib.request.Request(url, data=body_bytes, method="POST")
        for key, value in headers:
            if key.lower() not in ("host", "content-length", "transfer-encoding"):
                req.add_header(key, value)
        return urllib.request.urlopen(req)

    def do_POST(self):
        path = self.path

        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""

        try:
            request_json = json.loads(raw_body) if raw_body else {}
        except json.JSONDecodeError:
            request_json = {}

        headers_list = [(k, v) for k, v in self.headers.items()]
        ts = int(time.time())

        # 日志：请求摘要 + 完整请求体
        log.info("=" * 80)
        log.info(f"REQUEST: POST {path} [mode={MODE}]")
        summary = {
            "model": request_json.get("model"),
            "stream": request_json.get("stream"),
            "tools_count": len(request_json.get("tools", [])),
            "tools": [(t.get("type"), t.get("name", t.get("type"))) for t in request_json.get("tools", [])],
            "extra_keys": [k for k in request_json.keys() if k not in ("model", "input", "stream", "tools", "temperature", "max_output_tokens", "top_p")],
        }
        log.info(f"Summary: {json.dumps(summary, ensure_ascii=False)}")
        save_log(f"req_{ts}.json", request_json)

        # ========== PASSTHROUGH 模式：只透传 + 记录日志 ==========

        if MODE == "passthrough":
            try:
                url = f"{LITELLM_URL}{path}"
                req = urllib.request.Request(url, data=raw_body, method="POST")
                for key, value in headers_list:
                    if key.lower() not in ("host", "content-length", "transfer-encoding"):
                        req.add_header(key, value)
                resp = urllib.request.urlopen(req)
                response_body = resp.read()

                # 记录响应日志
                log.info(f"RESPONSE: {resp.status}")
                log.info(f"Response headers: {dict(resp.headers)}")
                try:
                    resp_json = json.loads(response_body)
                    save_log(f"resp_{ts}.json", resp_json)
                except json.JSONDecodeError:
                    save_log(f"resp_{ts}.raw", response_body)
                    log.info(f"Response is not JSON (probably SSE stream)")

                # 透传响应
                self.send_response(resp.status)
                for key, value in dict(resp.headers).items():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response_body)

            except urllib.error.HTTPError as e:
                error_body = e.read()
                log.error(f"RESPONSE ERROR: {e.code} {e.reason}")
                log.error(f"Error body: {error_body.decode('utf-8', errors='replace')[:2000]}")
                save_log(f"err_{ts}.json", error_body.decode("utf-8", errors="replace"))

                self.send_response(e.code)
                for key, value in dict(e.headers).items():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(error_body)

            return  # passthrough 模式结束

        # ========== DROP-PARAMS 模式：只过滤参数，不过滤 tool ==========

        if MODE == "drop-params":
            dropped = drop_params(request_json)
            dropped_bytes = json.dumps(dropped).encode("utf-8")

            save_log(f"req_dropped_{ts}.json", dropped)

            try:
                url = f"{LITELLM_URL}{path}"
                req = urllib.request.Request(url, data=dropped_bytes, method="POST")
                for key, value in headers_list:
                    if key.lower() not in ("host", "content-length", "transfer-encoding"):
                        req.add_header(key, value)
                resp = urllib.request.urlopen(req)
                response_body = resp.read()

                log.info(f"RESPONSE: {resp.status}")
                log.info(f"Response headers: {dict(resp.headers)}")
                try:
                    resp_json = json.loads(response_body)
                    save_log(f"resp_{ts}.json", resp_json)
                except json.JSONDecodeError:
                    save_log(f"resp_{ts}.raw", response_body)
                    log.info(f"Response is not JSON (probably SSE stream)")

                self.send_response(resp.status)
                for key, value in dict(resp.headers).items():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response_body)

            except urllib.error.HTTPError as e:
                error_body = e.read()
                log.error(f"RESPONSE ERROR: {e.code} {e.reason}")
                log.error(f"Error body: {error_body.decode('utf-8', errors='replace')[:2000]}")
                save_log(f"err_{ts}.json", error_body.decode("utf-8", errors="replace"))

                self.send_response(e.code)
                for key, value in dict(e.headers).items():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(error_body)

            return  # drop-params 模式结束

        # ========== FILTER 模式：过滤 + 降级 ==========

        # === /v1/responses：过滤后转发，失败则降级 ===
        if path == "/v1/responses":
            filtered = filter_request(request_json)
            filtered_bytes = json.dumps(filtered).encode("utf-8")

            save_log(f"req_filtered_{ts}.json", filtered)

            # 第一步：尝试 LiteLLM /responses
            try:
                resp = self.forward(f"{LITELLM_URL}/v1/responses", headers_list, filtered_bytes)
                response_body = resp.read()

                if filtered.get("stream"):
                    log.info(f"RESPONSE: 200 (stream, forwarded directly)")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(response_body)
                    return

                log.info(f"RESPONSE: {resp.status} (from /responses)")
                try:
                    resp_json = json.loads(response_body)
                    save_log(f"resp_responses_{ts}.json", resp_json)
                except json.JSONDecodeError:
                    save_log(f"resp_responses_{ts}.raw", response_body)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(response_body)
                return

            except urllib.error.HTTPError as e:
                error_body = e.read()
                log.warning(f"/responses failed: {e.code}, trying /chat/completions fallback")
                save_log(f"err_responses_{ts}.json", error_body.decode("utf-8", errors="replace"))

                # 第二步：降级到 /chat/completions
                chat_body = responses_to_chat(filtered)
                chat_bytes = json.dumps(chat_body).encode("utf-8")

                save_log(f"req_chat_fallback_{ts}.json", chat_body)
                log.info(f"FALLBACK: model={chat_body.get('model')}, messages={len(chat_body.get('messages', []))}, tools={len(chat_body.get('tools', []))}")

                try:
                    resp = self.forward(f"{LITELLM_URL}/v1/chat/completions", headers_list, chat_bytes)
                    response_body = resp.read()

                    if filtered.get("stream"):
                        log.info(f"FALLBACK RESPONSE: 200 (stream from /chat/completions)")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.end_headers()
                        self.wfile.write(response_body)
                        return

                    chat_resp = json.loads(response_body)
                    responses_resp = chat_to_responses(chat_resp, filtered.get("model", "unknown"))

                    log.info(f"FALLBACK RESPONSE: converted back to /responses format")
                    save_log(f"resp_chat_fallback_{ts}.json", chat_resp)
                    save_log(f"resp_final_{ts}.json", responses_resp)

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(responses_resp).encode("utf-8"))
                    return

                except urllib.error.HTTPError as e2:
                    error2_body = e2.read()
                    log.error(f"/chat/completions also failed: {e2.code}")
                    save_log(f"err_chat_fallback_{ts}.json", error2_body.decode("utf-8", errors="replace"))

                    self.send_response(e2.code)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(error2_body)
                    return

        # === /v1/chat/completions：过滤后直接转发 ===
        elif path == "/v1/chat/completions":
            filtered = filter_request(request_json)
            filtered_bytes = json.dumps(filtered).encode("utf-8")

            try:
                resp = self.forward(f"{LITELLM_URL}/v1/chat/completions", headers_list, filtered_bytes)
                response_body = resp.read()
                log.info(f"RESPONSE: {resp.status} (from /chat/completions)")
                save_log(f"resp_{ts}.json", response_body.decode("utf-8", errors="replace"))
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.end_headers()
                self.wfile.write(response_body)
            except urllib.error.HTTPError as e:
                error_body = e.read()
                log.error(f"/chat/completions failed: {e.code}")
                save_log(f"err_{ts}.json", error_body.decode("utf-8", errors="replace"))
                self.send_response(e.code)
                self.end_headers()
                self.wfile.write(error_body)

        # === 其他路径：直接透传 ===
        else:
            try:
                url = f"{LITELLM_URL}{path}"
                req = urllib.request.Request(url, data=raw_body, method="POST")
                for key, value in headers_list:
                    if key.lower() not in ("host", "content-length", "transfer-encoding"):
                        req.add_header(key, value)
                resp = urllib.request.urlopen(req)
                response_body = resp.read()
                self.send_response(resp.status)
                for key, value in dict(resp.headers).items():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response_body)
            except urllib.error.HTTPError as e:
                error_body = e.read()
                self.send_response(e.code)
                self.end_headers()
                self.wfile.write(error_body)

    def do_GET(self):
        path = self.path
        url = f"{LITELLM_URL}{path}"
        ts = int(time.time())

        log.info("=" * 80)
        log.info(f"REQUEST: GET {path} [mode={MODE}]")

        # 透传 Authorization
        auth = self.headers.get("Authorization", "")
        headers_list = [("Authorization", auth)] if auth else []

        req = urllib.request.Request(url, method="GET")
        for key, value in headers_list:
            req.add_header(key, value)

        try:
            resp = urllib.request.urlopen(req)
            response_body = resp.read()

            log.info(f"RESPONSE: {resp.status}")
            log.info(f"Response headers: {dict(resp.headers)}")
            try:
                resp_json = json.loads(response_body)
                save_log(f"resp_get_{ts}.json", resp_json)
                log.info(f"Response saved to resp_get_{ts}.json")
            except json.JSONDecodeError:
                save_log(f"resp_get_{ts}.raw", response_body)
                log.info(f"Response saved to resp_get_{ts}.raw")

            self.send_response(resp.status)
            for key, value in dict(resp.headers).items():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(response_body)

        except urllib.error.HTTPError as e:
            error_body = e.read()
            log.error(f"RESPONSE ERROR: {e.code} {e.reason}")
            log.error(f"Error body: {error_body.decode('utf-8', errors='replace')[:2000]}")
            save_log(f"err_get_{ts}.json", error_body.decode("utf-8", errors="replace"))

            self.send_response(e.code)
            for key, value in dict(e.headers).items():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(error_body)

        except Exception as e:
            log.error(f"GET request exception: {type(e).__name__}: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Proxy error: {type(e).__name__}: {e}"}).encode("utf-8"))

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), ProxyHandler)
    log.info(f"Codex Proxy running on :{PORT}")
    log.info(f"Forwarding to LiteLLM at :{LITELLM_URL}")
    log.info(f"Log dir: {LOG_DIR}")
    log.info(f"Mode: {MODE}")
    if MODE == "filter":
        log.info(f"Dropping params: {PARAMS_TO_DROP}")
        log.info(f"Dropping tool types: {TOOL_TYPES_TO_DROP}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        server.server_close()