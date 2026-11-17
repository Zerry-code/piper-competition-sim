"""Password-protected loopback gateway for an HTTPS tunnel to the demo.

The simulation itself stays on localhost. Only explicitly listed demo routes
are forwarded. Credentials belong in an ignored, permission-0600 JSON file.
"""

import argparse
import hashlib
import hmac
import json
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOGIN = b'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PiPER Remote</title><style>body{font-family:system-ui;background:#101722;color:#dce7f4;max-width:460px;margin:12vh auto;padding:24px}form{background:#182331;padding:28px;border-radius:16px}input,button{box-sizing:border-box;width:100%;font:inherit;padding:14px;margin-top:14px;border-radius:8px;border:1px solid #41536a}input{background:#101722;color:white}button{background:#64dfb8;color:#102b25;cursor:pointer}p{line-height:1.7;color:#a1b0c3}</style><h1>PiPER / Remote</h1><p>'''+ '输入访问口令，连接仿真与本地千问服务。'.encode()+b'''</p><form method="post" action="/login"><label for="code">'''+ '访问口令'.encode()+b'''</label><input id="code" name="code" type="password" autocomplete="current-password" required autofocus><button type="submit">'''+ '进入控制界面'.encode()+b'''</button></form></html>'''


def session_token(key, now=None):
    payload = f"{int(now or time.time()) + 8 * 3600}.{secrets.token_hex(12)}"
    return payload + "." + hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()


def valid_session(token, key, now=None):
    try:
        expiry, nonce, signature = token.split(".")
        payload = expiry + "." + nonce
        expected = hmac.new(key, payload.encode(), hashlib.sha256).hexdigest()
        return int(expiry) > int(now or time.time()) and hmac.compare_digest(signature, expected)
    except (ValueError, TypeError):
        return False


def make_server(upstream, credentials, port):
    parsed = urllib.parse.urlsplit(upstream)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port:
        raise ValueError("Upstream must be an explicit loopback HTTP port")
    code = credentials["access_code"].encode()
    key = bytes.fromhex(credentials["cookie_key"])
    if len(code) < 16 or len(key) < 32:
        raise ValueError("Use a random access code and 32-byte cookie key")
    failures = {}
    lock = threading.Lock()
    allowed_posts = {"/api/run", "/api/pause", "/api/resume", "/api/stop", "/api/reset", "/api/release"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, payload, status=200, kind="text/html; charset=utf-8", headers=None):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        def error(self, text, status):
            self.send(json.dumps({"error": text}, ensure_ascii=False).encode(), status,
                      "application/json; charset=utf-8")

        def authenticated(self):
            cookie = SimpleCookie()
            try:
                cookie.load(self.headers.get("Cookie", ""))
                return "piper_session" in cookie and valid_session(cookie["piper_session"].value, key)
            except Exception:  # Malformed untrusted cookie is never authentication.
                return False

        def forward(self, body=None):
            headers = {"Content-Type": "application/json"}
            # Gateway has already checked browser origin. Translate it to the local origin.
            if body is not None:
                headers["Origin"] = upstream
            request = urllib.request.Request(upstream + self.path, data=body, headers=headers)
            try:
                response = urllib.request.urlopen(request, timeout=15)
            except urllib.error.HTTPError as exc:
                response = exc
            except (urllib.error.URLError, TimeoutError):
                self.error("仿真服务暂时不可用，请确认运行模型的电脑和 demo 服务在线。", 503)
                return
            with response:
                payload = response.read()
                kind = response.headers.get("Content-Type", "application/octet-stream")
                if self.path == "/api/state" and response.status == 200:
                    def sanitize(value):
                        if isinstance(value, dict):
                            return {k: sanitize(v) for k, v in value.items() if k not in ("frame_path", "result_path")}
                        if isinstance(value, list):
                            return [sanitize(v) for v in value]
                        return value
                    payload = json.dumps(sanitize(json.loads(payload)), ensure_ascii=False).encode()
                elif self.path == "/" and response.status == 200:
                    banner = '<p>远程控制 · 所有浏览器共享同一个仿真场景。<a href="/login" style="color:#64dfb8">登录页</a></p>'
                    payload = payload.replace(b'<div class="top">', banner.encode()+b'<div class="top">', 1)
                self.send(payload, response.status, kind)

        def do_GET(self):
            if self.path == "/login" or (self.path == "/" and not self.authenticated()):
                self.send(LOGIN)
            elif not self.authenticated():
                self.error("请先输入访问口令登录。", 401)
            elif self.path in ("/", "/api/state"):
                self.forward()
            else:
                self.error("Not found", 404)

        def do_POST(self):
            origin = self.headers.get("Origin")
            host = self.headers.get("Host", "")
            if origin and origin not in ("https://" + host, "http://" + host):
                print(json.dumps({"event":"origin_rejected","path":self.path,"origin":origin,"host":host,"forwarded_host":self.headers.get("X-Forwarded-Host"),"forwarded_proto":self.headers.get("X-Forwarded-Proto"),"fetch_site":self.headers.get("Sec-Fetch-Site")}), flush=True)
                if self.path == "/login":
                    self.send(LOGIN.replace(b"<form ", "<p style=\"color:#ffb0a6\">登录页面已更新，请在此页面重新输入口令。</p><form ".encode()), 403)
                else:
                    self.error("跨站请求被拒绝", 403)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 20000:
                    raise ValueError()
            except ValueError:
                self.error("Invalid request size", 400)
                return
            body = self.rfile.read(length)
            if self.path == "/login":
                identity = self.headers.get("CF-Connecting-IP", self.client_address[0])
                now = time.monotonic()
                with lock:
                    recent = [t for t in failures.get(identity, []) if now-t < 60]
                    if len(recent) >= 5:
                        self.error("尝试过多，请一分钟后再试。", 429)
                        return
                    failures[identity] = recent + [now]
                entered = urllib.parse.parse_qs(body.decode(errors="replace")).get("code", [""])[0]
                if not hmac.compare_digest(entered.encode(), code):
                    self.send(LOGIN.replace(b'<form ', '<p style="color:#ffb0a6">口令不正确，请重试。</p><form '.encode()), 401)
                    return
                with lock:
                    failures.pop(identity, None)
                cookie = f"piper_session={session_token(key)}; HttpOnly; Secure; SameSite=Strict; Path=/; Max-Age=28800"
                self.send(b"", 303, headers={"Set-Cookie": cookie, "Location": "/"})
            elif not self.authenticated():
                self.error("请先输入访问口令登录。", 401)
            elif self.path not in allowed_posts:
                self.error("Not found", 404)
            elif self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                self.error("JSON required", 400)
            else:
                self.forward(body)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", default="http://127.0.0.1:8771")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--credentials", type=Path, required=True)
    args = parser.parse_args()
    server = make_server(args.upstream, json.loads(args.credentials.read_text(encoding="utf-8")), args.port)
    print(f"Protected gateway listening on 127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
