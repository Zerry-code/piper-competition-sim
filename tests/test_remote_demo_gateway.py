import http.client
import json
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from remote_demo_gateway import make_server, session_token, valid_session  # noqa: E402


def test_session_tampering_and_expiry():
    key = b"a" * 32
    token = session_token(key, now=100)
    assert valid_session(token, key, now=101)
    assert not valid_session(token, key, now=100 + 8*3600)
    assert not valid_session(token + "0", key, now=101)
    assert not valid_session(token, b"b"*32, now=101)


def test_gateway_auth_origin_and_forwarding():
    received = []

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            data = json.dumps({"stage": "idle", "frame_path": "/private/path", "rgb": "test"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            received.append((self.path, self.headers.get("Origin"), self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(202)
            self.end_headers()
            self.wfile.write(b'{"accepted": true}')

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    base = f"http://127.0.0.1:{upstream.server_port}"
    credentials = {"access_code": "test-only-strong-access-code", "cookie_key": "ab"*32}
    gateway = make_server(base, credentials, 0)
    for server in (upstream, gateway):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    connection = http.client.HTTPConnection("127.0.0.1", gateway.server_port)

    def request(method, path, body=None, headers=None):
        connection.request(method, path, body, headers or {})
        r = connection.getresponse()
        return r.status, dict(r.getheaders()), r.read()

    try:
        status, headers, _ = request("GET", "/")
        assert status == 200
        assert headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        # no-referrer on a native form POST causes Origin:null in browsers.
        status, headers, body = request("POST", "/login", b"code=wrong", {"Origin": "null"})
        assert status == 403 and b"<form " in body and "Set-Cookie" not in headers
        assert request("GET", "/api/state")[0] == 401
        assert request("POST", "/api/run", b"{}")[0] == 401
        assert not received
        assert request("POST", "/login", b"code=wrong")[0] == 401
        status, headers, _ = request("POST", "/login", urllib.parse.urlencode({"code": credentials["access_code"]}))
        assert status == 303
        assert all(value in headers["Set-Cookie"] for value in ("HttpOnly", "Secure", "SameSite=Strict"))
        auth = {"Cookie": headers["Set-Cookie"].split(";")[0], "Content-Type": "application/json"}
        status, _, body = request("GET", "/api/state", headers=auth)
        assert status == 200 and "frame_path" not in json.loads(body)
        assert request("POST", "/api/run", b"{}", {**auth, "Origin": "https://attacker.invalid"})[0] == 403
        assert request("GET", "/../../etc/passwd", headers=auth)[0] == 404
        assert request("POST", "/api/run", b"{}", auth)[0] == 202
        assert received == [("/api/run", base, b"{}")]
    finally:
        connection.close()
        for server in (gateway, upstream):
            server.shutdown()
            server.server_close()


def test_gateway_never_forwards_to_nonlocal_service():
    with pytest.raises(ValueError, match="loopback"):
        make_server("https://example.com:443", {}, 0)
