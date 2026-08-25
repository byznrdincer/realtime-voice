import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from queue_service.events import EVENT_CHANNEL, recent_events
from queue_service.interrupt import request_interrupt
from queue_service.messages import encode_text_message
from queue_service.redis_client import text_redis, USER_INPUT_QUEUE


UI_DIR = Path(__file__).resolve().parent
UI_PORT = int(os.getenv("UI_PORT", "7860"))

_listen_lock = threading.Lock()
_listen_handlers = {
    "pause": None,
    "resume": None,
    "is_listening": lambda: True,
}


def set_listen_controller(pause, resume, is_listening):
    with _listen_lock:
        _listen_handlers["pause"] = pause
        _listen_handlers["resume"] = resume
        _listen_handlers["is_listening"] = is_listening


def listening_active():
    with _listen_lock:
        checker = _listen_handlers["is_listening"]
    try:
        return bool(checker())
    except Exception:
        return True


class UIHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send(self, status, body, content_type="text/html; charset=utf-8"):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlparse(self.path).path

        if path in {"/", "/index.html"}:
            self._send(200, (UI_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            return

        if path == "/api/status":
            self._send(
                200,
                json.dumps({"listening": listening_active()}),
                "application/json",
            )
            return

        if path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.wfile.write(b"event: hello\ndata: {}\n\n")
            listen_frame = json.dumps(
                {
                    "type": "listen",
                    "active": listening_active(),
                },
                ensure_ascii=False,
            )
            self.wfile.write(f"data: {listen_frame}\n\n".encode("utf-8"))
            for item in recent_events():
                self.wfile.write(f"data: {item}\n\n".encode("utf-8"))
            self.wfile.flush()

            pubsub = text_redis.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe(EVENT_CHANNEL)

            try:
                for message in pubsub.listen():
                    if message is None or message.get("type") != "message":
                        continue
                    data = message.get("data") or ""
                    frame = f"data: {data}\n\n".encode("utf-8")
                    self.wfile.write(frame)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
            finally:
                try:
                    pubsub.close()
                except Exception:
                    pass
            return

        self._send(404, "Not found")

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/pause":
            with _listen_lock:
                pause = _listen_handlers["pause"]
            if callable(pause):
                pause()
            self._send(
                200,
                json.dumps(
                    {
                        "ok": True,
                        "listening": listening_active(),
                    }
                ),
                "application/json",
            )
            return

        if path == "/api/resume":
            with _listen_lock:
                resume = _listen_handlers["resume"]
            if callable(resume):
                resume()
            self._send(
                200,
                json.dumps(
                    {
                        "ok": True,
                        "listening": listening_active(),
                    }
                ),
                "application/json",
            )
            return

        if path != "/api/say":
            self._send(404, "Not found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"

        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
            text = str(payload.get("text") or "").strip()
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
            self._send(400, json.dumps({"ok": False}), "application/json")
            return

        if not text:
            self._send(400, json.dumps({"ok": False}), "application/json")
            return

        turn_id = request_interrupt("ui-input")
        text_redis.rpush(
            USER_INPUT_QUEUE,
            encode_text_message(turn_id, text),
        )
        self._send(200, json.dumps({"ok": True, "turn_id": turn_id}), "application/json")


def run_ui_server():
    server = ThreadingHTTPServer(("127.0.0.1", UI_PORT), UIHandler)
    print(f"[UI] http://127.0.0.1:{UI_PORT}")
    server.serve_forever()


def start_ui():
    thread = threading.Thread(target=run_ui_server, daemon=True)
    thread.start()
    return thread
