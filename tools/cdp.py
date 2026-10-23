# -*- coding: utf-8 -*-
"""极简 Chrome DevTools Protocol 客户端（纯标准库）—— 用于 headless Edge 验收界面"""
import base64
import json
import os
import socket
import struct
import subprocess
import time
import urllib.request

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
PORT = 9333


def _http(path, method="GET", timeout=10):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path), method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def start_edge(width=1440, height=900):
    args = [EDGE, "--headless=new", "--remote-debugging-port=%d" % PORT,
            "--window-size=%d,%d" % (width, height), "--hide-scrollbars",
            "--no-first-run", "--no-default-browser-check",
            "--user-data-dir=" + os.path.join(os.environ["TEMP"], "cdp-ud-%d" % time.time()),
            "about:blank"]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(80):
        try:
            if _http("/json/version").get("webSocketDebuggerUrl"):
                return proc
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("Edge 没起来")


class WS:
    def __init__(self, url, timeout=30):
        u = url.split("://", 1)[1]
        hostport, path = u.split("/", 1)
        host, port = hostport.split(":")
        self.s = socket.create_connection((host, int(port)), timeout=timeout)
        self.s.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        req = ("GET /%s HTTP/1.1\r\nHost: %s\r\nUpgrade: websocket\r\n"
               "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\n"
               "Sec-WebSocket-Version: 13\r\n\r\n" % (path, hostport, key))
        self.s.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.s.recv(4096)
        self.buf = buf.split(b"\r\n\r\n", 1)[1]
        self.mid = 0

    def _recv(self, n):
        while len(self.buf) < n:
            chunk = self.s.recv(65536)
            if not chunk:
                raise ConnectionError("closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def send(self, obj):
        data = json.dumps(obj).encode()
        hdr = bytearray([0x81])
        n = len(data)
        if n < 126:
            hdr.append(0x80 | n)
        elif n < 65536:
            hdr.append(0x80 | 126)
            hdr += struct.pack(">H", n)
        else:
            hdr.append(0x80 | 127)
            hdr += struct.pack(">Q", n)
        mask = os.urandom(4)
        hdr += mask
        self.s.sendall(bytes(hdr) + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def recv(self):
        b1, b2 = self._recv(2)
        op = b1 & 0x0F
        n = b2 & 0x7F
        if n == 126:
            n = struct.unpack(">H", self._recv(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", self._recv(8))[0]
        masked = b2 & 0x80
        mk = self._recv(4) if masked else None
        payload = self._recv(n)
        if masked:
            payload = bytes(b ^ mk[i % 4] for i, b in enumerate(payload))
        if op == 8:
            raise ConnectionError("ws closed")
        if op == 9:
            return self.recv()
        return json.loads(payload.decode("utf-8"))

    def call(self, method, **params):
        self.mid += 1
        mid = self.mid
        self.send({"id": mid, "method": method, "params": params})
        while True:
            msg = self.recv()
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError("%s -> %s" % (method, msg["error"]))
                return msg.get("result", {})


class Page:
    def __init__(self, url=None, width=1440, height=900):
        self.proc = start_edge(width, height)
        tab = _http("/json/new?about:blank", "PUT")
        self.ws = WS(tab["webSocketDebuggerUrl"])
        self.ws.call("Page.enable")
        self.ws.call("Runtime.enable")
        if url:
            self.navigate(url)

    def navigate(self, url, wait=2.0):
        self.ws.call("Page.navigate", url=url)
        time.sleep(wait)

    def js(self, expr, wait=0.0, is_async=False):
        r = self.ws.call("Runtime.evaluate", expression=expr, returnByValue=True,
                         awaitPromise=is_async)
        if r.get("exceptionDetails"):
            raise RuntimeError(r["exceptionDetails"].get("text") or str(r["exceptionDetails"]))
        if wait:
            time.sleep(wait)
        return r.get("result", {}).get("value")

    def shot(self, path, wait=0.4):
        time.sleep(wait)
        r = self.ws.call("Page.captureScreenshot", format="png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(r["data"]))
        return path

    def close(self):
        try:
            self.ws.s.close()
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass
