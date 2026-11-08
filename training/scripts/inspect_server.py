"""Test HTTP server for clothes_inspect (model side), shaped like the MCP contract.

POST /clothes_inspect        {"imageBase64": "<jpeg>", "mode": "laid"|"worn", "view": null|"front"|"side"|"back",
                              "mirrored": false, "dismissed": [{"kind","where","box"}]}
POST /clothes_inspect_views  {"images": ["<jpeg>", ...], "mode": ..., "views": [...], "mirrored": false, "dismissed": [...]}
GET  /health
Photos are only decoded in memory for the check and never written to disk. Bind it to the Tailscale address only.
usage: python inspect_server.py <project root> <host> <port>
"""
import base64, json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import clothes_inspect_model as cim

R, HOST, PORT = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
MAX_BODY = 20 * 1024 * 1024
inspector = cim.ClothesInspector(R / "runs/yolo26n_v6_tuned/weights/best.pt", R / "weights/yolo11n-pose.pt")
lock = threading.Lock()  # one GPU, one request at a time


def decode(b64):
    if "," in b64[:100]:  # allow data URLs
        b64 = b64.split(",", 1)[1]
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지를 읽을 수 없어요.")
    return img


def as_contract(result):
    """Add the contract's garments[].defects shape (kind/where/confidence) next to the model's own fields."""
    result["garments"] = [{"defects": [{k: d[k] for k in ("kind", "where", "confidence", "status", "askNumber", "box") if k in d}
                                       for d in result["defects"]]}]
    return result


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True, "modelVersion": cim.MODEL_VERSION, "tell": cim.TELL, "ask": cim.ASK})
        self._send(404, {"error": {"code": "not_found", "message": "없는 경로예요."}})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > MAX_BODY:
            return self._send(413, {"error": {"code": "too_large", "message": "사진이 너무 크거나 비어 있어요."}})
        try:
            req = json.loads(self.rfile.read(n))
            t0 = time.time()
            with lock:
                if self.path == "/clothes_inspect":
                    res = inspector.inspect(decode(req["imageBase64"]), req.get("mode") or "laid", req.get("view"),
                                            bool(req.get("mirrored")), req.get("dismissed"))
                    body = as_contract(res)
                elif self.path == "/clothes_inspect_views":
                    imgs = [decode(b) for b in req["images"]]
                    res = inspector.inspect_views(imgs, req.get("mode") or "worn", req.get("views"),
                                                  bool(req.get("mirrored")), req.get("dismissed"))
                    body = {"results": [as_contract(r) for r in res]}
                else:
                    return self._send(404, {"error": {"code": "not_found", "message": "없는 경로예요."}})
            body["elapsedMs"] = round((time.time() - t0) * 1000)
            self._send(200, body)
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self._send(400, {"error": {"code": "bad_request", "message": str(e) or "요청 형식이 맞지 않아요."}})
        except Exception as e:  # keep the server up; report instead of crashing
            self._send(500, {"error": {"code": "inspect_failed", "message": f"{type(e).__name__}: {e}"}})

    def log_message(self, fmt, *args):  # no request bodies in the log
        sys.stderr.write(f"{time.strftime('%F %T')} {self.client_address[0]} {fmt % args}\n")


if __name__ == "__main__":
    print(f"clothes_inspect test server on http://{HOST}:{PORT} ({cim.MODEL_VERSION}, tell {cim.TELL} / ask {cim.ASK})", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
