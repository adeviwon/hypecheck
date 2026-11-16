"""Exercise real report, static isolation and HTTP security behavior."""
import json
from pathlib import Path
import threading
import unittest
import tempfile
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

from serve import Handler

ROOT = Path(__file__).resolve().parent


class WebChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def test_allowlisted_symlink_cannot_expose_private_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "public"
            root.mkdir()
            secret = Path(tmp) / "private.txt"
            secret.write_text("not public")
            (root / "report.json").symlink_to(secret)
            with patch("serve.DOCS", root):
                with self.assertRaises(HTTPError) as caught:
                    urlopen(self.base + "/report.json", timeout=5)
                self.assertEqual(caught.exception.code, 404)

    def test_real_report_and_security_headers(self):
        with urlopen(self.base + "/report.json", timeout=5) as response:
            report = json.load(response)
            self.assertEqual(len(report["predictions"]), report["metrics"][0]["n"])
            self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
            self.assertEqual(response.headers["X-Frame-Options"], "DENY")
            self.assertIn("script-src 'self'", response.headers["Content-Security-Policy"])
            self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
            self.assertNotIn("Access-Control-Allow-Origin", response.headers)

    def test_private_paths_and_mutations_are_blocked(self):
        for path in ["/../pipeline.py", "/%2e%2e/pipeline.py", "/.env", "/data/posts.parquet", "/.git/config", "/pipeline.py", "/artifacts/manifest.json"]:
            with self.subTest(path=path):
                with self.assertRaises(HTTPError) as error:
                    urlopen(self.base + path, timeout=5)
                self.assertEqual(error.exception.code, 404)
        for method in ["POST", "PUT", "DELETE", "PATCH"]:
            with self.subTest(method=method):
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(self.base + "/report.json", data=b"{}", method=method), timeout=5)
                self.assertEqual(error.exception.code, 405)

    def test_head_and_local_assets(self):
        for path in ["/", "/app.js", "/style.css", "/favicon.svg", "/live.html", "/live.js", "/live.css", "/live-model.json"]:
            with urlopen(Request(self.base + path, method="HEAD"), timeout=5) as response:
                self.assertEqual(response.status, 200)
                self.assertGreater(int(response.headers["Content-Length"]), 0)
                self.assertEqual(response.read(), b"")
        html = (ROOT / "docs/index.html").read_text()
        self.assertIn("Content-Security-Policy", html)
        self.assertNotIn("unsafe-inline", html)
        self.assertNotIn("<script>", html)
        js = (ROOT / "docs/app.js").read_text()
        for unsafe in ["innerHTML", "document.write", "eval(", "new Function("]:
            self.assertNotIn(unsafe, js)


if __name__ == "__main__":
    unittest.main()
