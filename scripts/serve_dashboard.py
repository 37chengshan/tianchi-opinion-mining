from __future__ import annotations

import argparse
import os
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve and open the Tianchi Autopilot dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="state directory to serve; defaults to <repo>/dashboard so existing runs keep working",
    )
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--url-path", default="/", help="path shown in the printed/opened URL")
    args = parser.parse_args()

    repo_dashboard = Path(__file__).resolve().parents[1] / "dashboard"
    root = args.root.resolve() if args.root is not None else repo_dashboard
    if not root.exists():
        raise SystemExit(f"dashboard root does not exist: {root}")
    # Every console needs the UI; copy it in so a secondary console is fully
    # self-contained and never shares state files with the primary one.
    index_target = root / "index.html"
    if root != repo_dashboard and not index_target.exists():
        index_source = repo_dashboard / "index.html"
        if index_source.exists():
            index_target.write_text(index_source.read_text(encoding="utf-8"), encoding="utf-8")
    os.chdir(root)
    path = args.url_path if args.url_path.startswith("/") else f"/{args.url_path}"
    url = f"http://{args.host}:{args.port}{path}"

    if not args.no_open:
        def opener() -> None:
            time.sleep(0.6)
            webbrowser.open(url)
        threading.Thread(target=opener, daemon=True).start()

    print(f"Dashboard: {url}", flush=True)
    print(f"Serving state from: {root}", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), SimpleHTTPRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
