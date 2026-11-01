"""Start the backend for the desktop app.

    python -m pixforge.server [--port 0] [--device auto]

The token comes from the environment variable PIXFORGE_TOKEN (never the command line, where
other processes could read it). Binds 127.0.0.1 only. When ready, prints one line
``PIXFORGE_READY <port>`` to stdout. Exits when stdin closes, i.e. when Electron is gone,
even if it crashed and could not kill us.
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import secrets
import socket
import sys
import threading

import uvicorn

from pixforge import __version__
from pixforge.paths import logs_dir


def _setup_logging(verbose: bool) -> None:
    logs_dir().mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        logs_dir() / "backend.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    if verbose:
        root.addHandler(logging.StreamHandler(sys.stderr))


def _exit_with_parent() -> None:
    """Block on stdin; when the parent closes it (or dies), terminate immediately."""
    try:
        while sys.stdin.buffer.read(1024):
            pass
    except Exception:
        pass
    logging.getLogger(__name__).info("stdin closed, exiting")
    os._exit(0)


def main() -> int:
    ap = argparse.ArgumentParser(prog="pixforge-server")
    ap.add_argument("--port", type=int, default=0, help="0 = random free port")
    ap.add_argument("--device", default=os.environ.get("PIXFORGE_DEVICE", "auto"))
    ap.add_argument("--no-parent-watch", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    _setup_logging(args.verbose)
    log = logging.getLogger("pixforge.server")
    token = os.environ.get("PIXFORGE_TOKEN") or secrets.token_urlsafe(24)
    if "PIXFORGE_TOKEN" not in os.environ:
        print(f"PIXFORGE_TOKEN {token}", flush=True)  # manual start for development

    import truststore

    truststore.inject_into_ssl()  # model downloads behind TLS-scanning antivirus

    from pixforge.server.app import create_app

    app = create_app(token=token, device=args.device)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(
        socket.SOL_SOCKET,
        socket.SO_EXCLUSIVEADDRUSE
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE")
        else socket.SO_REUSEADDR,
        1,
    )
    sock.bind(("127.0.0.1", args.port))
    port = sock.getsockname()[1]

    config = uvicorn.Config(app, log_level="warning", access_log=False, lifespan="on")
    server = uvicorn.Server(config)

    original_startup = server.startup

    async def startup(sockets=None):  # type: ignore[no-untyped-def]
        await original_startup(sockets=sockets)
        print(f"PIXFORGE_READY {port}", flush=True)
        log.info("Pixforge %s backend ready on 127.0.0.1:%d", __version__, port)

    server.startup = startup  # type: ignore[method-assign]
    if not args.no_parent_watch:
        threading.Thread(target=_exit_with_parent, daemon=True).start()
    server.run(sockets=[sock])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
