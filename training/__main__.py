"""TinkerCloud server entrypoint: `python -m training [--backend B] [--host H] [--port P]`.

Backend selection is explicit (Constitution P3): the flag, else TINKERCLOUD_BACKEND,
else the config default. Everything else comes from the environment via
`training.config.TrainingConfig` and the per-backend config models under
`training/backends/<backend>/config.py`.
"""
import argparse

import uvicorn

from .api import create_app
from .backends.factory import SUPPORTED_BACKENDS
from .config import get_config, set_config


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="python -m training", description="TinkerCloud training API")
    parser.add_argument("--backend", choices=list(SUPPORTED_BACKENDS), default=None,
                        help="training backend (default: TINKERCLOUD_BACKEND, else miles)")
    parser.add_argument("--host", default=None, help="bind host (default: TRAINING_HOST or 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="bind port (default: TRAINING_PORT or 8000)")
    args = parser.parse_args(argv)

    config = get_config()
    if args.backend:
        config.backend.backend_type = args.backend
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    set_config(config)

    uvicorn.run(
        create_app(config),
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower(),
        access_log=config.server.access_log,
    )


if __name__ == "__main__":
    main()
