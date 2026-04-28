"""Allow ``python -m inference_optimizer ...``."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
