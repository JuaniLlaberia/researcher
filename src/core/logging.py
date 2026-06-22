"""
Structured, colored console logging for the research system.

One logger per component (`researcher.<component>`), each rendered with a fixed
label and color so a live run reads as a narrative of who is doing what. Uses the
stdlib `logging` only — no third-party dependency. Color auto-disables when the
output is not a TTY or `NO_COLOR` is set, so piped/file output stays clean.
"""
import logging
import sys
import time
from contextlib import contextmanager
from os import environ

# --- ANSI palette ---------------------------------------------------------
_RESET = "\033[0m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_BLUE = "\033[34m"
_MAGENTA = "\033[35m"
_GREY = "\033[90m"
_BRIGHT_YELLOW = "\033[93m"
_WHITE = "\033[37m"

# component (logger suffix) -> (label, color)
COMPONENT_STYLES: dict[str, tuple[str, str]] = {
    "orchestrator": ("ORCH", _BRIGHT_YELLOW),
    "hypothesizer": ("HYPO", _CYAN),
    "literature_reviewer": ("LIT", _GREEN),
    "critic": ("CRITIC", _RED),
    "writer": ("WRITE", _BLUE),
    "searcher": ("SEARCH", _GREY),
    "memory": ("MEM", _MAGENTA),
    "db": ("DB", _DIM + _WHITE),
}

_LEVEL_COLORS = {
    "DEBUG": _DIM,
    "INFO": _GREEN,
    "WARNING": _BRIGHT_YELLOW,
    "ERROR": _RED,
    "CRITICAL": _RED,
}

# Third-party loggers that would otherwise drown out ours.
_NOISY = ("httpx", "httpcore", "urllib3", "langchain", "langchain_core", "transformers", "docling", "asyncio")

_configured = False

class ColorFormatter(logging.Formatter):
    """Renders `HH:MM:SS LEVEL [LABEL] message`, colored by component."""
    def __init__(self, use_color: bool):
        super().__init__(datefmt="%H:%M:%S")
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        component = record.name.split(".")[-1]
        label, color = COMPONENT_STYLES.get(component, (component.upper(), _WHITE))
        ts = self.formatTime(record, self.datefmt)
        message = record.getMessage()

        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"

        if not self.use_color:
            return f"{ts} {record.levelname:<7} [{label}] {message}"

        lvl_color = _LEVEL_COLORS.get(record.levelname, _WHITE)
        return (
            f"{_DIM}{ts}{_RESET} "
            f"{lvl_color}{record.levelname:<7}{_RESET} "
            f"{color}[{label}]{_RESET} {message}"
        )

def setup_logging(level: str = "INFO", log_file: str | None = None) -> None:
    """
    Configure the `researcher` logger tree once. Idempotent — safe to call from
    both the CLI entrypoint and agent constructors.

    Args:
        level (str): Root level for our loggers ("INFO", "DEBUG", ...).
        log_file (str | None): If set, also write a plain (no-color) transcript here.
    """
    global _configured
    if _configured:
        return

    use_color = sys.stderr.isatty() and "NO_COLOR" not in environ

    root = logging.getLogger("researcher")
    root.setLevel(level.upper())
    root.propagate = False

    # Real-time console: StreamHandler flushes on every emit.
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(ColorFormatter(use_color=use_color))
    root.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(ColorFormatter(use_color=False))
        root.addHandler(file_handler)

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True

def get_logger(component: str) -> logging.Logger:
    """
    Return the logger for a component (e.g. "hypothesizer", "orchestrator").

    Args:
        component (str): Component key; should match a COMPONENT_STYLES entry.
    Returns:
        logging.Logger: The namespaced logger.
    """
    return logging.getLogger(f"researcher.{component}")

@contextmanager
def log_stage(logger: logging.Logger, message: str, level: int = logging.INFO):
    """
    Time a unit of work and log its start and elapsed duration. Use around heavy
    steps (agent runs, ingestion, retrieval) to surface where time goes.

    Args:
        logger (logging.Logger): The component logger.
        message (str): Human label for the stage, e.g. "literature review".
        level (int): Logging level for the start/done lines.
    """
    logger.log(level, "%s …", message)
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.log(level, "%s done in %.2fs", message, time.perf_counter() - start)
