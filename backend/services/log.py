import json
import linecache
import logging
import os
import sys
import traceback
from collections import deque
from datetime import datetime, timezone
from types import TracebackType
from typing import IO, Any, TextIO
from functools import lru_cache

LEVEL_COLORS = {
    "DEBUG": "\033[36m",  # cyan
    "INFO": "\033[32m",  # green
    "WARNING": "\033[33m",  # yellow
    "ERROR": "\033[31m",  # red
    "CRITICAL": "\033[1;31m",  # bold red
}
RESET = "\033[0m"


def _frames_from_tb(tb: TracebackType | None, cwd: str) -> list[dict[str, Any]]:
    """Walk ``tb`` from inner toward outer; return frames outer → inner."""
    raw: list[dict[str, Any]] = []
    while tb is not None:
        fr = tb.tb_frame
        code = fr.f_code
        path = code.co_filename
        try:
            path = os.path.relpath(path, cwd)
        except ValueError:
            pass
        lineno = tb.tb_lineno
        src = linecache.getline(code.co_filename, lineno) or ""
        raw.append(
            {
                "file": path.replace(os.sep, "/") + ":" + str(lineno),
                "func_name": code.co_name,
                "source_line": src.rstrip("\n\r") or None,
            }
        )
        tb = tb.tb_next
    raw.reverse()
    return raw


def build_error_path(
    exc_info: bool
    | tuple[type[BaseException], BaseException, TracebackType | None]
    | None = True,
    *,
    cwd: str | None = None,
    log_message: str | None = None,
) -> dict[str, Any]:
    """
    Build a JSON-serializable description of the current or given exception.

    ``message`` is the caller-facing log text (e.g. from ``.exception("…")``)
    when ``log_message`` is passed; ``error_message`` is ``str(exc_value)``.

    Top-level ``file`` / ``co_name`` / ``source_line`` refer to the **innermost**
    frame. ``traceback`` lists frames outer → inner. ``cause`` repeats this shape
    for ``raise … from`` (without the outer ``log_message``).
    """
    cwd = cwd or os.getcwd()
    if exc_info is True:
        exc_info = sys.exc_info()
    if exc_info is None or exc_info[1] is None:
        return {
            "type": None,
            "message": log_message,
            "error_message": None,
            "file": None,
            "func_name": None,
            "source_line": None,
            "traceback": [],
        }

    exc_type, exc_value, exc_tb = exc_info
    frames = _frames_from_tb(exc_tb, cwd)
    inner = frames[-1] if frames else {}

    out: dict[str, Any] = {
        "type": exc_type.__name__ if exc_type is not None else None,
        "message": log_message,
        "error_message": str(exc_value),
        "file": inner.get("file"),
        "func_name": inner.get("func_name"),
        "source_line": inner.get("source_line"),
        "traceback": frames,
    }
    cause = getattr(exc_value, "__cause__", None)
    if cause is not None:
        out["cause"] = build_error_path(
            (type(cause), cause, cause.__traceback__),
            cwd=cwd,
        )
    return out


def _record_relpath(record: logging.LogRecord, cwd: str) -> str:
    try:
        return os.path.relpath(record.pathname, cwd)
    except ValueError:
        return record.pathname


class StructuredFormatter(logging.Formatter):
    """One JSON object per log line (no ANSI)."""

    def __init__(self, cwd: str | None = None):
        super().__init__()
        self._cwd = cwd or os.getcwd()

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "message": msg,
            "file": _record_relpath(record, self._cwd).replace(os.sep, "/")
            + ":"
            + str(record.lineno),
            "func_name": record.funcName,
        }
        if record.exc_info and record.exc_info[1] is not None:
            exc_type, exc_value, exc_tb = record.exc_info
            payload["error"] = build_error_path(
                (exc_type, exc_value, exc_tb),
                cwd=self._cwd,
                log_message=msg,
            )
        return json.dumps(payload, ensure_ascii=False)


class AlignedFormatter(logging.Formatter):
    """
    Human-readable aligned lines. If ``json_errors`` and the record has
    ``exc_info``, output is **only** pretty JSON from :func:`build_error_path`.
    """

    def __init__(
        self,
        window: int = 50,
        color: bool = True,
        cwd: str | None = None,
        json_errors: bool = True,
    ):
        super().__init__()
        self._window = window
        self._color = color
        self._cwd = cwd or os.getcwd()
        self._json_errors = json_errors
        self._recent_loc_lens: deque[int] = deque(maxlen=window)
        self._max_level_len = max(len(name) for name in LEVEL_COLORS)

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()

        level = record.levelname
        padded_level = level.ljust(self._max_level_len)
        if self._color:
            color = LEVEL_COLORS.get(level, "")
            level_str = f"{color}{padded_level}{RESET}"
        else:
            level_str = padded_level

        filepath = _record_relpath(record, self._cwd)
        loc = f"{filepath}:{record.lineno}"

        self._recent_loc_lens.append(len(loc))
        max_loc = max(self._recent_loc_lens)
        padded_loc = loc.ljust(max_loc)

        msg = record.getMessage()

        if record.exc_info:
            msg = ""
            exc_type, exc_value, exc_tb = record.exc_info
            if exc_value is not None:
                if self._json_errors:
                    payload = build_error_path(
                        (exc_type, exc_value, exc_tb),
                        cwd=self._cwd,
                        log_message=record.getMessage(),
                    )
                    msg += "\n" + json.dumps(payload, indent=2, ensure_ascii=False)
                else:
                    chain = traceback.format_exception(
                        exc_type, exc_value, exc_tb, chain=True
                    )
                    msg += "\n" + "".join(chain).rstrip("\n")

        return f"[{ts}] [{level_str}] [{padded_loc}]  {msg}"


class DailyPathFileHandler(logging.Handler):
    """Append to ``{folder}/{YYYY_MM_DD}.{ext}``, rolling over at UTC midnight."""

    def __init__(self, folder: str, extension: str, encoding: str = "utf-8") -> None:
        super().__init__()
        self.terminator = "\n"
        self._folder = folder
        self._extension = extension.lstrip(".")
        self.encoding = encoding
        self._current_path: str | None = None
        self._stream: IO[str] | None = None

    def emit(self, record: logging.LogRecord) -> None:
        day = datetime.now(timezone.utc).strftime("%Y_%m_%d")
        path = os.path.join(self._folder, f"{day}.{self._extension}")
        try:
            if path != self._current_path:
                if self._stream is not None:
                    self._stream.close()
                    self._stream = None
                os.makedirs(self._folder, exist_ok=True)
                self._stream = open(path, "a", encoding=self.encoding)
                self._current_path = path
            assert self._stream is not None
            msg = self.format(record)
            self._stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
            self._current_path = None
        super().close()


def _formatter_for_stream(
    *,
    structured: bool,
    window: int,
    color: bool,
    cwd: str | None,
    json_errors: bool,
    stream: TextIO,
) -> logging.Formatter:
    cwd_resolved = cwd or os.getcwd()
    if structured:
        return StructuredFormatter(cwd=cwd_resolved)
    use_color = color and stream.isatty()
    return AlignedFormatter(
        window=window,
        color=use_color,
        cwd=cwd_resolved,
        json_errors=json_errors,
    )


def _formatter_for_file(
    *,
    structured_in_file: bool,
    window: int,
    cwd: str | None,
) -> logging.Formatter:
    cwd_resolved = cwd or os.getcwd()
    if structured_in_file:
        return StructuredFormatter(cwd=cwd_resolved)
    return AlignedFormatter(
        window=window,
        color=False,
        cwd=cwd_resolved,
        json_errors=False,
    )


class Logger:
    """Configure stdlib logging with optional stdout, stderr, and daily file."""

    def __init__(
        self,
        name: str | None = None,
        level: int = logging.DEBUG,
        window: int = 50,
        color: bool = True,
        json_errors: bool = True,
        cwd: str | None = None,
        stdout: bool = True,
        stderr: bool = False,
        structured: bool = False,
        structured_in_file: bool = True,
        log_folder: str | None = None,
    ):
        self._logger = logging.getLogger(name or __name__)
        self._logger.setLevel(level)
        self._logger.propagate = False

        if not stdout and not stderr and log_folder is None:
            stdout = True

        if not self._logger.handlers:
            if stdout:
                h = logging.StreamHandler(sys.stdout)
                h.setFormatter(
                    _formatter_for_stream(
                        structured=structured,
                        window=window,
                        color=color,
                        cwd=cwd,
                        json_errors=json_errors,
                        stream=sys.stdout,
                    )
                )
                self._logger.addHandler(h)
            if stderr:
                h = logging.StreamHandler(sys.stderr)
                h.setFormatter(
                    _formatter_for_stream(
                        structured=structured,
                        window=window,
                        color=color,
                        cwd=cwd,
                        json_errors=json_errors,
                        stream=sys.stderr,
                    )
                )
                self._logger.addHandler(h)
            if log_folder is not None:
                ext = "json" if structured_in_file else "log"
                fh = DailyPathFileHandler(log_folder, ext)
                fh.setFormatter(
                    _formatter_for_file(
                        structured_in_file=structured_in_file,
                        window=window,
                        cwd=cwd,
                    )
                )
                self._logger.addHandler(fh)

    def debug(self, msg: str, *args, stacklevel: int = 2, **kwargs):
        self._logger.debug(msg, *args, stacklevel=stacklevel, **kwargs)

    def info(self, msg: str, *args, stacklevel: int = 2, **kwargs):
        self._logger.info(msg, *args, stacklevel=stacklevel, **kwargs)

    def warning(self, msg: str, *args, stacklevel: int = 2, **kwargs):
        self._logger.warning(msg, *args, stacklevel=stacklevel, **kwargs)

    def error(self, msg: str, *args, stacklevel: int = 2, **kwargs):
        self._logger.error(msg, *args, stacklevel=stacklevel, **kwargs)

    def exception(self, msg: str, *args, stacklevel: int = 2, **kwargs):
        """Log at ERROR with traceback. Call inside ``except`` (uses ``exc_info=True``)."""
        kwargs.setdefault("exc_info", True)
        self._logger.error(msg, *args, stacklevel=stacklevel, **kwargs)

    def critical(self, msg: str, *args, stacklevel: int = 2, **kwargs):
        self._logger.critical(msg, *args, stacklevel=stacklevel, **kwargs)


@lru_cache(maxsize=None)
def get_logger(
    name: str | None = None,
    level: int = logging.DEBUG,
    window: int = 50,
    color: bool = True,
    json_errors: bool = True,
    cwd: str | None = None,
    stdout: bool = True,
    stderr: bool = False,
    structured: bool = False,
    structured_in_file: bool = True,
    log_folder: str | None = None,
) -> Logger:
    return Logger(
        name=name,
        level=level,
        window=window,
        color=color,
        json_errors=json_errors,
        cwd=cwd,
        stdout=stdout,
        stderr=stderr,
        structured=structured,
        structured_in_file=structured_in_file,
        log_folder=log_folder,
    )


def main():
    logger = get_logger()
    logger.info("Hello, world!")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")




if __name__ == "__main__":
    main()
