"""libcst-based AST scanner for framework source flag discovery.

Three patterns cover ~99% of vllm/sglang engine flags (design §9.3):

* **argparse** -- ``parser.add_argument("--max-model-len", type=int, ...)``
* **dataclass field** -- ``@dataclass class EngineArgs: max_num_seqs: int = 256``
* **pydantic BaseModel field** -- ``class SchedulerConfig(BaseModel): ...``

Per-file failure isolation: when ``libcst.parse_module`` raises, we
fall back to :func:`grep_scanner.scan_module_via_grep` for that file
only; other files keep their AST results. The aggregate ``mode`` flag
in :class:`AstScanResult` is set to ``"grep_fallback"`` when >=10% of
files needed the fallback (KB confidence downgrade signal).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import libcst as cst
import libcst.metadata as cst_meta

from .flag_discovery import DiscoveredFlag, dedup_and_rank, path_to_module
from .grep_scanner import scan_module_via_grep
from .source_resolver import collect_target_files


# Aggregate-mode threshold: if more than this fraction of files fall back
# to grep, the whole scan is labelled "grep_fallback" so KB writes
# downgrade confidence.
_GREP_FALLBACK_RATIO_THRESHOLD = 0.10


@dataclass(frozen=True)
class AstScanResult:
    """Output of :func:`scan_framework_args`."""

    flags: list[DiscoveredFlag] = field(default_factory=list)
    mode: Literal["libcst", "grep_fallback"] = "libcst"
    files_scanned: int = 0
    parse_failures: int = 0
    failed_files: list[tuple[str, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# libcst helpers
# ---------------------------------------------------------------------------
def _get_attr_name(node: cst.BaseExpression) -> str:
    """Return the trailing attribute name of an Attribute / Name node."""
    if isinstance(node, cst.Attribute):
        return node.attr.value
    if isinstance(node, cst.Name):
        return node.value
    return ""


def _kw_arg(args: list, kw: str) -> cst.BaseExpression | None:
    """Return the value node for a keyword argument named ``kw``."""
    for a in args:
        keyword = getattr(a, "keyword", None)
        if isinstance(keyword, cst.Name) and keyword.value == kw:
            return a.value
    return None


def _stringify(node: cst.CSTNode | None) -> str:
    """Render a libcst node back to source code (best-effort)."""
    if node is None:
        return ""
    try:
        return cst.Module(body=[]).code_for_node(node).strip()
    except Exception:  # noqa: BLE001
        return ""


def _has_dataclass_decorator(decorators: list) -> bool:
    """Check whether a ClassDef's decorator list contains @dataclass."""
    for deco in decorators:
        target = deco.decorator
        if isinstance(target, cst.Call):
            target = target.func
        name = _get_attr_name(target)
        if name == "dataclass":
            return True
    return False


def _has_basemodel_base(bases: list) -> bool:
    """Check whether a ClassDef inherits from pydantic BaseModel."""
    for b in bases:
        name = _get_attr_name(b.value)
        if name == "BaseModel":
            return True
    return False


# ---------------------------------------------------------------------------
# Collector visitor
# ---------------------------------------------------------------------------
class _ArgScanCollector(cst.CSTVisitor):
    """Walk one module, collect DiscoveredFlag across the 3 patterns."""

    METADATA_DEPENDENCIES = (cst_meta.PositionProvider,)

    def __init__(
        self,
        source_path: Path,
        framework: str,
        source_root: Path | None,
    ) -> None:
        super().__init__()
        self.found: list[DiscoveredFlag] = []
        self.source_path = source_path
        self.framework = framework
        self.source_root = source_root
        self._stack: list[tuple[str, bool, bool]] = []
        # tuple per nested ClassDef: (class_name, is_dataclass, is_basemodel)

    @property
    def _in_dc_or_basemodel(self) -> bool:
        return any(is_dc or is_pyd for _, is_dc, is_pyd in self._stack)

    @property
    def _current_class(self) -> str:
        return self._stack[-1][0] if self._stack else ""

    # --------------- pattern A: argparse.add_argument -----------------
    def visit_Call(self, node: cst.Call) -> None:
        attr = _get_attr_name(node.func)
        if attr != "add_argument" or not node.args:
            return
        first = node.args[0].value
        if not isinstance(first, cst.SimpleString):
            return
        raw = first.value.strip("'\"")
        if not raw.startswith("--"):
            return
        type_node = _kw_arg(list(node.args), "type")
        default_node = _kw_arg(list(node.args), "default")
        help_node = _kw_arg(list(node.args), "help")
        try:
            pos = self.get_metadata(cst_meta.PositionProvider, node).start
            line = pos.line
        except KeyError:
            line = 0
        help_text = ""
        if isinstance(help_node, cst.SimpleString):
            help_text = help_node.value.strip("'\"")[:160]
        self.found.append(DiscoveredFlag(
            flag_name=raw,
            module=path_to_module(self.source_path, self.source_root) if self.source_root else self.source_path.stem,
            source_path=str(self.source_path),
            line=line,
            via="argparse",
            type_hint=_stringify(type_node) or "str",
            default_repr=_stringify(default_node) or "_MISSING_",
            help_text=help_text,
            surface="cli",
            framework=self.framework,
        ))

    # --------------- pattern B/C: ClassDef + AnnAssign ----------------
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        is_dc = _has_dataclass_decorator(list(node.decorators))
        is_pyd = _has_basemodel_base(list(node.bases))
        self._stack.append((node.name.value, is_dc, is_pyd))

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        if self._stack:
            self._stack.pop()

    def visit_AnnAssign(self, node: cst.AnnAssign) -> None:
        if not self._in_dc_or_basemodel:
            return
        if not isinstance(node.target, cst.Name):
            return
        name = node.target.value
        if name.startswith("_"):
            return  # skip private / dunder fields
        # Distinguish dataclass vs pydantic by the innermost class flags.
        _, is_dc, is_pyd = self._stack[-1]
        via: str = "dataclass" if is_dc else ("pydantic" if is_pyd else "dataclass")
        try:
            pos = self.get_metadata(cst_meta.PositionProvider, node).start
            line = pos.line
        except KeyError:
            line = 0
        self.found.append(DiscoveredFlag(
            flag_name=name,
            module=path_to_module(self.source_path, self.source_root) if self.source_root else self.source_path.stem,
            source_path=str(self.source_path),
            line=line,
            via=via,  # type: ignore[arg-type]
            type_hint=_stringify(node.annotation.annotation) or "str",
            default_repr=_stringify(node.value) if node.value else "_MISSING_",
            help_text=self._current_class[:60],
            surface="config",
            framework=self.framework,
        ))


# ---------------------------------------------------------------------------
# Per-file + per-scan drivers
# ---------------------------------------------------------------------------
def scan_module(
    path: Path,
    *,
    framework: str,
    source_root: Path | None = None,
) -> tuple[list[DiscoveredFlag], str | None]:
    """Scan one file. Returns (flags, error-string-or-None).

    Errors are returned (not raised) so callers can keep going through
    the remaining files and aggregate failure counts.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], f"read_failed: {exc}"
    except UnicodeDecodeError as exc:
        return [], f"encoding_failed: {exc}"
    try:
        module = cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        return [], f"libcst_parse_failed: {exc}"
    wrapper = cst_meta.MetadataWrapper(module)
    collector = _ArgScanCollector(path, framework, source_root)
    wrapper.visit(collector)
    return collector.found, None


def scan_framework_args(
    framework: str,
    source_root: Path,
) -> AstScanResult:
    """Scan ``source_root`` for one framework. Aggregates per-file results.

    Sequential by default -- libcst parse is CPU-bound but a typical
    vllm/sglang tree is small enough (1-3s single-threaded). PR-F may
    later add a process pool when measurement shows benefit.
    """
    files = collect_target_files(framework, source_root)
    all_flags: list[DiscoveredFlag] = []
    failed_files: list[tuple[str, str]] = []
    for f in files:
        flags, err = scan_module(f, framework=framework, source_root=source_root)
        if err is None:
            all_flags.extend(flags)
        else:
            # Per-file fallback: try grep on this single file before
            # giving up on it entirely.
            grep_flags = scan_module_via_grep(
                f, framework=framework, source_root=source_root,
            )
            all_flags.extend(grep_flags)
            failed_files.append((str(f), err))
    n = max(1, len(files))
    fallback_ratio = len(failed_files) / n
    mode: Literal["libcst", "grep_fallback"] = (
        "grep_fallback"
        if fallback_ratio >= _GREP_FALLBACK_RATIO_THRESHOLD
        else "libcst"
    )
    return AstScanResult(
        flags=dedup_and_rank(all_flags),
        mode=mode,
        files_scanned=len(files),
        parse_failures=len(failed_files),
        failed_files=failed_files,
    )


__all__ = [
    "AstScanResult",
    "scan_framework_args",
    "scan_module",
]
