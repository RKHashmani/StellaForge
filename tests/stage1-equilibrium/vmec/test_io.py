"""I/O validation checks for the Stage 1 VMEC contract.

This file is intentionally standalone and lives at the repository root.  It does
not import vmec_jax or JAX; pass it simple dictionary-like payloads from a
parser, NetCDF reader, or workflow wrapper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from dataclasses import dataclass
from typing import Any


REQUIRED_INPUT_SCALARS = ("NFP", "MPOL", "NTOR", "PHIEDGE")
REQUIRED_INPUT_INDEXED = ("RBC", "ZBS")
REQUIRED_WOUT_SCALARS = (
    "ns",
    "mpol",
    "ntor",
    "nfp",
    "mnmax",
    "mnmax_nyq",
    "wb",
    "volume_p",
    "fsqr",
    "fsqz",
    "fsql",
    "aspect",
    "Aminor_p",
    "Rmajor_p",
)
REQUIRED_WOUT_MAIN_FIELDS = ("xm", "xn", "rmnc", "rmns", "zmnc", "zmns", "lmnc", "lmns")
REQUIRED_WOUT_NYQUIST_FIELDS = (
    "xm_nyq",
    "xn_nyq",
    "gmnc",
    "gmns",
    "bsupumnc",
    "bsupumns",
    "bsupvmnc",
    "bsupvmns",
    "bsubumnc",
    "bsubumns",
    "bsubvmnc",
    "bsubvmns",
    "bsubsmns",
    "bsubsmnc",
    "bmnc",
    "bmns",
)
REQUIRED_WOUT_RADIAL_FIELDS = (
    "phipf",
    "chipf",
    "phips",
    "iotaf",
    "iotas",
    "phi",
    "vp",
    "pres",
    "presf",
    "buco",
    "bvco",
    "jcuru",
    "jcurv",
)
_ASSIGN_RE = re.compile(r"(?P<key>[A-Za-z_]\w*(?:\([^\)]*\))?)\s*=", re.MULTILINE)
_REPEAT_RE = re.compile(r"^(?P<count>\d+)\*(?P<value>.+)$")


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def _value(mapping: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in mapping:
        return mapping[key]
    upper = key.upper()
    if upper in mapping:
        return mapping[upper]
    lower = key.lower()
    if lower in mapping:
        return mapping[lower]
    return default


def _contains(mapping: dict[str, Any], key: str) -> bool:
    return _value(mapping, key) is not None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if hasattr(value, "shape") and hasattr(value, "ravel"):
        return list(value.ravel())
    return [value]


def _shape(value: Any) -> tuple[int, ...] | None:
    if hasattr(value, "shape"):
        return tuple(int(x) for x in value.shape)
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return (len(value), len(value[0]))
        return (len(value),)
    return None


def _positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except Exception:
        return False


def _strip_fortran_comment(line: str) -> str:
    out: list[str] = []
    in_quote = False
    for ch in line:
        if ch == "'":
            in_quote = not in_quote
            out.append(ch)
        elif ch == "!" and not in_quote:
            break
        else:
            out.append(ch)
    return "".join(out)


def _tokenize_values(text: str) -> list[str]:
    tokens: list[str] = []
    buf: list[str] = []
    in_quote = False
    for ch in text.strip():
        if ch == "'":
            in_quote = not in_quote
            buf.append(ch)
        elif not in_quote and ch in {",", " ", "\n", "\t", "\r"}:
            if buf:
                token = "".join(buf).strip()
                if token:
                    tokens.append(token)
                buf = []
        else:
            buf.append(ch)
    if buf:
        token = "".join(buf).strip()
        if token:
            tokens.append(token)
    expanded: list[str] = []
    for token in tokens:
        match = _REPEAT_RE.match(token)
        if match and int(match.group("count")) > 0:
            expanded.extend([match.group("value").strip()] * int(match.group("count")))
        else:
            expanded.append(token)
    return expanded


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if len(token) >= 2 and token[0] == "'" and token[-1] == "'":
        return token[1:-1]
    upper = token.upper()
    if upper in {".TRUE.", "TRUE", "T", ".T."}:
        return True
    if upper in {".FALSE.", "FALSE", "F", ".F."}:
        return False
    if re.fullmatch(r"[+-]?\d+", token):
        try:
            return int(token)
        except Exception:
            pass
    try:
        return float(token.replace("D", "E").replace("d", "E"))
    except Exception:
        return token


def _parse_key(key: str) -> tuple[str, tuple[int, ...] | None]:
    key = key.strip()
    if "(" not in key:
        return key.upper(), None
    base, rest = key.split("(", 1)
    rest = rest.rstrip(")")
    if ":" in rest:
        return base.upper(), None
    return base.upper(), tuple(int(part.strip()) for part in rest.split(",") if part.strip())


def read_indata_payload(path: str | Path) -> dict[str, Any]:
    """Read enough of a VMEC &INDATA file to run the root-level validator."""

    path = Path(path)
    text = path.read_text()
    start = re.search(r"&\s*INDATA", text, flags=re.IGNORECASE)
    if not start:
        raise ValueError(f"{path} does not contain an &INDATA block")
    end = re.search(r"\n\s*/\s*(?:\n|$)", text[start.end() :], flags=re.MULTILINE)
    if not end:
        raise ValueError(f"{path} does not terminate the &INDATA block with '/'")
    block = text[start.end() : start.end() + end.start()]
    cleaned = "\n".join(_strip_fortran_comment(line) for line in block.splitlines())

    scalars: dict[str, Any] = {}
    indexed: dict[str, dict[tuple[int, ...], Any]] = {}
    matches = list(_ASSIGN_RE.finditer(cleaned))
    for idx, match in enumerate(matches):
        key, key_index = _parse_key(match.group("key"))
        value_start = match.end()
        value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
        chunk = re.sub(r",\s*$", "", cleaned[value_start:value_end].strip())
        values = [_parse_scalar(token) for token in _tokenize_values(chunk)]
        if not values:
            continue
        value: Any = values[0] if len(values) == 1 else values
        if key_index is None:
            scalars[key] = value
        else:
            indexed.setdefault(key, {})[key_index] = value
    return {"scalars": scalars, "indexed": indexed}


def print_report(label: str, report: ValidationReport) -> None:
    if report.ok:
        print(f"{label}: OK")
    else:
        print(f"{label}: FAILED")
    for warning in report.warnings:
        print(f"  warning: {warning}")
    for error in report.errors:
        print(f"  error: {error}")


def validate_input_payload(payload: dict[str, Any]) -> ValidationReport:
    """Validate a parsed INDATA-like payload.

    Expected format:
      {"scalars": {"NFP": 2, ...}, "indexed": {"RBC": {(0, 0): 1.0}, ...}}
    """

    scalars = dict(payload.get("scalars", payload))
    indexed = dict(payload.get("indexed", {}))
    errors: list[str] = []
    warnings: list[str] = []

    for field in REQUIRED_INPUT_SCALARS:
        if _value(scalars, field) is None:
            errors.append(f"missing required input scalar {field}")
    if _value(scalars, "NS_ARRAY") is None and _value(scalars, "NS") is None:
        errors.append("missing required input radial resolution NS_ARRAY or NS")
    for field in REQUIRED_INPUT_INDEXED:
        if not indexed.get(field):
            errors.append(f"missing required input boundary coefficients {field}(n,m)")

    for field in ("NFP", "MPOL"):
        value = _value(scalars, field)
        if value is not None and not _positive_int(value):
            errors.append(f"{field} must be a positive integer")
    ntor = _value(scalars, "NTOR")
    if ntor is not None:
        try:
            if int(ntor) < 0:
                errors.append("NTOR must be non-negative")
        except Exception:
            errors.append("NTOR must be an integer")
    phiedge = _value(scalars, "PHIEDGE")
    if phiedge is not None:
        try:
            if float(phiedge) == 0.0:
                errors.append("PHIEDGE must be nonzero")
        except Exception:
            errors.append("PHIEDGE must be numeric")

    ns_values = _as_list(_value(scalars, "NS_ARRAY", _value(scalars, "NS")))
    for idx, value in enumerate(ns_values):
        try:
            if int(value) < 3:
                errors.append(f"NS_ARRAY[{idx}] must be at least 3")
        except Exception:
            errors.append(f"NS_ARRAY[{idx}] must be an integer")
    for field in ("NITER_ARRAY", "FTOL_ARRAY"):
        values = _as_list(_value(scalars, field))
        if values and ns_values and len(values) not in (1, len(ns_values)):
            warnings.append(f"{field} length does not match NS_ARRAY; VMEC may ignore or truncate extra entries")

    rbc = indexed.get("RBC", {})
    if rbc and (0, 0) not in rbc:
        warnings.append("RBC(0,0) is absent; major-radius offset may default to zero")

    return ValidationReport(errors=tuple(errors), warnings=tuple(warnings))


def validate_wout_payload(payload: dict[str, Any]) -> ValidationReport:
    """Validate a WOUT-like dictionary for required fields and basic shapes."""

    errors: list[str] = []
    for field in REQUIRED_WOUT_SCALARS:
        if not _contains(payload, field):
            errors.append(f"missing required wout scalar {field}")
    if errors:
        return ValidationReport(errors=tuple(errors))

    try:
        ns = int(_value(payload, "ns"))
        mnmax = int(_value(payload, "mnmax"))
        mnmax_nyq = int(_value(payload, "mnmax_nyq"))
        ntor = int(_value(payload, "ntor"))
    except Exception:
        return ValidationReport(errors=("wout scalar metadata ns, mnmax, mnmax_nyq, and ntor must be integers",))
    if ns <= 0 or mnmax <= 0 or mnmax_nyq <= 0 or ntor < 0:
        return ValidationReport(
            errors=("wout scalar metadata ns, mnmax, and mnmax_nyq must be positive; ntor must be non-negative",)
        )

    for field in ("xm", "xn"):
        if _shape(_value(payload, field)) != (mnmax,):
            errors.append(f"{field} must have shape ({mnmax},)")
    for field in ("xm_nyq", "xn_nyq"):
        if _shape(_value(payload, field)) != (mnmax_nyq,):
            errors.append(f"{field} must have shape ({mnmax_nyq},)")
    for field in REQUIRED_WOUT_MAIN_FIELDS[2:]:
        if _shape(_value(payload, field)) != (ns, mnmax):
            errors.append(f"{field} must have shape ({ns}, {mnmax})")
    for field in REQUIRED_WOUT_NYQUIST_FIELDS[2:]:
        if _shape(_value(payload, field)) != (ns, mnmax_nyq):
            errors.append(f"{field} must have shape ({ns}, {mnmax_nyq})")
    for field in REQUIRED_WOUT_RADIAL_FIELDS:
        if _shape(_value(payload, field)) != (ns,):
            errors.append(f"{field} must have shape ({ns},)")
    for field in ("raxis_cc", "zaxis_cs", "raxis_cs", "zaxis_cc"):
        if _contains(payload, field) and _shape(_value(payload, field)) != (ntor + 1,):
            errors.append(f"{field} must have shape ({ntor + 1},)")

    return ValidationReport(errors=tuple(errors))


def _zeros(shape: tuple[int, ...]) -> Any:
    if len(shape) == 1:
        return [0.0] * shape[0]
    return [[0.0] * shape[1] for _ in range(shape[0])]


def _valid_wout_payload() -> dict[str, Any]:
    ns, mnmax, mnmax_nyq, ntor = 4, 3, 5, 2
    payload: dict[str, Any] = {
        "ns": ns,
        "mpol": 3,
        "ntor": ntor,
        "nfp": 2,
        "mnmax": mnmax,
        "mnmax_nyq": mnmax_nyq,
        "wb": 0.0,
        "volume_p": 1.0,
        "fsqr": 0.0,
        "fsqz": 0.0,
        "fsql": 0.0,
        "aspect": 5.0,
        "Aminor_p": 0.2,
        "Rmajor_p": 1.0,
    }
    for field in ("xm", "xn"):
        payload[field] = _zeros((mnmax,))
    for field in ("xm_nyq", "xn_nyq"):
        payload[field] = _zeros((mnmax_nyq,))
    for field in REQUIRED_WOUT_MAIN_FIELDS[2:]:
        payload[field] = _zeros((ns, mnmax))
    for field in REQUIRED_WOUT_NYQUIST_FIELDS[2:]:
        payload[field] = _zeros((ns, mnmax_nyq))
    for field in REQUIRED_WOUT_RADIAL_FIELDS:
        payload[field] = _zeros((ns,))
    for field in ("raxis_cc", "zaxis_cs", "raxis_cs", "zaxis_cc"):
        payload[field] = _zeros((ntor + 1,))
    return payload


def test_valid_input_payload() -> None:
    payload = {
        "scalars": {"NFP": 2, "MPOL": 5, "NTOR": 4, "NS_ARRAY": [15, 31], "PHIEDGE": 0.083},
        "indexed": {"RBC": {(0, 0): 1.0, (0, 1): 0.2}, "ZBS": {(0, 1): 0.2}},
    }
    assert validate_input_payload(payload).ok


def test_invalid_input_payload_reports_missing_fields() -> None:
    payload = {"scalars": {"NFP": 1, "MPOL": 5, "NTOR": 0}, "indexed": {"RBC": {(0, 0): 1.0}}}
    report = validate_input_payload(payload)
    assert not report.ok
    assert "missing required input scalar PHIEDGE" in report.errors
    assert "missing required input radial resolution NS_ARRAY or NS" in report.errors
    assert "missing required input boundary coefficients ZBS(n,m)" in report.errors


def test_repo_minimal_seed_input_file_validates() -> None:
    path = Path(__file__).resolve().parent / "examples" / "data" / "input.minimal_seed_nfp2"
    report = validate_input_payload(read_indata_payload(path))
    assert report.ok


def test_repo_multiline_boundary_input_file_validates() -> None:
    path = Path(__file__).resolve().parent / "examples" / "data" / "input.cth_like_fixed_bdy"
    report = validate_input_payload(read_indata_payload(path))
    assert report.ok


def test_valid_wout_payload() -> None:
    assert validate_wout_payload(_valid_wout_payload()).ok


def test_invalid_wout_payload_reports_bad_shapes() -> None:
    payload = _valid_wout_payload()
    payload["rmnc"] = _zeros((payload["ns"], payload["mnmax"] + 1))
    report = validate_wout_payload(payload)
    assert not report.ok
    assert "rmnc must have shape (4, 3)" in report.errors


def _run_self_tests() -> None:
    tests = (
        test_valid_input_payload,
        test_invalid_input_payload_reports_missing_fields,
        test_repo_minimal_seed_input_file_validates,
        test_repo_multiline_boundary_input_file_validates,
        test_valid_wout_payload,
        test_invalid_wout_payload_reports_bad_shapes,
    )
    for test in tests:
        test()
    print("I/O validation checks passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Stage 1 VMEC input/output payloads.")
    parser.add_argument(
        "input_file",
        nargs="?",
        help="Optional VMEC &INDATA file, e.g. input.* or vmec_input.*. If omitted, run the built-in self-checks.",
    )
    args = parser.parse_args(argv)
    if args.input_file is None:
        _run_self_tests()
        return 0

    payload = read_indata_payload(args.input_file)
    report = validate_input_payload(payload)
    print_report(str(args.input_file), report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
