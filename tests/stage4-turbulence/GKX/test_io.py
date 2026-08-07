"""
Run IO validation checks

run with 
python run_io_validation_checks.py --config path/to/runtime.toml --out path/to/output_dir

Outputs a json file of summaries, including missing params, and pass/fail flag
"""

from __future__ import annotations

import argparse
import importlib
import json
from dataclasses import fields, is_dataclass, MISSING
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

def _load_toml_module() -> Any:
    try:
        return importlib.import_module("tomllib")
    except ModuleNotFoundError:
        try:
            return importlib.import_module("tomli")
        except ModuleNotFoundError as exc:
            raise SystemExit("Need a TOML parser: install Python>=3.11 or `pip install tomli`") from exc

from gkx.config import GeometryConfig, GridConfig, InitializationConfig, TimeConfig
from gkx.runtime_config import (
    RuntimeCollisionConfig,
    RuntimeExpertConfig,
    RuntimeNormalizationConfig,
    RuntimePhysicsConfig,
    RuntimeSpeciesConfig,
    RuntimeTermsConfig,
)

def _default_from_dataclass(cls: type[Any]) -> dict[str, Any]:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    out: dict[str, Any] = {}
    for f in fields(cls):
        if f.default is not MISSING:
            out[f.name] = f.default
        elif f.default_factory is not MISSING:
            out[f.name] = f.default_factory()
        else:
            out[f.name] = None
    return out


def _annotation_accepts(annotation: Any, value: Any) -> bool:
    if annotation is Any:
        return True

    origin = get_origin(annotation)
    if origin is None:
        if annotation is type(None):
            return value is None
        if annotation is bool:
            return isinstance(value, bool)
        if annotation is int:
            return isinstance(value, int) and not isinstance(value, bool)
        if annotation is float:
            return (isinstance(value, float) or isinstance(value, int)) and not isinstance(value, bool)
        if annotation is str:
            return isinstance(value, str)
        if annotation in (dict, list, tuple):
            return isinstance(value, annotation)
        return True

    if origin in (list, tuple):
        return isinstance(value, origin)
    if origin is dict:
        return isinstance(value, dict)

    args = get_args(annotation)
    if origin in (Union, UnionType):
        return any(_annotation_accepts(arg, value) for arg in args)

    return True


def _validate_section_dict(
    name: str,
    section: dict[str, Any],
    cls: type[Any],
    errors: list[str],
    warnings: list[str],
) -> None:
    schema = {f.name: f.type for f in fields(cls)}
    for key, value in section.items():
        if key not in schema:
            warnings.append(f"[{name}] unknown key: {key}")
            continue
        if not _annotation_accepts(schema[key], value):
            errors.append(f"[{name}] key '{key}' has incompatible type: {type(value).__name__}")


def _validate_toml_schema(data: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    norm = dict(data)

    if "experts" in norm:
        errors.append("Use [expert] (singular). [experts] is not valid in GKX convention.")

    required_sections = ["species", "geometry", "physics", "run"]
    missing_required = [name for name in required_sections if name not in norm]
    if missing_required:
        errors.append(f"Missing required sections: {', '.join(missing_required)}")

    if "species" in norm:
        species = norm["species"]
        if not isinstance(species, list) or len(species) == 0:
            errors.append("[[species]] must be a non-empty array of tables")
        else:
            for i, sp in enumerate(species):
                if not isinstance(sp, dict):
                    errors.append(f"[[species]] entry {i} is not a table")
                    continue
                _validate_section_dict(f"species[{i}]", sp, RuntimeSpeciesConfig, errors, warnings)

    section_map: dict[str, type[Any]] = {
        "grid": GridConfig,
        "time": TimeConfig,
        "geometry": GeometryConfig,
        "init": InitializationConfig,
        "physics": RuntimePhysicsConfig,
        "collisions": RuntimeCollisionConfig,
        "normalization": RuntimeNormalizationConfig,
        "terms": RuntimeTermsConfig,
        "expert": RuntimeExpertConfig,
    }

    for section_name, cls in section_map.items():
        if section_name not in norm:
            continue
        section = norm[section_name]
        if not isinstance(section, dict):
            errors.append(f"[{section_name}] must be a table")
            continue
        _validate_section_dict(section_name, section, cls, errors, warnings)

    if "run" in norm and not isinstance(norm["run"], dict):
        errors.append("[run] must be a table")

    defaults = {
        "species": [_default_from_dataclass(RuntimeSpeciesConfig)],
        "grid": _default_from_dataclass(GridConfig),
        "time": _default_from_dataclass(TimeConfig),
        "init": _default_from_dataclass(InitializationConfig),
        "collisions": _default_from_dataclass(RuntimeCollisionConfig),
        "normalization": _default_from_dataclass(RuntimeNormalizationConfig),
        "terms": _default_from_dataclass(RuntimeTermsConfig),
        "expert": _default_from_dataclass(RuntimeExpertConfig),
    }

    return {
        "errors": errors,
        "warnings": warnings,
        "missing_required": missing_required,
        "defaults": defaults,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="GKX runtime TOML file to validate.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tools_out") / "io_validation" / "summary.json",
        help="Output summary JSON path.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    toml_mod = _load_toml_module()

    cfg_path = args.config.expanduser().resolve()
    if not cfg_path.exists():
        raise SystemExit(f"Config file not found: {cfg_path}")

    text = cfg_path.read_text(encoding="utf-8")
    data = toml_mod.loads(text)
    schema_result = _validate_toml_schema(data)

    summary = {
        "config": str(cfg_path),
        "format": "toml",
        "schema_validation": {
            "ok": len(schema_result["errors"]) == 0,
            "errors": schema_result["errors"],
            "warnings": schema_result["warnings"],
            "missing_required": schema_result["missing_required"],
        },
        "defaults": schema_result["defaults"],
    }

    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"saved {out_path}")
    print(json.dumps(summary["schema_validation"], indent=2, sort_keys=True))
    return 0 if summary["schema_validation"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())