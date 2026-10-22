"""配置合并器: 按优先级合并多个配置源 + 配置溯源追踪."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError
from pydantic_core import InitErrorDetails, PydanticCustomError

from od_platform.common.string_utils import pad_to_width


class ConfigSource(str, Enum):
    """配置来源 (内置 3 档). 自定义源直接用字符串, 不必扩 enum."""
    DEFAULT = "DEFAULT"
    YAML    = "YAML"
    CLI     = "CLI"


@dataclass
class ConfigMetadata:
    """单个字段的溯源信息. 链表节点."""
    key:             str
    value:           Any
    source:          Union[ConfigSource, str]
    timestamp:       datetime
    overridden_from: Optional["ConfigMetadata"] = None

    @property
    def source_label(self) -> str:
        if isinstance(self.source, ConfigSource):
            return self.source.value
        return self.source

    def chain(self) -> List["ConfigMetadata"]:
        result = [self]
        current = self.overridden_from
        while current:
            result.append(current)
            current = current.overridden_from
        return result

    def chain_str(self) -> str:
        parts = [f"{m.value}({m.source_label})" for m in self.chain()]
        return " ← ".join(parts)


T = TypeVar("T", bound=BaseModel)

REPORT_WIDTH: int = 70


def _center(text: str, width: int = REPORT_WIDTH) -> str:
    return pad_to_width(text, width, align="center")


class ConfigMerger:
    """sources list 接口的三源 (及更多) 合并 + 链表式溯源."""

    def __init__(self, track_sources: bool = True):
        self.track_sources = track_sources
        self._metadata: Dict[str, ConfigMetadata] = {}
        self._overridden_keys: List[str] = []
        self._last_config_class: Optional[Type[BaseModel]] = None

    def merge(
        self,
        config_class: Type[T],
        *,
        sources: Optional[List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]]] = None,
    ) -> T:
        merged = self._do_merge(config_class, sources)
        try:
            return config_class(**merged)
        except ValidationError as e:
            if self.track_sources:
                raise self._enhance_validation_error(e) from e
            raise

    def preview(
        self,
        config_class: Type[BaseModel],
        *,
        sources: Optional[List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]]] = None,
    ) -> Dict[str, Any]:
        return self._do_merge(config_class, sources)

    def get_metadata(self, key: str) -> Optional[ConfigMetadata]:
        return self._metadata.get(key)

    def get_source_report(self) -> str:
        if not self.track_sources:
            return "配置溯源未启用"
        lines = ["=" * REPORT_WIDTH, _center("配置来源报告"), "=" * REPORT_WIDTH]
        by_source: Dict[str, List[str]] = {}
        for key, meta in self._metadata.items():
            by_source.setdefault(meta.source_label, []).append(key)
        builtin_order = [s.value for s in
                         [ConfigSource.CLI, ConfigSource.YAML, ConfigSource.DEFAULT]]
        ordered = [s for s in builtin_order if s in by_source]
        custom  = sorted(s for s in by_source if s not in builtin_order)
        all_labels = ordered + custom
        for label in all_labels:
            keys = sorted(by_source[label])
            lines.append(f"\n{label} ({len(keys)} 项)")
            lines.append("-" * 70)
            if not keys:
                lines.append("  (无)")
            else:
                for key in keys:
                    value = self._display_value(key, self._metadata[key].value)
                    lines.append(f"  {key} = {value}")
        return "\n".join(lines)

    def get_conflict_report(self) -> str:
        if not self.track_sources:
            return "配置溯源未启用"
        overridden = list(dict.fromkeys(self._overridden_keys))
        lines = ["=" * REPORT_WIDTH, _center("配置覆盖报告"), "=" * REPORT_WIDTH]
        lines.append(f"\n共 {len(overridden)} 项配置被覆盖\n")
        if not overridden:
            lines.append("  (无)")
            return "\n".join(lines)
        for key in sorted(overridden):
            meta = self._metadata.get(key)
            if not meta:
                continue
            chain = meta.chain()
            if len(chain) <= 1:
                continue
            hops = " → ".join(
                f"{self._display_value(key, node.value)} ({node.source_label})"
                for node in reversed(chain)
            )
            lines.append(f"  {key}: {hops}")
        return "\n".join(lines)

    def to_audit_log(self) -> Dict[str, Any]:
        if not self.track_sources:
            return {
                "merger_completed_at": datetime.now().isoformat(timespec="seconds"),
                "track_sources":       False,
            }
        by_source: Dict[str, List[str]] = {}
        for key, meta in self._metadata.items():
            by_source.setdefault(meta.source_label, []).append(key)
        by_source = {k: sorted(v) for k, v in by_source.items()}
        overridden = sorted(set(self._overridden_keys))
        return {
            "merger_completed_at": datetime.now().isoformat(timespec="seconds"),
            "track_sources":       True,
            "fields_count_total":  len(self._metadata),
            "fields_by_source":    by_source,
            "overridden_count":    len(overridden),
            "overridden_fields":   overridden,
        }

    def _do_merge(
        self,
        config_class: Type[BaseModel],
        sources: Optional[List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]]],
    ) -> Dict[str, Any]:
        self._metadata.clear()
        self._overridden_keys.clear()
        self._last_config_class = config_class
        defaults = self._extract_defaults(config_class)
        all_sources: List[Tuple[Union[ConfigSource, str], Mapping[str, Any]]] = [
            (ConfigSource.DEFAULT, defaults),
        ]
        all_sources.extend(sources or [])
        merged: Dict[str, Any] = {}
        for source, cfg in all_sources:
            self._apply_source(merged, dict(cfg or {}), source)
        return merged

    def _apply_source(
        self,
        merged: Dict[str, Any],
        config: Mapping[str, Any],
        source: Union[ConfigSource, str],
    ) -> None:
        for key, value in config.items():
            if value is None:
                continue
            if key in merged and self.track_sources:
                self._overridden_keys.append(key)
            merged[key] = value
            if self.track_sources:
                prev_meta = self._metadata.get(key)
                self._metadata[key] = ConfigMetadata(
                    key=key, value=value, source=source,
                    timestamp=datetime.now(), overridden_from=prev_meta,
                )

    @staticmethod
    def _extract_defaults(config_class: Type[BaseModel]) -> Dict[str, Any]:
        defaults = {}
        for name, field in config_class.model_fields.items():
            if field.default is not None:
                defaults[name] = field.default
        return defaults

    def _enhance_validation_error(self, error: ValidationError) -> ValidationError:
        line_errors: List[InitErrorDetails] = []
        for err in error.errors(include_url=False):
            loc = err.get("loc", ())
            key = loc[0] if loc else None
            meta = self._metadata.get(key) if isinstance(key, str) else None
            if meta is None:
                line_errors.append(InitErrorDetails(
                    type=err["type"], loc=err.get("loc", ()),
                    input=err.get("input"), ctx=err.get("ctx"),
                ))
                continue
            msg = f"{err.get('msg', '')} [来源: {meta.chain_str()}]"
            line_errors.append(InitErrorDetails(
                type=PydanticCustomError("value_error", msg),
                loc=err.get("loc", ()),
                input=err.get("input"),
            ))
        title = self._last_config_class.__name__ if self._last_config_class else "Config"
        return ValidationError.from_exception_data(title, line_errors)

    def _is_sensitive(self, key: str) -> bool:
        if self._last_config_class is None:
            return False
        if not hasattr(self._last_config_class, "sensitive_field_names"):
            return False
        return key in self._last_config_class.sensitive_field_names()

    def _display_value(self, key: str, value: Any) -> Any:
        if value is None:
            return value
        if self._is_sensitive(key):
            return getattr(self._last_config_class, "SENSITIVE_MASK", "***")
        return value
