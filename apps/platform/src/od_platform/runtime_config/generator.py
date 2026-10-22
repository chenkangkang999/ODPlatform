"""配置文件生成器: 从 Pydantic 配置类反射生成 YAML 模板."""
from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Type, Union

from pydantic import BaseModel

from od_platform.common.paths import runtime_config_path
from od_platform.runtime_config.registry import CONFIG_REGISTRY

logger = logging.getLogger(__name__)


class ConfigGenerator:
    """从 Pydantic 配置类反射生成 YAML 模板."""

    def __init__(self, indent: int = 2):
        self.indent = indent

    def generate(
        self,
        config_class: Type[BaseModel],
        output_path: Union[str, Path],
        *,
        overwrite: bool = False,
        backup:    bool = True,
        title:     Optional[str] = None,
    ) -> bool:
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            logger.info(f"配置文件已存在, 跳过生成: {output_path}")
            return False
        if output_path.exists() and backup:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = output_path.with_name(
                f"{output_path.name}.bak.{stamp}"
            )
            shutil.copy2(output_path, backup_path)
            logger.warning(f"覆盖前已备份原配置: {backup_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = self._generate_yaml(config_class, title)
        output_path.write_text(content, encoding="utf-8")
        logger.info(f"配置文件已生成: {output_path}")
        return True

    def _generate_yaml(
        self,
        config_class: Type[BaseModel],
        title: Optional[str] = None,
    ) -> str:
        lines: List[str] = []
        lines.append("#" + "=" * 78)
        lines.append(f"# {title or config_class.__name__}")
        lines.append(f"# 自动生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("#" + "=" * 78)
        lines.append("")
        config = config_class()
        groups = config.get_field_groups()
        for group_name, field_names in groups.items():
            lines.append("")
            lines.append("#" + "-" * 78)
            lines.append(f"# {group_name}")
            lines.append("#" + "-" * 78)
            lines.append("")
            for field_name in field_names:
                lines.extend(self._generate_field(config, field_name))
                lines.append("")
        lines.append("")
        lines.append("#" + "=" * 78)
        lines.append("# 常见问题")
        lines.append("#" + "=" * 78)
        lines.append("#")
        lines.append("# Q: 如何修改配置?")
        lines.append("# A: 直接编辑对应参数的值, 保存后重新运行即可")
        lines.append("#")
        lines.append("# Q: 命令行参数会覆盖配置文件吗?")
        lines.append("# A: 是的, 命令行参数优先级最高: CLI > YAML > DEFAULT")
        lines.append("#")
        lines.append("# Q: 如何恢复默认配置?")
        lines.append("# A: 删除此文件, 程序会自动报错并提示重新生成")
        lines.append("#")
        lines.append("# Q: 如何用新版默认值重新生成此模板?")
        lines.append("# A: 跑 'odp-gen-config <name> --overwrite' (会自动备份原文件)")
        lines.append("#")
        lines.append("#" + "=" * 78)
        return "\n".join(lines)

    def _generate_field(
        self,
        config: BaseModel,
        field_name: str,
    ) -> List[str]:
        lines: List[str] = []
        metadata = config.get_field_metadata(field_name)
        yaml_comment = metadata.get("yaml_comment") or metadata.get("description")
        if yaml_comment:
            lines.append(f"# {yaml_comment}")
        examples = metadata.get("examples", [])
        if examples:
            examples_str = ", ".join(str(e) for e in examples[:5])
            lines.append(f"# 示例: {examples_str}")
        tips = metadata.get("tips", [])
        if tips:
            lines.append("# 提示:")
            for tip in tips:
                lines.append(f"#   - {tip}")
        value = getattr(config, field_name)
        yaml_value = self._format_value(value)
        lines.append(f"{field_name}: {yaml_value}")
        return lines

    def _format_value(self, value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            if any(c in value for c in [":", "#", "[", "]", "{", "}"]):
                return f'"{value}"'
            return value
        if isinstance(value, (list, tuple)):
            if not value:
                return "[]"
            items = ", ".join(self._format_value(v) for v in value)
            return f"[{items}]"
        if isinstance(value, dict):
            return "{}"
        return str(value)


def main():
    parser = argparse.ArgumentParser(
        prog="odp-gen-config",
        description="生成 YOLO 运行配置 YAML 模板",
    )
    parser.add_argument(
        "name",
        choices=list(CONFIG_REGISTRY),
        help="要生成的配置名 (train / val / infer)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出路径 (默认: configs/runtime/<name>.yaml)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有文件 (默认不覆盖)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="覆盖时不备份原文件 (默认会备份成 <name>.yaml.bak.<时间戳>)",
    )
    args = parser.parse_args()
    config_class, title = CONFIG_REGISTRY[args.name]
    output_path = args.output or runtime_config_path(args.name)
    gen = ConfigGenerator()
    success = gen.generate(
        config_class,
        output_path,
        overwrite=args.overwrite,
        backup=not args.no_backup,
        title=title,
    )
    if success:
        print(f"✓ 已生成: {output_path}")
    else:
        print(
            f"- 文件已存在, 未覆盖 (避免覆盖你已编辑的配置).\n"
            f"  路径: {output_path}\n"
            f"  如需重新生成, 加 --overwrite (覆盖前会自动备份)"
        )


if __name__ == "__main__":
    main()
