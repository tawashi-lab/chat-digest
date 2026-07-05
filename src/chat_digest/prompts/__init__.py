"""プロンプトプロファイルのロード。

プロファイルは YAML ファイル(キー: TOPIC_DETECT_USER_PROMPT 等、値: プロンプト文字列)。
検索順: 設定の prompt_dirs(ローカル上書き)→ パッケージ同梱デフォルト。
`general.yaml`(タイトル生成・ハイライト等の共通プロンプト)は常にベースとして読み込まれ、
プロファイル本体と job 単位の上書きがその上にマージされる。
"""

from importlib import resources
from pathlib import Path
from typing import Iterable, Optional

import yaml

GENERAL_PROFILE = "general"


class PromptProfileNotFoundError(FileNotFoundError):
    pass


def _bundled_dir() -> Path:
    return Path(resources.files("chat_digest.prompts"))  # type: ignore[arg-type]


def _find_profile_file(name: str, extra_dirs: Iterable[Path]) -> Optional[Path]:
    filename = f"{name}.yaml"
    for directory in [*extra_dirs, _bundled_dir()]:
        candidate = Path(directory) / filename
        if candidate.is_file():
            return candidate
    return None


def _load_yaml(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Prompt profile must be a mapping: {path}")
    return {str(k): str(v) for k, v in data.items()}


def list_bundled_profiles() -> list[str]:
    return sorted(p.stem for p in _bundled_dir().glob("*.yaml"))


def load_prompt_profile(
    profile: str,
    extra_dirs: Iterable[str | Path] = (),
    overrides: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """general + プロファイル + 上書きをマージした prompt_dict を返す。"""
    dirs = [Path(d) for d in extra_dirs]

    merged: dict[str, str] = {}
    general_path = _find_profile_file(GENERAL_PROFILE, dirs)
    if general_path is not None:
        merged.update(_load_yaml(general_path))

    profile_path = _find_profile_file(profile, dirs)
    if profile_path is None:
        searched = ", ".join(str(d) for d in [*dirs, _bundled_dir()])
        raise PromptProfileNotFoundError(
            f"Prompt profile '{profile}' not found (searched: {searched}). "
            f"Bundled profiles: {', '.join(list_bundled_profiles())}"
        )
    merged.update(_load_yaml(profile_path))

    if overrides:
        merged.update(overrides)
    return merged
