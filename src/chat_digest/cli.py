"""chat-digest CLI。

例:
    chat-digest run --job tantore-24h
    chat-digest run --job tantore-24h --post-to test   # 投稿先だけ差し替えて試す
    chat-digest jobs
    chat-digest validate
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_CONFIG_CANDIDATES = [
    Path("config/config.yaml"),
    Path("config.yaml"),
]


def _resolve_config_path(arg_value: str | None) -> Path:
    if arg_value:
        path = Path(arg_value)
        if not path.is_file():
            raise SystemExit(f"config file not found: {path}")
        return path
    for candidate in DEFAULT_CONFIG_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "config file not found. Use --config or create config/config.yaml "
        "(see config/config.example.yaml)."
    )


def _load_env(config_path: Path) -> None:
    # config と同じディレクトリ → カレントディレクトリの順に .env を読む
    load_dotenv(config_path.parent / ".env")
    load_dotenv(config_path.parent.parent / ".env")
    load_dotenv(".env")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chat-digest",
        description="チャットチャンネルを LLM でトピック別に要約し、Discord Webhook へ投稿する",
    )
    parser.add_argument("--config", "-c", help="設定 YAML のパス(既定: config/config.yaml)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="ジョブを実行する")
    run_parser.add_argument("--job", "-j", required=True, help="config.yaml の jobs のキー")
    run_parser.add_argument(
        "--post-to",
        help="投稿先を差し替える(カンマ区切りの outputs キー。テスト用)",
    )
    run_parser.add_argument(
        "--simple-post-to",
        help="ヘッドライン版の投稿先を差し替える(カンマ区切り。空文字で無効化)",
    )

    subparsers.add_parser("jobs", help="定義済みジョブの一覧を表示する")
    subparsers.add_parser("validate", help="設定とプロンプトプロファイルを検証する")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    config_path = _resolve_config_path(args.config)
    _load_env(config_path)

    from chat_digest.config import load_config
    from chat_digest.prompts import load_prompt_profile

    config = load_config(config_path)

    if args.command == "jobs":
        for name, job in config.jobs.items():
            print(f"{name}: [{job.source}/{job.channel}] {job.title} ({job.hours}h)")
        return 0

    if args.command == "validate":
        for name, job in config.jobs.items():
            prompts = load_prompt_profile(
                job.prompt_profile, extra_dirs=config.prompt_dirs, overrides=job.prompt_overrides
            )
            missing = [
                key
                for key in (
                    "TOPIC_DETECT_SYS_PROMPT",
                    "TOPIC_DETECT_USER_PROMPT",
                    "TOPIC_VOTE_SYSTEM_PROMPT",
                    "TOPIC_VOTE_USER_PROMPT",
                    "TOPIC_JUDGE_SYS_FORMAT",
                    "TOPIC_JUDGE_USER_FORMAT",
                    "TOPIC_SUMMARY_CHAT_USER_PROMPT",
                )
                if key not in prompts
            ]
            if missing:
                print(f"NG {name}: プロンプトキー不足 {missing}")
                return 1
            print(f"OK {name} (profile={job.prompt_profile})")
        print("config OK")
        return 0

    if args.command == "run":
        from chat_digest.pipeline import PipelineError, run_job

        post_to_override = None
        if args.post_to:
            post_to_override = [s.strip() for s in args.post_to.split(",") if s.strip()]
        simple_post_to_override = None
        if args.simple_post_to is not None:
            simple_post_to_override = [
                s.strip() for s in args.simple_post_to.split(",") if s.strip()
            ]

        try:
            run_job(
                config,
                args.job,
                post_to_override=post_to_override,
                simple_post_to_override=simple_post_to_override,
            )
        except PipelineError as exc:
            print(f"pipeline error: {exc}", file=sys.stderr)
            return 1
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
