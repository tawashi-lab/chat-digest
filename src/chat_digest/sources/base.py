"""メッセージ取得層の共通インターフェース。

各 source は fetch() で FetchResult を返す。effective は以下の列を持つこと:
  id, timestamp, author, total_reaction_count, integrated_content
(integrated_content が LLM に渡される要約対象テキスト)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from chat_digest.config import SourceConfig


@dataclass
class FetchResult:
    raw: pd.DataFrame        # 取得したままのメッセージ(ログ保存用)
    effective: pd.DataFrame  # 前処理済み・要約対象


@dataclass(frozen=True)
class UrlContext:
    """メッセージへのリンク URL の構成要素。空文字の部分は URL から省略される。"""

    url_base: str
    server_part: str
    channel_part: str

    def message_url(self, message_id) -> str:
        parts = [self.url_base, self.server_part, self.channel_part, str(message_id)]
        return "/".join(p for p in parts if p != "")


class MessageSource(ABC):
    def __init__(self, name: str, config: SourceConfig):
        self.name = name
        self.config = config

    def resolve_channel(self, channel: str):
        try:
            return self.config.channels[channel]
        except KeyError:
            raise KeyError(f"source '{self.name}' has no channel '{channel}'") from None

    @abstractmethod
    def fetch(
        self,
        channel: str,
        *,
        hours: int,
        start_delta_hours: int = 0,
        limit: int = 7000,
    ) -> FetchResult:
        """現在時刻から start_delta_hours 遡った時点を終端として hours 分の履歴を取得する。"""

    @abstractmethod
    def url_context(self, channel: str) -> UrlContext:
        """このチャンネルのメッセージ URL を組み立てるための情報を返す。"""
