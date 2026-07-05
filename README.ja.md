# chat-digest

[English README](README.md)

**chat-digest** は、流速の速い Discord / Telegram のチャットチャンネルを LLM でトピック別に要約し、Discord Webhook に投稿するツールです。cron からの定期実行を想定しています。

<!-- TODO: まとめ投稿のスクリーンショットを追加 -->

## 仕組み

```
履歴取得 ──► 前処理 ──► トピック抽出 ──► 全メッセージを分類
 (Discord Bot /   (連投の      (複数候補を        (並列 LLM 呼び出し、
  selfbot /        グルーピング)  生成して投票)      プロンプトキャッシュ)
  Telegram)
                                                    │
Discord へ投稿 ◄── 埋め込み組み立て ◄── 加工 ◄── トピック別に要約
 (Webhook、          (フィールド上限に   (短縮・タイトル・50字ハイライト・
  ヘッドライン版、     収まるよう分割)     Grok 検索によるファクトチェック・
  盛り上がりグラフ)                       embedding による参照リンク付与)
```

- **取得層は差し替え可能**: 公式 Bot API(推奨)/ セルフボット(⚠️ Discord 利用規約違反・自己責任)/ Telegram(Telethon)
- **ジョブ定義は YAML**: チャンネル×時間窓ごとにプリセット化(`chat-digest run --job general-24h`)
- **プロンプトプロファイル**: 仮想通貨 / 株式 / 雑談 / AI コミュニティ向けを同梱。ローカル YAML やジョブ単位で上書き可能
- **マルチプロバイダ LLM**: [LiteLLM](https://github.com/BerriAI/litellm) 経由で OpenAI / Anthropic / Gemini。ステップごとにモデルを選べ、大量呼び出しになる分類ステップはプロンプトキャッシュ対応

## インストール

Python 3.10+。

```bash
pip install 'chat-digest[discord] @ git+https://github.com/tawashi-lab/chat-digest'
# extras: discord(公式Bot)/ selfbot / telegram / grok
```

> `discord` と `selfbot` はどちらも `discord` モジュールを提供するため、同時にはインストールできません。

## セットアップ

1. **Discord Webhook を作成**: まとめを投稿したいチャンネルの設定 → 連携サービス → Webhook。
2. **取得元を準備**:
   - *Discord Bot*: [Developer Portal](https://discord.com/developers/applications) で Bot を作成し、**Message Content Intent** を有効化。メッセージ履歴の読み取り権限付きでサーバーに招待。
   - *Telegram*: [my.telegram.org](https://my.telegram.org) で `api_id` / `api_hash` を取得。初回実行時に電話番号認証があります。
3. **設定ファイルを作成**:

```bash
cp config/config.example.yaml config/config.yaml   # サーバー・チャンネルIDとジョブを編集
cp .env.example .env                               # APIキー・トークン・Webhook URL を記入
```

4. **検証して実行**:

```bash
chat-digest validate
chat-digest run --job general-24h
```

### cron 設定例

```cron
0 4 * * * cd /path/to/chat-digest && .venv/bin/chat-digest run --job general-24h >> cron.log 2>&1
```

## 設定リファレンス

コメント付きの例は [config/config.example.yaml](config/config.example.yaml) を参照してください。

| セクション | 役割 |
|---|---|
| `sources` | メッセージの取得元(種別・トークンの環境変数名・チャンネルID) |
| `outputs` | 投稿先 Discord Webhook(環境変数名で参照) |
| `jobs` | 要約実行のプリセット: 対象チャンネル・時間窓・トピック設定・機能・プロンプトプロファイル・投稿先 |
| `llm.models` | パイプラインのステップごとの使用モデル(`gpt-5.x-hh/hm/hl/mh` の reasoning エイリアス対応) |
| `branding` | Webhook の表示名・アバター・フッター |

ジョブごとの機能フラグ: `condense`(要約の短縮)、`highlight`(50字ヘッドライン)、`graph`(トピック推移グラフ)、`references`(embedding で要約中のフレーズに元チャットへのリンクを付与)、`topic_context` / `fact_check`(Grok のライブ検索。`GROK_API_KEY` が必要)。

### プロンプトのカスタマイズ

同梱プロファイル: `crypto` / `stocks` / `casual` / `ai` / `crypto_airdrop`。カスタマイズする場合は、config の `prompt_dirs:` に列挙したディレクトリに `<プロファイル名>.yaml` を置くと同梱版より優先されます。ジョブ単位の `prompt_overrides:` で個別キーの上書きも可能です。

## 開発

```bash
pip install -e '.[discord,telegram,grok,dev]'
pytest
ruff check src tests
```

## 免責事項

- セルフボット取得はユーザーアカウントの自動化であり、**Discord の利用規約に違反**します。アカウント停止のリスクを理解した上で自己責任で使用してください。Bot を招待できる環境では公式 Bot API を推奨します。
- 要約は LLM 生成のため誤りを含む可能性があります。ファクトチェック機能は誤りを減らしますがゼロにはできません。

## ライセンス

[MIT](LICENSE)
