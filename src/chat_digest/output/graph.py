"""トピック別メッセージ数の時系列グラフと、盛り上がりピークへのリンク生成。"""

import copy
import warnings

# matplotlib に日本語フォントを登録する。matplotlib-fontja は japanize-matplotlib の
# 後継(Python 3.12 対応)。旧環境向けに japanize-matplotlib にもフォールバックする。
try:
    import matplotlib_fontja  # noqa: F401
except ImportError:
    try:
        import japanize_matplotlib  # noqa: F401
    except ImportError:
        warnings.warn(
            "No Japanese font package found (matplotlib-fontja); "
            "non-ASCII graph labels may not render correctly."
        )

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

from chat_digest.sources.base import UrlContext


def create_pivot_table(id_df: pd.DataFrame,
                       index='hour',
                       columns='topic',
                       values='timestamp',
                       aggfunc='count',
                       fill_value=0) -> pd.DataFrame:
    id_df['timestamp'] = pd.to_datetime(id_df['timestamp'])
    id_df['hour'] = id_df['timestamp'].dt.floor('h')

    pivot_df = pd.pivot_table(
        id_df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        fill_value=fill_value
    )
    return pivot_df


def get_peaks_graph(pivot_df: pd.DataFrame,
                    png_path="peakgraph.png") -> str:
    """トピックごとの時系列を塗りつぶし折れ線で描画して PNG に保存する。"""
    # 凡例を件数の多い順に並べる
    col_sums = pivot_df.sum().sort_values(ascending=False)
    pivot_df = pivot_df[col_sums.index]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor('white')
    ax.set_facecolor('white')

    for col in pivot_df.columns:
        ax.plot(pivot_df.index, pivot_df[col], marker='o', label=col)
        ax.fill_between(pivot_df.index, 0, pivot_df[col], alpha=0.3)

    ax.set_ylabel('')
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    plt.xticks(fontsize=30)

    # fill_between 由来の重複凡例を除去
    handles, labels = ax.get_legend_handles_labels()
    new_handles, new_labels = [], []
    for handle, label in zip(handles, labels):
        if label != '_nolegend_' and label not in new_labels:
            new_handles.append(handle)
            new_labels.append(label)

    # 凡例ラベルは8文字ごとに改行(最大8行)
    wrapped_labels = []
    for label in new_labels:
        lines = []
        for i in range(0, len(label), 8):
            lines.append(label[i:i + 8])
            if len(lines) == 8:
                break
        wrapped_labels.append("\n".join(lines))

    ax.legend(
        new_handles,
        wrapped_labels,
        loc='center right',
        bbox_to_anchor=(0, 0.5),
        fontsize=25,
        facecolor='lightgray'
    )

    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    return str(png_path)


def get_peaks_urls(chat_df: pd.DataFrame,
                   pivot_df: pd.DataFrame,
                   topic_list: list,
                   url_ctx: UrlContext) -> list:
    """トピックごとに盛り上がりピーク時刻の先頭メッセージへの URL リストを返す。"""
    peak_info = {}

    topic_idx_dict = {}
    for idx, topic in enumerate(topic_list):
        topic_idx_dict[topic[0]] = idx

    for col in pivot_df.columns:
        y = pivot_df[col].values

        peaks, properties = find_peaks(
            y,
            height=int(chat_df.shape[0] / 200),
            distance=3,
            prominence=int(chat_df.shape[0] / 100),
            width=True,
            rel_height=0.5,
        )

        result_df = pd.DataFrame({
            "peak_idx": peaks,
            "peak_time": pivot_df.index[peaks],
            "peak_value": y[peaks],
            "left_base_idx": properties["left_bases"],
            "right_base_idx": properties["right_bases"],
            "width": properties["widths"],
            "left_ips": properties["left_ips"],
            "right_ips": properties["right_ips"],
            "prominence": properties["prominences"],
        })
        result_df.index = range(len(result_df))
        peak_info[col] = result_df

    topic_peak_df = copy.deepcopy(chat_df)

    peaks_message_ids = []
    for topic_idx, topic in enumerate(topic_list):
        topic = topic[0]
        topic_id = topic_idx_dict[topic] + 1
        topic_peaks = peak_info[topic]
        peak_message_ids = []
        for topic_peak in topic_peaks.itertuples():
            peak_message = get_first_message_peak_hour(topic_peak_df, topic_id, topic_peak.peak_time)
            if peak_message is not None:
                peak_message_ids.append({"timestamp": peak_message.timestamp, "id": peak_message.id})
        peaks_message_ids.append(peak_message_ids)

    urls_list = []
    for peak_message_ids in peaks_message_ids:
        urls = []
        for message_id in peak_message_ids:
            urls.append({
                "timestamp": message_id["timestamp"],
                "url": url_ctx.message_url(message_id["id"]),
            })
        urls_list.append(urls)

    return urls_list


def get_first_message_peak_hour(chat_df, topic_id, peak_time):
    """ピーク時刻の1時間枠内で最初のメッセージ行を返す。"""
    previous_time = peak_time - pd.Timedelta(hours=1)
    mask = (
        (chat_df['topic_id'] == topic_id)
        & (chat_df['timestamp'] >= previous_time)
        & (chat_df['timestamp'] <= previous_time + pd.Timedelta(hours=1))
    )
    filtered_df = chat_df[mask]
    if not filtered_df.empty:
        return filtered_df.iloc[0]
    return None
