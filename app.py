import os
import itertools
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import requests


# =========================
# 競輪場コード
# =========================
VENUES = {
    "函館": "11",
    "青森": "12",
    "いわき平": "13",
    "弥彦": "21",
    "前橋": "22",
    "取手": "23",
    "宇都宮": "24",
    "大宮": "25",
    "西武園": "26",
    "京王閣": "27",
    "立川": "28",
    "松戸": "31",
    "川崎": "34",
    "平塚": "35",
    "小田原": "36",
    "伊東": "37",
    "静岡": "38",
    "名古屋": "42",
    "岐阜": "43",
    "大垣": "44",
    "豊橋": "45",
    "富山": "46",
    "松阪": "47",
    "四日市": "48",
    "福井": "51",
    "奈良": "53",
    "向日町": "54",
    "和歌山": "55",
    "岸和田": "56",
    "玉野": "61",
    "防府": "63",
    "高松": "71",
    "小松島": "73",
    "高知": "74",
    "松山": "75",
    "小倉": "81",
    "久留米": "83",
    "武雄": "84",
    "佐世保": "85",
    "別府": "86",
    "熊本": "87",
}

CODE_TO_VENUE = {v: k for k, v in VENUES.items()}


# =========================
# AIモデル
# =========================
# GitHub Releaseからモデルを自動取得
import zipfile

MODEL_FILES = [
    "gen4_features.pkl",
    "gen4_model_1st.pkl",
    "gen4_model_2nd.pkl",
    "gen4_model_3rd.pkl",
]

if not all(os.path.exists(f) for f in MODEL_FILES):

    zip_url = (
        "https://github.com/"
        "ybf2gyc8yw-tech/keirin-ai/"
        "releases/download/gen4-models/"
        "gen4_models.zip"
    )

    r = requests.get(
        zip_url,
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    r.raise_for_status()

    with open("gen4_models.zip", "wb") as f:
        f.write(r.content)

    with zipfile.ZipFile("gen4_models.zip", "r") as z:
        z.extractall(".")

features_g4 = joblib.load("gen4_features.pkl")
model_1st_g4 = joblib.load("gen4_model_1st.pkl")
model_2nd_g4 = joblib.load("gen4_model_2nd.pkl")
model_3rd_g4 = joblib.load("gen4_model_3rd.pkl")


feature_fill = {
    "banum": 4,
    "age": 35,
    "term": 100,
    "race_score": 80,
    "mark_num": 0,
    "win_rate_4m": 0,
    "top2_rate_4m": 0,
    "top3_rate_4m": 0,
    "nige_4m": 0,
    "maku_4m": 0,
    "score_rank": 4,
    "win_rank": 4,
    "top2_rank": 4,
    "top3_rank": 4,
    "score_diff": 0,
    "win_diff": 0,
    "score_from_top": 0,
}


# =========================
# 第4世代特徴量
# =========================
def add_gen4_features(df):
    x = df.copy()

    base = [
        "race_score",
        "win_rate_4m",
        "top2_rate_4m",
        "top3_rate_4m",
    ]

    for col in base:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    g = x.groupby("race_id")

    x["score_rank"] = g["race_score"].rank(
        ascending=False, method="min"
    )
    x["win_rank"] = g["win_rate_4m"].rank(
        ascending=False, method="min"
    )
    x["top2_rank"] = g["top2_rate_4m"].rank(
        ascending=False, method="min"
    )
    x["top3_rank"] = g["top3_rate_4m"].rank(
        ascending=False, method="min"
    )

    x["score_diff"] = (
        x["race_score"]
        - g["race_score"].transform("mean")
    )

    x["win_diff"] = (
        x["win_rate_4m"]
        - g["win_rate_4m"].transform("mean")
    )

    x["score_from_top"] = (
        x["race_score"]
        - g["race_score"].transform("max")
    )

    return x


# =========================
# 月データ取得
# =========================
def load_month(year, month):
    url = (
        "https://raw.githubusercontent.com/"
        "Kenseimk/keirin-data/main/keirin_data/"
        f"{year}_{month:02d}_keirin.csv"
    )

    df = pd.read_csv(url)
    df["date"] = pd.to_datetime(df["date"])
    df["race_id"] = df["race_id"].astype(str)

    return df


# =========================
# 日付データ
# =========================
def prepare_day(date_value):
    dt = pd.to_datetime(date_value)

    df = load_month(dt.year, dt.month)

    day = df[
        df["date"] == dt.normalize()
    ].copy()

    if day.empty:
        return day

    day["jo_code"] = (
        day["race_id"]
        .astype(str)
        .str[:2]
    )

    day = add_gen4_features(day)

    for col in features_g4:
        if col not in day.columns:
            day[col] = feature_fill.get(col, 0)

        day[col] = pd.to_numeric(
            day[col],
            errors="coerce"
        )

        day[col] = day[col].fillna(
            feature_fill.get(col, 0)
        )

    return day


# =========================
# AI 6点
# =========================
def make_top6(race):
    race = race.copy()

    race["P1"] = model_1st_g4.predict_proba(
        race[features_g4]
    )[:, 1]

    race["P2"] = model_2nd_g4.predict_proba(
        race[features_g4]
    )[:, 1]

    race["P3"] = model_3rd_g4.predict_proba(
        race[features_g4]
    )[:, 1]

    cars = race["banum"].astype(int).tolist()

    p = race.set_index("banum")[
        ["P1", "P2", "P3"]
    ]

    combos = []

    for a, b, c in itertools.permutations(cars, 3):

        score = (
            p.loc[a, "P1"]
            * p.loc[b, "P2"]
            * p.loc[c, "P3"]
        )

        combos.append({
            "買い目": f"{a}-{b}-{c}",
            "AIスコア": round(float(score), 4),
        })

    return (
        pd.DataFrame(combos)
        .sort_values(
            "AIスコア",
            ascending=False
        )
        .head(6)
        .reset_index(drop=True)
    )


# =========================
# OddsParkオッズ
# =========================
def get_odds_table(
    date_value,
    jo_code,
    race_no,
    first
):
    url = "https://www.oddspark.com/keirin/Odds.do"

    params = {
        "joCode": str(jo_code).zfill(2),
        "kaisaiBi": str(date_value).replace("-", ""),
        "raceNo": int(race_no),
        "betType": 9,
        "jikuCode": 1,
        "shaban": int(first),
    }

    html = requests.get(
        url,
        params=params,
        timeout=10,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    ).text

    tables = pd.read_html(
        StringIO(html)
    )

    return tables[1]


def get_one_odds(
    table,
    second,
    third
):
    try:
        second_row = table.iloc[1, 2:]

        second_cols = {}

        for i, val in enumerate(second_row, start=2):
            try:
                second_cols[int(float(val))] = i
            except:
                pass

        third_rows = {}

        for i in range(2, len(table)):
            try:
                val = table.iloc[i, 1]
                third_rows[int(float(val))] = i
            except:
                pass

        col = second_cols.get(int(second))
        row = third_rows.get(int(third))

        if col is None or row is None:
            return None

        value = table.iloc[row, col]

        return float(value)

    except:
        return None


def add_odds(
    date_value,
    jo_code,
    race_no,
    result
):
    result = result.copy()

    odds_list = []

    tables = {}

    for bet in result["買い目"]:

        a, b, c = map(
            int,
            bet.split("-")
        )

        if a not in tables:
            try:
                tables[a] = get_odds_table(
                    date_value,
                    jo_code,
                    race_no,
                    a
                )
            except:
                tables[a] = None

        if tables[a] is None:
            odds = None
        else:
            odds = get_one_odds(
                tables[a],
                b,
                c
            )

        odds_list.append(odds)

    result["現在オッズ"] = odds_list

    return result


# =========================
# 開催場
# =========================
def get_venues(date_value):
    try:
        day = prepare_day(date_value)

        if day.empty:
            return gr.update(
                choices=[],
                value=None
            )

        codes = sorted(
            day["jo_code"]
            .dropna()
            .astype(str)
            .unique()
        )

        venues = [
            CODE_TO_VENUE.get(
                code,
                f"{code}場"
            )
            for code in codes
        ]

        return gr.update(
            choices=venues,
            value=venues[0]
            if venues else None
        )

    except:
        return gr.update(
            choices=[],
            value=None
        )


# =========================
# レース一覧
# =========================
def get_races(
    date_value,
    venue
):
    try:
        if not venue:
            return gr.update(
                choices=[],
                value=None
            )

        day = prepare_day(date_value)

        jo_code = VENUES.get(venue)

        if jo_code is None:
            return gr.update(
                choices=[],
                value=None
            )

        v = day[
            day["jo_code"] == jo_code
        ]

        races = sorted(
            pd.to_numeric(
                v["race_no"],
                errors="coerce"
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        return gr.update(
            choices=races,
            value=races[0]
            if races else None
        )

    except:
        return gr.update(
            choices=[],
            value=None
        )


# =========================
# 予想
# =========================
def predict_keirin(
    date_value,
    venue,
    race_no
):
    try:
        if not venue or race_no is None:
            return pd.DataFrame({
                "買い目": [
                    "レースを選択してください"
                ]
            })

        day = prepare_day(date_value)

        jo_code = VENUES[venue]

        race = day[
            (day["jo_code"] == jo_code)
            &
            (
                pd.to_numeric(
                    day["race_no"],
                    errors="coerce"
                )
                == int(race_no)
            )
        ].copy()

        if race.empty:
            return pd.DataFrame({
                "買い目": [
                    "レースデータなし"
                ]
            })

        top6 = make_top6(race)

        result = add_odds(
            date_value,
            jo_code,
            race_no,
            top6
        )

        return result

    except Exception as e:
        return pd.DataFrame({
            "エラー": [str(e)]
        })


# =========================
# 初期画面
# =========================
today = datetime.now(
    ZoneInfo("Asia/Tokyo")
).strftime("%Y-%m-%d")

try:
    initial_day = prepare_day(today)

    initial_codes = sorted(
        initial_day["jo_code"]
        .dropna()
        .astype(str)
        .unique()
    )

    initial_venues = [
        CODE_TO_VENUE.get(
            c,
            f"{c}場"
        )
        for c in initial_codes
    ]

except:
    initial_venues = []


initial_venue = (
    initial_venues[0]
    if initial_venues
    else None
)

initial_races = []

if initial_venue:

    code = VENUES.get(
        initial_venue
    )

    if code:
        initial_races = sorted(
            pd.to_numeric(
                initial_day[
                    initial_day["jo_code"] == code
                ]["race_no"],
                errors="coerce"
            )
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )


# =========================
# Gradio
# =========================
with gr.Blocks(
    title="第4世代 競輪AI"
) as app:

    gr.Markdown(
        """
# 🔥 第4世代 競輪AI
### AI上位6点 × 現在オッズ
"""
    )

    date_box = gr.Textbox(
        label="📅 日付",
        value=today
    )

    venue_box = gr.Dropdown(
        label="🏟️ 競輪場",
        choices=initial_venues,
        value=initial_venue
    )

    race_box = gr.Dropdown(
        label="🏁 レース",
        choices=initial_races,
        value=(
            initial_races[0]
            if initial_races
            else None
        )
    )

    predict_button = gr.Button(
        "🔥 AI予想する",
        variant="primary"
    )

    output = gr.Dataframe(
        interactive=False
    )

    gr.Markdown(
        """
※ AIスコアは的中確率ではなく、買い目を順位付けするための指標です。  
※ オッズは取得時点の値で随時変動します。  
※ 過去の成績は将来の利益を保証するものではありません。
"""
    )

    date_box.change(
        get_venues,
        date_box,
        venue_box
    ).then(
        get_races,
        [
            date_box,
            venue_box
        ],
        race_box
    )

    venue_box.change(
        get_races,
        [
            date_box,
            venue_box
        ],
        race_box
    )

    predict_button.click(
        predict_keirin,
        [
            date_box,
            venue_box,
            race_box
        ],
        output
    )


# =========================
# Render起動
# =========================
port = int(
    os.environ.get(
        "PORT",
        7860
    )
)

app.launch(
    server_name="0.0.0.0",
    server_port=port,
    share=False
)
