import streamlit as st
import polars as pl
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_auc_score
from preprocessing import AuthorEncoder
from ml_tests import (SegmentedRegressionTest, BoundaryTest,
                      DecisionTableTest, PairwiseTest, StabilityTest)

st.set_page_config(page_title="ML QA Framework", layout="wide")
st.title("ML QA Framework — тестирование ML методами QA")
st.markdown("Датасет: **VK-LSVD** | Модель: **LogisticRegression** | Разработчик: Анна Гамгия")

@st.cache_data
def load_data():
    train = pl.read_parquet("data/subsamples/up0.001_ip0.001/train/week_24.parquet")
    val = pl.read_parquet("data/subsamples/up0.001_ip0.001/validation/week_25.parquet")
    items_meta = pl.read_parquet("data/metadata/items_metadata.parquet")

    train = train.join(items_meta.select(["item_id", "author_id", "duration"]), on="item_id", how="left")
    val = val.join(items_meta.select(["item_id", "author_id", "duration"]), on="item_id", how="left")

    train = train.filter(pl.col("duration").is_not_null())
    val = val.filter(pl.col("duration").is_not_null())

    np.random.seed(42)
    all_items = pl.concat([train.select("item_id"), val.select("item_id")]).unique().to_series().to_list()
    category_map = {item_id: i % 20 for i, item_id in enumerate(all_items)}
    train = train.with_columns(pl.col("item_id").replace(category_map).cast(pl.UInt8).alias("category"))
    val = val.with_columns(pl.col("item_id").replace(category_map).cast(pl.UInt8).alias("category"))

    author_counts = train.group_by("author_id").len().rename({"len": "author_popularity"})
    train = train.join(author_counts, on="author_id", how="left")
    val = val.join(author_counts, on="author_id", how="left")

    train = train.with_columns(pl.col("author_popularity").fill_null(0))
    val = val.with_columns(pl.col("author_popularity").fill_null(0))

    return train, val

train, val = load_data()

with open("pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

feature_cols = ["timespent", "duration", "category", "author_popularity"]
X_val = val.select(feature_cols).to_pandas()
pred_val = pipeline.predict_proba(X_val)[:, 1]
val = val.with_columns(pl.Series("pred", pred_val))

train_pd = train.select(feature_cols).to_pandas()
train_medians = train_pd.median().to_dict()
train_modes = {col: train_pd[col].mode()[0] for col in train_pd.columns if train_pd[col].dtype == 'object' or train_pd[col].nunique() < 20}

srt = SegmentedRegressionTest(val, pipeline, threshold=0.65, min_group_size=30)
srt_res, srt_det, srt_auc, srt_sizes, srt_fail = srt.run()

bt = BoundaryTest(pipeline, ["timespent", "duration", "category", "author_popularity"], train_medians, train_modes)
bt_res, bt_det, bt_cases = bt.run()

dtt = DecisionTableTest(val, pipeline, min_group_size=30, min_effect_size=0.02)
dtt_res, dtt_det, dtt_scenarios = dtt.run()

timespent_bins = [0, 1, 10, 50, 100, 150, 200, 255]
duration_bins = [5, 10, 15, 30, 60, 90, 120, 150, 180]
params = {
    "timespent": timespent_bins,
    "duration": duration_bins,
    "category": list(range(20)),
    "author_popularity": [0, 1, 10, 100, 500, 1000]
}
pt = PairwiseTest(pipeline, ["timespent", "duration", "category", "author_popularity"], params, train_medians, train_modes)
pt_res, pt_det, pt_comb, pt_cov = pt.run()

st_test = StabilityTest(train, val,
                        model_columns=["timespent", "duration", "category", "author_popularity"],
                        context_columns=["platform", "place", "agent"],
                        threshold=0.25)
st_res, st_det, st_psi, st_new_auth, st_fail = st_test.run()

page = st.sidebar.radio("Навигация", [
    "Сводка", "Segmented AUC", "Boundary Test",
    "Decision Table", "Pairwise Test", "Stability Test", "Справочник"
])

if page == "Сводка":
    st.header("Сводка результатов тестирования")
    cols = st.columns(3)
    tests = [
        ("Segmented AUC", srt_res, srt_det),
        ("Boundary", bt_res, bt_det),
        ("Decision Table", dtt_res, dtt_det),
        ("Pairwise", pt_res, pt_det),
        ("Stability", st_res, st_det)
    ]
    for idx, (name, res, det) in enumerate(tests):
        with cols[idx % 3]:
            if res == "PASS": st.success(f"✅ {name}\n{det}")
            else: st.error(f"❌ {name}\n{det}")

elif page == "Segmented AUC":
    st.header("Segmented AUC — качество по категориям контента")
    st.markdown("""
    **Что показывает график:**  
    Модель предсказывает вероятность лайка. Мы разделили все видео на 20 групп.  
    Для каждой группы посчитали AUC. Чем выше столбец, тем лучше модель предсказывает лайк.  
    Красная линия — порог 0.65. Столбец ниже линии означает, что в этой группе модель работает хуже.

    Цифры 0–19 — это номера групп. В реальном проекте здесь были бы названия: «спорт», «юмор», «мода».  
    Рядом с каждым столбцом указано количество записей. Если их меньше 30 или отсутствуют представители одного из классов, AUC не считается, и группа помечается как «недостаточно данных».
    """)
    if srt_res == "PASS": st.success(srt_det)
    else: st.error(srt_det)
    if srt_auc:
        cat_names = [f"{k} ({srt_sizes.get(k, '?')} зап.)" for k in srt_auc]
        fig = px.bar(x=cat_names, y=list(srt_auc.values()), labels={"x": "Категория", "y": "AUC"})
        fig.add_hline(y=0.65, line_dash="dash", line_color="red")
        st.plotly_chart(fig)
    all_segments = list(range(20))
    missing = [s for s in all_segments if s not in srt_auc]
    if missing:
        st.warning(f"Категории с недостаточно данных: {missing}")

elif page == "Boundary Test":
    st.header("Boundary Test — граничные значения")
    st.markdown("""
    **Что показывает таблица:**  
    Подаём крайние значения четырёх признаков: `timespent`, `duration`, `category`, `author_popularity`.  
    Для каждого признака взято по 4 граничных значения. Для неварьируемых признаков подставляются медианные/модальные значения.  
    Тест проверяет, что на экстремальных входных данных production-пайплайн не падает, а предсказания остаются в [0, 1] и не содержат NaN.
    """)
    if bt_res == "PASS": st.success(bt_det)
    else: st.error(bt_det)
    st.markdown(f"Проверено **{len(bt_cases)}** комбинаций.")
    df_bt = pd.DataFrame(bt_cases[:30])
    st.dataframe(df_bt[["timespent", "duration", "category", "author_popularity", "Предсказание", "Статус"]])

elif page == "Decision Table":
    st.header("Decision Table — проверка бизнес-гипотез")
    st.markdown("""
    **Что показывает таблица:**  
    Мы сформулировали ожидаемые правила и проверили их через сравнение средних предсказанных вероятностей.  
    Для каждой гипотезы указаны две группы (А и Б), их средние вероятности и разница (B−A).

    **Как читать вердикты:**
    - ✅ **Подтверждена** — разница направлена в ожидаемую сторону. Для гипотез с направлением (greater / less) этого достаточно, порог не применяется. Для гипотез без направления (two-sided) разница должна быть больше 0.02.
    - ❌ **Противоречит** — разница направлена в противоположную сторону (ожидали рост, а получили падение, или наоборот).
    - ⚠️ **Недостаточно данных** — разница есть, но она слишком мала (меньше 0.02), чтобы считать гипотезу подтверждённой. Применяется только к гипотезам без направления (two-sided).

    **Почему для направленных гипотез нет порога:**  
    Если модель говорит «досмотренное видео лайкают чаще», но разница всего 1% — это всё ещё верное направление. Модель не ошиблась, просто эффект слабый. Мы не наказываем модель за слабый эффект, если она правильно поняла закономерность. Порог 0.02 оставлен только для случаев, где мы проверяем сам факт различия без указания направления.

    Статистические тесты не применяются — фреймворк сознательно смещает фокус на практическую значимость (Effect Size).
    """)
    human = {
        "Недосмотр хуже полного досмотра": "Недосмотренное видео получает более низкую вероятность лайка, чем досмотренное?",
        "Короткие > Длинные": "Короткие (<30 с) лайкают чаще длинных (>120 с)?",
        "Популярный автор ≠ Обычный": "Популярный автор vs обычный — есть ли разница?",
        "Короткие <60 vs >=60": "Видео короче 60 с vs длиннее 60 с"
    }
    if dtt_res == "PASS": st.success(dtt_det)
    else: st.warning(dtt_det)
    table = []
    for s in dtt_scenarios:
        if s.get("inconclusive"):
            status = "⚠️ недостаточно данных"
        elif s["passed"]:
            status = "✅ подтверждена"
        else:
            status = "❌ противоречит"
        min_effect_display = f"{s['min_effect']:.2f}" if s["expected"] == "two-sided" else "не применяется"
        table.append({
            "Гипотеза": human.get(s["name"], s["name"]),
            "Группа А": s["condition_a"],
            "Группа Б": s["condition_b"],
            "Средняя А": f"{s['mean_A']:.4f}",
            "Средняя Б": f"{s['mean_B']:.4f}",
            "Разница (B−A)": f"{s['diff']:+.4f}",
            "Мин. значимый эффект": min_effect_display,
            "Ожидаемое направление": s["expected"],
            "Вердикт": status
        })
    st.dataframe(table)

elif page == "Pairwise Test":
    st.header("Pairwise Test — попарное тестирование параметров")
    st.markdown("""
    **Что показывает таблица:**  
    Признаки: `timespent` (разбит на диапазоны, квантован), `duration` (разбит на диапазоны, квантован),  
    `category` (дискретный, 0–19), `author_popularity` (дискретный, 0, 1, 10, 100, 500, 1000).  
    Алгоритм AllPairs сгенерировал минимальный набор, покрывающий все возможные пары. Для нашего набора данных получилось 182 комбинации.
    """)
    if pt_res == "PASS": st.success(pt_det)
    else: st.error(pt_det)
    st.markdown(f"Сгенерировано **{pt_comb}** комбинаций, покрытие пар **{pt_cov:.2%}**.")
    df_pt = pd.DataFrame(pt.combinations[:100])
    cols = ["timespent", "duration", "category", "author_popularity", "Предсказанная вероятность"]
    if "Статус" in df_pt.columns:
        cols.append("Статус")
    st.dataframe(df_pt[cols])

elif page == "Stability Test":
    st.header("Stability Test — контроль дрейфа (PSI)")
    st.markdown("""
    **Что показывает таблица:**  
    PSI для признаков модели (`timespent`, `duration`, `category`, `author_popularity`) и контекстных признаков (`platform`, `place`, `agent`).  
    Для `author_id` классический PSI не вычисляется из-за высокой кардинальности. Вместо этого контролируется доля новых, ранее не встречавшихся авторов.
    """)
    if st_res == "PASS": st.success(st_det)
    else: st.error(st_det)
    model_psi = {k: v for k, v in st_psi.items() if k in ["timespent", "duration", "category", "author_popularity"]}
    context_psi = {k: v for k, v in st_psi.items() if k in ["platform", "place", "agent"]}
    st.subheader("Признаки модели")
    st.dataframe(pd.DataFrame({"Признак": list(model_psi.keys()), "PSI": list(model_psi.values())}))
    st.subheader("Контекстные признаки")
    st.dataframe(pd.DataFrame({"Признак": list(context_psi.keys()), "PSI": list(context_psi.values())}))
    if st_new_auth is not None:
        st.metric("Доля новых авторов", f"{st_new_auth:.2%}")
    fig = px.bar(x=list(st_psi.keys()), y=list(st_psi.values()), labels={"x": "Признак", "y": "PSI"})
    fig.add_hline(y=0.25, line_dash="dash", line_color="red")
    st.plotly_chart(fig)

elif page == "Справочник":
    st.header("Справочник терминов и допущений")
    with st.expander("AUC"):
        st.markdown("Метрика качества. 1.0 — идеал, 0.5 — гадание, <0.5 — модель вредная.")
    with st.expander("Эквивалентное разбиение"):
        st.markdown("Классическая техника тест-дизайна: делим входные данные на группы и проверяем, что в каждой группе система ведёт себя одинаково. В нашем случае группы — категории контента.")
    with st.expander("Граничные значения"):
        st.markdown("Проверка модели на самых крайних допустимых значениях. Цель — убедиться, что production-пайплайн не падает и не выдаёт некорректных результатов.")
    with st.expander("Pairwise"):
        st.markdown("Техника попарного тестирования. Алгоритм генерирует минимальный набор комбинаций, покрывающий все возможные пары значений. Непрерывные признаки предварительно разбиваются на диапазоны.")
    with st.expander("PSI"):
        st.markdown("Индекс стабильности популяции. Сравнивает распределения признака в двух выборках. > 0.25 — значительный дрейф.")
    with st.expander("Минимальный значимый эффект"):
        st.markdown("Порог практической значимости. Разница в предсказанной вероятности должна быть больше этого порога, чтобы считаться важной. Установлен 0.02 для демонстрации. Применяется только для гипотез без направления (two-sided).")
    with st.expander("Направление гипотезы"):
        st.markdown("Указывает, какое соотношение между группами мы ожидаем: greater (Б > А), less (Б < А), two-sided (А ≠ Б).")
    with st.expander("Допущения"):
        st.markdown("""
        - Категории 0–19 — суррогат настоящих тематик.
        - Пороги (AUC=0.65, размер эффекта=0.02, мин. группа=30) выбраны для демонстрации.
        - Pairwise покрывает пары, но не тройки и более высокие взаимодействия.
        - Модель — простейшая логистическая регрессия.
        """)