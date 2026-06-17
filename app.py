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
st.title("🧪 ML QA Framework — тестирование ML методами QA")
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
    
    return train, val

train, val = load_data()

with open("pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

X_val = val.select(["timespent", "duration", "category", "author_id"]).to_pandas()
pred_val = pipeline.predict_proba(X_val)[:, 1]
val = val.with_columns(pl.Series("pred", pred_val))

train_pd = train.select(["timespent", "duration", "category", "author_id"]).to_pandas()
train_medians = train_pd.median().to_dict()
train_modes = train_pd.mode().iloc[0].to_dict()

top_authors = train.group_by("author_id").count().sort("count", descending=True).head(50)["author_id"].to_list()

srt = SegmentedRegressionTest(val, pipeline, threshold=0.65, min_group_size=100)
srt_res, srt_det, srt_auc, srt_sizes, srt_fail = srt.run()

bt = BoundaryTest(pipeline, ["timespent", "duration", "category", "author_id"], train_medians, train_modes)
bt_res, bt_det, bt_cases = bt.run()

dtt = DecisionTableTest(val, pipeline, min_group_size=100, min_effect_size=0.02)
dtt.top_authors = top_authors
dtt_res, dtt_det, dtt_scenarios = dtt.run()

params = {
    "timespent": [0, 1, 10, 50, 100, 150, 200, 255],
    "duration": [5, 10, 15, 30, 60, 90, 120, 150, 180],
    "category": list(range(20)),
    "author_id": train_pd["author_id"].quantile([0.1, 0.5, 0.9]).tolist()
}
pt = PairwiseTest(pipeline, ["timespent", "duration", "category", "author_id"], params, train_medians, train_modes)
pt_res, pt_det, pt_comb, pt_cov = pt.run()

st_test = StabilityTest(train, val,
                        model_columns=["timespent", "duration", "category", "author_id"],
                        context_columns=["platform", "place", "agent"],
                        threshold=0.25)
st_res, st_det, st_psi, st_fail = st_test.run()

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
    Для каждой группы посчитали AUC. Чем выше столбец, тем лучше модель предсказывает лайк в этой группе.  
    Красная линия — порог 0.65. Столбец ниже линии означает, что в этой группе модель работает хуже.

    **Откуда взялись цифры:**  
    Цифры 0–19 — это номера групп. В реальном проекте здесь были бы названия: «спорт», «юмор», «мода».  
    В этом датасете таких названий нет, поэтому мы используем номера.

    **Размер группы:**  
    Рядом с каждым столбцом указано количество записей. Если их меньше 100, AUC не считается.
    """)
    if srt_res == "PASS": st.success(srt_det)
    else: st.error(srt_det)
    if srt_auc:
        cat_names = [f"{k} ({srt_sizes.get(k, '?')} зап.)" for k in srt_auc]
        fig = px.bar(x=cat_names, y=list(srt_auc.values()), labels={"x": "Категория", "y": "AUC"})
        fig.add_hline(y=0.65, line_dash="dash", line_color="red")
        st.plotly_chart(fig)

elif page == "Boundary Test":
    st.header("Boundary Test — граничные значения")
    st.markdown("""
    **Что показывает таблица:**  
    Мы подаём модели крайние значения признаков: минимальное время просмотра, максимальную длительность и т.д.  
    Каждая строка — одна комбинация. В колонках — значения признаков, которые мы подали.  
    В колонке «Предсказание» — вероятность лайка. Если она некорректна (NaN, <0 или >1), статус «Ошибка».
    """)
    if bt_res == "PASS": st.success(bt_det)
    else: st.error(bt_det)
    st.markdown(f"Проверено **{len(bt_cases)}** комбинаций.")
    df_bt = pd.DataFrame(bt_cases[:30])
    st.dataframe(df_bt[["timespent", "duration", "category", "author_id", "Предсказание", "Статус"]])

elif page == "Decision Table":
    st.header("Decision Table — проверка бизнес-гипотез")
    st.markdown("""
    **Что показывает таблица:**  
    Мы сформулировали ожидаемые правила и проверили их статистически.  
    Колонка «p‑value» показывает, случайна ли разница (если < 0.05 — не случайна).  
    Колонка «Разница (B−A)» — насколько велика разница. Колонка «Мин. значимый эффект» — порог, ниже которого разница считается несущественной.  
    Вердикт выносится с учётом направления, статистической значимости и практической важности.
    """)
    human = {
        "Досмотр > Недосмотр": "Досмотренное видео лайкают чаще, чем недосмотренное?",
        "Перемотка хуже досмотра": "Перемотанное видео получает более низкую вероятность лайка, чем досмотренное?",
        "Короткие > Длинные": "Короткие (<30 с) лайкают чаще длинных (>120 с)?",
        "Любое время > Нулевое": "Любое время просмотра > нулевого?",
        "Популярный автор ≠ Обычный": "Популярный автор vs обычный — есть ли разница?",
        "Короткие <60 vs >=60": "Видео короче 60 с vs длиннее 60 с"
    }
    if dtt_res == "PASS": st.success(dtt_det)
    else: st.warning(dtt_det)
    table = []
    for s in dtt_scenarios:
        table.append({
            "Гипотеза": human.get(s["name"], s["name"]),
            "Группа А": s["condition_a"],
            "Группа Б": s["condition_b"],
            "Средняя А": f"{s['mean_A']:.4f}",
            "Средняя Б": f"{s['mean_B']:.4f}",
            "Разница (B−A)": f"{s['diff']:+.4f}",
            "p‑value": f"{s['p_value']:.4f}",
            "Мин. значимый эффект": f"{s['min_effect']:.2f}",
            "Ожидаемое направление": s["expected"],
            "Вердикт": s["status"]
        })
    st.dataframe(table)

elif page == "Pairwise Test":
    st.header("Pairwise Test — попарное тестирование параметров")
    st.markdown("""
    **Что показывает таблица:**  
    Мы проверили модель на комбинациях параметров. Каждая строка — одна комбинация: время, длительность, категория, автор.  
    В колонке «Предсказанная вероятность» — результат модели. Если бы модель выдала некорректное значение (NaN, <0, >1), тест бы упал и показал эту комбинацию.
    """)
    if pt_res == "PASS": st.success(pt_det)
    else: st.error(pt_det)
    st.markdown(f"Сгенерировано **{pt_comb}** комбинаций, покрытие пар **{pt_cov:.2%}**.")
    df_pt = pd.DataFrame(pt.combinations[:100])
    cols = ["timespent", "duration", "category", "author_id", "Предсказанная вероятность"]
    if "Статус" in df_pt.columns:
        cols.append("Статус")
    st.dataframe(df_pt[cols])

elif page == "Stability Test":
    st.header("Stability Test — дрейф данных (PSI)")
    st.markdown("""
    **Что показывает таблица:**  
    Мы сравнили распределение признаков в обучающей и проверочной выборках.  
    **Признаки модели** — те, что подаются на вход модели. Их дрейф влияет на предсказания.  
    **Контекстные признаки** — дополнительные, не влияют на модель, но сигнализируют об изменении среды.
    """)
    if st_res == "PASS": st.success(st_det)
    else: st.error(st_det)
    model_psi = {k: v for k, v in st_psi.items() if k in ["timespent", "duration", "category", "author_id"]}
    context_psi = {k: v for k, v in st_psi.items() if k in ["platform", "place", "agent"]}
    st.subheader("Признаки модели")
    st.dataframe(pd.DataFrame({"Признак": list(model_psi.keys()), "PSI": list(model_psi.values())}))
    st.subheader("Контекстные признаки")
    st.dataframe(pd.DataFrame({"Признак": list(context_psi.keys()), "PSI": list(context_psi.values())}))
    fig = px.bar(x=list(st_psi.keys()), y=list(st_psi.values()), labels={"x": "Признак", "y": "PSI"})
    fig.add_hline(y=0.25, line_dash="dash", line_color="red")
    st.plotly_chart(fig)

elif page == "Справочник":
    st.header("Справочник терминов и допущений")
    
    with st.expander("AUC (Area Under the ROC Curve)"):
        st.markdown("Метрика качества бинарной классификации. Показывает, насколько хорошо модель разделяет положительный класс (лайк) и отрицательный (не лайк). Значение 0.5 — случайное угадывание, 1.0 — идеальное разделение. В дашборде используется для оценки равномерности качества по категориям контента.")
    with st.expander("Эквивалентное разбиение (Equivalence Partitioning)"):
        st.markdown("Классическая техника тест-дизайна, при которой входные данные делятся на группы (классы эквивалентности), внутри которых поведение системы должно быть одинаковым. В дашборде применено в тесте Segmented AUC: категории видео выступают классами, и мы проверяем, что качество модели в каждой категории не падает ниже порога.")
    with st.expander("Граничные значения (Boundary Value Analysis)"):
        st.markdown("Техника тестирования, фокусирующаяся на минимальных, максимальных и близких к ним значениях входных параметров — именно там чаще всего скрываются ошибки. В дашборде реализована в тесте Boundary Test: для каждого признака выбираются два значения «у границы» и два «сразу за границей» или внутри, и их попарные комбинации подаются модели.")
    with st.expander("Попарное тестирование (Pairwise Testing)"):
        st.markdown("Техника, позволяющая сократить число тестовых комбинаций, гарантируя при этом покрытие всех возможных пар значений между любыми двумя параметрами. В дашборде используется AllPairs алгоритм, который сгенерировал 182 комбинации, покрывающих 100% пар всех четырёх признаков. Помогает проверить, что модель не ломается ни при каком сочетании параметров, без полного перебора.")
    with st.expander("Таблица принятия решений (Decision Table)"):
        st.markdown("Инструмент систематизации логики «если условия, то ожидаемый результат». В контексте ML мы формулируем бизнес-гипотезы (например, «досмотренное видео получает более высокую вероятность лайка»), разбиваем данные на две группы по условиям и статистически проверяем, соответствует ли модель ожиданию.")
    with st.expander("PSI (Population Stability Index)"):
        st.markdown("Метрика для обнаружения дрейфа данных. Сравнивает распределение признака в тренировочной и тестовой (валидационной) выборках. Значение PSI > 0.25 обычно сигнализирует о значительном расхождении. В дашборде используется в тесте Stability Test для контроля стабильности входных данных.")
    with st.expander("p‑value (достигаемый уровень значимости)"):
        st.markdown("Вероятность получить наблюдаемое различие (или более сильное) при условии, что на самом деле различий нет. Если p‑value меньше заранее выбранного порога (обычно 0.05), различие считается статистически значимым — оно вряд ли вызвано случайностью.")
    with st.expander("Минимальный значимый эффект (Minimum Effect Size)"):
        st.markdown("Порог практической значимости, который в реальном проекте устанавливается бизнесом или владельцем продукта. Даже если разница статистически значима, она может быть настолько малой, что не имеет ценности для продукта (например, отличие в вероятности лайка на 0.1%). В дашборде используется значение 0.02 (2 п.п.) как демонстрационное.")
    with st.expander("Направление гипотезы (Direction)"):
        st.markdown("""
        Указывает, какое именно соотношение между группами мы ожидаем. Это ожидание также формируется бизнесом или экспертами предметной области:
        - greater — среднее группы Б должно быть больше среднего группы А;
        - less — среднее Б должно быть меньше среднего А;
        - two‑sided — нас интересует сам факт различия без указания конкретного направления («есть ли разница?»).
        """)
    with st.expander("Статистическая значимость vs практическая значимость"):
        st.markdown("Два уровня оценки гипотез. Статистическая значимость (p‑value) говорит, можно ли доверять найденному различию. Практическая значимость (минимальный эффект) говорит, стоит ли на это различие обращать внимание в реальной жизни. В дашборде они комбинируются: гипотеза считается подтверждённой, только если и p‑value < 0.05, и разница превышает минимальный эффект в ожидаемом направлении. Оба критерия задаются дата-сайентистом и бизнесом, а не QA.")
    with st.expander("Логистическая регрессия (Logistic Regression)"):
        st.markdown("Простая ML-модель, предсказывающая вероятность принадлежности к классу (в нашем случае — вероятность лайка). Выбрана для учебного дашборда из‑за прозрачности и скорости.")
    with st.expander("Препроцессинг (Preprocessing)"):
        st.markdown("""
        Шаги, преобразующие «сырые» данные в формат, пригодный для модели. В production-среде этот пайплайн создаётся дата-сайентистом и передаётся QA в виде готового артефакта (например, сериализованный sklearn Pipeline). В нашем проекте препроцессинг включает:
        - кодирование идентификатора автора (author_id) в признак author_enc (частотное кодирование);
        - стандартизацию (StandardScaler) числовых признаков (timespent, duration, author_enc).
        """)
    
    st.markdown("---")
    st.subheader("Допущения в учебном дашборде: что даёт бизнес и Data Scientist")
    st.markdown("""
    Все настройки, пороги и подготовка данных в реальном проекте поступают к QA от двух источников: бизнес (продуктолог, аналитик) и дата-сайентист. QA не обязан разбираться, как именно эти цифры были получены, – он проверяет, что модель им соответствует. Ниже перечислено, какие именно артефакты мы имитируем в учебном дашборде, и кто за них отвечал бы в реальности.

    **1. Артефакты, предоставляемые дата-сайентистом**

    - **Сериализованный препроцессинг-пайплайн (model_pipeline.pkl):** Чтобы Boundary и Pairwise тесты подавали данные в модель точно так же, как в production. QA не нужно знать шаги преобразования, достаточно использовать готовый объект. Пайплайн обучен на тренировочной выборке и сохранён. Дашборд загружает его и применяет ко всем тестовым комбинациям.
    - **Список признаков, подаваемых в модель:** Чтобы Stability Test (PSI) мониторил только те признаки, которые влияют на предсказание. Дополнительные колонки могут присутствовать в данных, но модель их не использует. Мониторятся только timespent, duration, category, author_enc. Признаки platform, place, agent считаются справочно и не вызывают алертов.
    - **Минимальный размер группы для Segmented AUC:** Категории со слишком малым числом записей не позволяют надёжно оценить AUC. DS сообщает порог, ниже которого AUC не вычисляется. Установлено: минимум 30 записей. Если меньше – показывается предупреждение «недостаточно данных».
    - **Порог тревоги для PSI:** Общепринятое значение PSI > 0.25 сигнализирует о значительном дрейфе. При необходимости DS может скорректировать порог под особенности данных. Используется стандартный порог 0.25.
    - **Стратегия заполнения неварьируемых признаков в Boundary/Pairwise тестах:** Когда мы варьируем два признака из четырёх, остальные нужно зафиксировать в «нейтральном» состоянии (обычно медиана или мода). DS определяет, какие значения использовать как типичные. Неварьируемые признаки заполняются медианными/модальными значениями тренировочной выборки.

    **2. Артефакты, предоставляемые бизнесом (совместно с DS)**

    - **Бизнес-гипотезы и их ожидаемое направление:** Какие закономерности модель должна отражать по мнению экспертов продукта? Например, «полный досмотр повышает вероятность лайка», «короткие видео лайкают чаще». Направление (greater/less/two‑sided) формулируется до тестирования. Сформулированы 4 гипотезы, направление указано в таблице Decision Table. Для демонстрации выбраны разумные, но условные утверждения.
    - **Минимальный практически значимый размер эффекта:** Бизнес определяет, какая разница в предсказанной вероятности считается достаточно важной, чтобы обращать на неё внимание. Меньшие различия могут быть статистически значимы, но не иметь продуктового смысла. Установлено значение 0.02 (2 п.п.). Выбрано для наглядной демонстрации, в реальности цифра будет другой.
    - **Порог AUC для Segmented AUC:** Какой минимальный уровень качества модель должна показывать в любой категории контента? Определяется бизнес-требованиями к продукту. Установлен порог 0.65 – ниже модель считается неравномерной.
    - **Порог статистической значимости (p‑value):** Обычно принимается стандартное значение 0.05, но в отдельных случаях бизнес может ужесточить или ослабить требование. Используется стандартный порог 0.05.

    **3. Ключевое правило для QA**

    QA-инженер не создаёт перечисленные выше параметры и артефакты. Он получает их как входную спецификацию и проверяет, что:
    - Модель предсказывает корректно на границах и в сгенерированных комбинациях (Boundary, Pairwise) при использовании предоставленного пайплайна.
    - Segmented AUC и PSI соответствуют заданным порогам для утверждённого списка признаков и категорий.
    - Бизнес-гипотезы либо подтверждаются (✅), либо отклоняются (❌), либо признаются практически несущественными (⚠️) строго по правилам, зафиксированным в спецификации.

    Все настройки в учебном дашборде имитируют такую спецификацию. В реальном проекте вместо демонстрационных значений будут стоять цифры, полученные от бизнеса и дата-сайентиста.
    """)