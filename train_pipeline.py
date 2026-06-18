import polars as pl
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

print("Загружаю данные...")
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

author_counts = train.group_by("author_id").count().rename({"count": "author_popularity"})
train = train.join(author_counts, on="author_id", how="left")
val = val.join(author_counts, on="author_id", how="left")

# Заполняем пропуски нулями – автор не встречался в обучении
train = train.with_columns(pl.col("author_popularity").fill_null(0))
val = val.with_columns(pl.col("author_popularity").fill_null(0))

feature_cols = ["timespent", "duration", "category", "author_popularity"]
X_train = train.select(feature_cols).to_pandas()
y_train = train.select("like").to_pandas().to_numpy().ravel()
X_val = val.select(feature_cols).to_pandas()
y_val = val.select("like").to_pandas().to_numpy().ravel()

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

print("Обучаю пайплайн...")
pipeline.fit(X_train, y_train)

with open("pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Пайплайн сохранён в pipeline.pkl")