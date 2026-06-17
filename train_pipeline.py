import polars as pl
import numpy as np
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from preprocessing import AuthorEncoder

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

train_pd = train.select(["timespent", "duration", "category", "author_id", "like"]).to_pandas()
val_pd = val.select(["timespent", "duration", "category", "author_id", "like"]).to_pandas()

X_train = train_pd[["timespent", "duration", "category", "author_id"]]
y_train = train_pd["like"]
X_val = val_pd[["timespent", "duration", "category", "author_id"]]
y_val = val_pd["like"]

pipeline = Pipeline([
    ("author_encoder", AuthorEncoder()),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

print("Обучаю пайплайн...")
pipeline.fit(X_train, y_train)

with open("pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Пайплайн сохранён в pipeline.pkl")