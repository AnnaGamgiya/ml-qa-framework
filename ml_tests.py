import polars as pl
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from allpairspy import AllPairs
from itertools import combinations, product

class SegmentedRegressionTest:
    def __init__(self, data, pipeline, segment_col="category", threshold=0.65, min_group_size=30):
        self.data = data
        self.pipeline = pipeline
        self.segment_col = segment_col
        self.threshold = threshold
        self.min_group_size = min_group_size
        self.result = None
        self.details = None
        self.segments_auc = {}
        self.group_sizes = {}

    def run(self):
        segments = self.data[self.segment_col].unique().to_list()
        failed = []
        for seg in segments:
            seg_data = self.data.filter(pl.col(self.segment_col) == seg)
            n = seg_data.height
            self.group_sizes[seg] = n
            if n < self.min_group_size:
                continue
            X = seg_data.select(["timespent", "duration", "category", "author_popularity"]).to_pandas()
            y_true = seg_data["like"].to_numpy()
            if len(np.unique(y_true)) < 2:
                continue
            y_pred = self.pipeline.predict_proba(X)[:, 1]
            auc = roc_auc_score(y_true, y_pred)
            self.segments_auc[seg] = auc
            if auc < self.threshold:
                failed.append(f"Категория {seg}: AUC={auc:.3f}")
        self.result = "PASS" if len(failed) == 0 else "FAIL"
        self.details = f"Провалено сегментов: {len(failed)}"
        return self.result, self.details, self.segments_auc, self.group_sizes, failed


class BoundaryTest:
    def __init__(self, pipeline, feature_names, train_medians, train_modes):
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.train_medians = train_medians
        self.train_modes = train_modes
        self.result = None
        self.details = None
        self.test_cases = []

    def run(self):
        self.test_cases = []
        bounds = {
            "timespent": [0, 1, 254, 255],
            "duration": [5, 6, 179, 180],
            "category": [0, 1, 18, 19],
            "author_popularity": [0, 1, self.train_medians.get("author_popularity", 500), self.train_medians.get("author_popularity", 500)*2]
        }
        feature_list = [f for f in self.feature_names if f in bounds]
        for f1, f2 in combinations(feature_list, 2):
            for v1, v2 in product(bounds[f1], bounds[f2]):
                case = {}
                for f in self.feature_names:
                    if f == f1:
                        case[f] = v1
                    elif f == f2:
                        case[f] = v2
                    elif f in self.train_modes:
                        case[f] = self.train_modes[f]
                    else:
                        case[f] = self.train_medians.get(f, 0)
                self.test_cases.append(case)

        for case in self.test_cases:
            X = pd.DataFrame([case])
            proba = self.pipeline.predict_proba(X)[:, 1]
            if np.isnan(proba).any() or (proba < 0).any() or (proba > 1).any():
                case["Предсказание"] = None
                case["Статус"] = "Ошибка"
            else:
                case["Предсказание"] = proba[0]
                case["Статус"] = "Корректно"

        errors = [c for c in self.test_cases if c["Статус"] == "Ошибка"]
        self.result = "PASS" if len(errors) == 0 else "FAIL"
        self.details = f"Проверено {len(self.test_cases)} граничных комбинаций, ошибок: {len(errors)}"
        return self.result, self.details, self.test_cases


class DecisionTableTest:
    def __init__(self, data, pipeline, min_group_size=30, min_effect_size=0.02):
        self.data = data
        self.pipeline = pipeline
        self.min_group_size = min_group_size
        self.min_effect_size = min_effect_size
        self.result = None
        self.details = None
        self.scenarios = []

    def _add(self, name, desc_a, desc_b, group_a_data, group_b_data, expected_direction):
        if group_a_data.height == 0 or group_b_data.height == 0:
            return
        if group_a_data.height < self.min_group_size or group_b_data.height < self.min_group_size:
            return

        pred_a = self.pipeline.predict_proba(
            group_a_data.select(["timespent", "duration", "category", "author_popularity"]).to_pandas()
        )[:, 1]
        pred_b = self.pipeline.predict_proba(
            group_b_data.select(["timespent", "duration", "category", "author_popularity"]).to_pandas()
        )[:, 1]

        mean_a = float(np.mean(pred_a))
        mean_b = float(np.mean(pred_b))
        diff = mean_b - mean_a

        passed = False
        inconclusive = False

        if expected_direction == 'greater':
            passed = diff > 0
        elif expected_direction == 'less':
            passed = diff < 0
        else:  # two-sided
            if abs(diff) > self.min_effect_size:
                passed = True
            else:
                inconclusive = True

        self.scenarios.append({
            "name": name,
            "condition_a": desc_a,
            "condition_b": desc_b,
            "mean_A": mean_a,
            "mean_B": mean_b,
            "diff": diff,
            "min_effect": self.min_effect_size,
            "expected": expected_direction,
            "passed": passed,
            "inconclusive": inconclusive
        })

    def run(self):
        self.scenarios = []

        top_authors = (
            self.data
            .group_by("author_id")
            .count()
            .sort("count", descending=True)
            .head(50)["author_id"]
            .to_list()
        )

        self._add(
            "Досмотр > Недосмотр",
            "timespent = 0",
            "timespent = duration",
            self.data.filter(pl.col("timespent") == 0),
            self.data.filter(pl.col("timespent") == pl.col("duration")),
            "greater"
        )

        self._add(
            "Недосмотр хуже полного досмотра",
            "timespent < duration",
            "timespent == duration",
            self.data.filter(pl.col("timespent") < pl.col("duration")),
            self.data.filter(pl.col("timespent") == pl.col("duration")),
            "greater"
        )

        self._add(
            "Короткие > Длинные",
            "duration > 120",
            "duration < 30",
            self.data.filter(pl.col("duration") > 120),
            self.data.filter(pl.col("duration") < 30),
            "greater"
        )

        self._add(
            "Популярный автор ≠ Обычный",
            "author_id в топ-50",
            "author_id не в топ-50",
            self.data.filter(pl.col("author_id").is_in(top_authors)),
            self.data.filter(~pl.col("author_id").is_in(top_authors)),
            "two-sided"
        )

        self._add(
            "Короткие <60 vs >=60",
            "duration >= 60",
            "duration < 60",
            self.data.filter(pl.col("duration") >= 60),
            self.data.filter(pl.col("duration") < 60),
            "two-sided"
        )

        failed = [s for s in self.scenarios if not s["passed"]]
        self.result = "PASS" if len(failed) == 0 else "FAIL"
        self.details = f"Пройдено сценариев: {len(self.scenarios) - len(failed)}/{len(self.scenarios)}"
        return self.result, self.details, self.scenarios


class PairwiseTest:
    def __init__(self, pipeline, feature_names, parameters, train_medians, train_modes):
        self.pipeline = pipeline
        self.feature_names = feature_names
        self.parameters = parameters
        self.train_medians = train_medians
        self.train_modes = train_modes
        self.result = None
        self.details = None
        self.num_combinations = 0
        self.coverage = 0.0
        self.combinations = []

    def run(self):
        param_values = list(self.parameters.values())
        pairs = list(AllPairs(param_values))
        self.num_combinations = len(pairs)
        self.combinations = []
        covered = set()
        total_pairs = 0
        keys = list(self.parameters.keys())
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                total_pairs += len(self.parameters[keys[i]]) * len(self.parameters[keys[j]])
        for combo in pairs:
            combo_dict = dict(zip(keys, combo))
            full_case = {}
            for f in self.feature_names:
                if f in combo_dict:
                    full_case[f] = combo_dict[f]
                elif f in self.train_modes:
                    full_case[f] = self.train_modes[f]
                else:
                    full_case[f] = self.train_medians.get(f, 0)
            self.combinations.append(full_case)
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    covered.add((keys[i], combo[i], keys[j], combo[j]))
        self.coverage = len(covered) / total_pairs if total_pairs > 0 else 0

        errors = []
        for case in self.combinations:
            X = pd.DataFrame([case])
            proba = self.pipeline.predict_proba(X)[:, 1]
            if np.isnan(proba).any() or (proba < 0).any() or (proba > 1).any():
                case["Предсказанная вероятность"] = None
                case["Статус"] = "Ошибка"
                errors.append(case)
            else:
                case["Предсказанная вероятность"] = proba[0]
                case["Статус"] = "OK"
        self.result = "PASS" if len(errors) == 0 else "FAIL"
        self.details = f"Проверено {self.num_combinations} комбинаций, ошибок: {len(errors)}"
        return self.result, self.details, self.num_combinations, self.coverage


class StabilityTest:
    def __init__(self, train, val, model_columns, context_columns, threshold=0.25):
        self.train = train
        self.val = val
        self.model_columns = model_columns
        self.context_columns = context_columns
        self.threshold = threshold
        self.result = None
        self.details = None
        self.psi_values = {}
        self.new_authors_frac = None

    def _psi(self, expected, actual, bins=10):
        breakpoints = np.histogram_bin_edges(np.concatenate([expected, actual]), bins=bins)
        expected_hist, _ = np.histogram(expected, bins=breakpoints)
        actual_hist, _ = np.histogram(actual, bins=breakpoints)
        expected_ratio = expected_hist / expected_hist.sum()
        actual_ratio = actual_hist / actual_hist.sum()
        mask = (expected_ratio > 0) & (actual_ratio > 0)
        psi_values = (actual_ratio[mask] - expected_ratio[mask]) * np.log(actual_ratio[mask] / expected_ratio[mask])
        return np.sum(psi_values)

    def run(self):
        failed = []
        for col in self.model_columns:
            if col not in self.train.columns or col not in self.val.columns:
                continue
            train_col = self.train[col].to_numpy()
            val_col = self.val[col].to_numpy()
            psi_val = self._psi(train_col, val_col)
            self.psi_values[col] = psi_val
            if psi_val > self.threshold:
                failed.append(f"{col}: PSI={psi_val:.4f}")

        for col in self.context_columns:
            if col not in self.train.columns or col not in self.val.columns:
                continue
            train_col = self.train[col].to_numpy()
            val_col = self.val[col].to_numpy()
            psi_val = self._psi(train_col, val_col)
            self.psi_values[col] = psi_val

        if "author_id" in self.train.columns and "author_id" in self.val.columns:
            train_authors = set(self.train["author_id"].unique().to_list())
            val_authors = self.val["author_id"].unique().to_list()
            new_authors = [a for a in val_authors if a not in train_authors]
            self.new_authors_frac = len(new_authors) / len(val_authors) if val_authors else 0

        self.result = "PASS" if len(failed) == 0 else "FAIL"
        self.details = f"Признаков с дрейфом: {len(failed)}"
        return self.result, self.details, self.psi_values, self.new_authors_frac, failed