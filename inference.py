import argparse
import json
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction import FeatureHasher

# -----------------------------
# Пути артефактов
# -----------------------------
ART_DIR = Path("artifacts_boundary")
CORPUS_DIR = Path("prepared_corpus")

MODEL_PATH = ART_DIR / "boundary_sgd.joblib"
BEST_THR_PATH = ART_DIR / "best_threshold.txt"
CONFIG_PATH = ART_DIR / "config_boundary.json"
UNIGRAM_PATH = CORPUS_DIR / "unigram_freq.tsv"

DEFAULT_OUTPUT = Path("submission.csv")

# -----------------------------
# Загрузка конфига/артефактов
# -----------------------------


def load_config() -> dict:
    cfg = {
        "WINDOW": 5,
        "LEXICON_TOPK": None,  # если None — берём все частоты
        "SEED": 42,
    }
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict):
                cfg.update(user_cfg)
        except Exception as e:
            print(f"[warn] не удалось прочитать {CONFIG_PATH}: {e}")
    else:
        print(f"[info] нет {CONFIG_PATH}, использую дефолт: {cfg}")
    return cfg


def load_threshold() -> float:
    if BEST_THR_PATH.exists():
        try:
            txt = BEST_THR_PATH.read_text(encoding="utf-8").strip()
            return float(txt)
        except Exception as e:
            print(f"[warn] не удалось прочитать {BEST_THR_PATH}: {e}")
    return 0.5


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Не найден файл модели: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_lexicon(topk: Optional[int]) -> Dict[str, int]:
    """
    Читает prepared_corpus/unigram_freq.tsv (формат: token \t freq).
    Если topk задан — берём первые topk.
    Возвращает dict: {token: count}.
    """
    lexicon: Dict[str, int] = {}
    if not UNIGRAM_PATH.exists():
        print(f"[info] нет {UNIGRAM_PATH} — продолжаю без лексикона")
        return lexicon

    with UNIGRAM_PATH.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            w, c = parts
            try:
                cnt = int(c)
            except ValueError:
                continue
            lexicon[w] = cnt
            if topk is not None and i >= topk:
                break
    print(f"[info] загружен лексикон: {len(lexicon)} токенов")
    return lexicon

# -----------------------------
# Фичи — как в notebook (минимальная версия)
# -----------------------------


def is_cyr(ch: str) -> bool:
    return 'А' <= ch <= 'я' or ch in "Ёё"


def is_lat(ch: str) -> bool:
    oc = ord(ch)
    return (65 <= oc <= 90) or (97 <= oc <= 122)


def is_digit(ch: str) -> bool:
    return ch.isdigit()


def is_punct(ch: str) -> bool:
    return ch in r""".,;:!?-—()[]{}«»"'/\|+*=_~%^<>…"""


def cat(ch: str) -> str:
    if is_cyr(ch):
        return "cyr"
    if is_lat(ch):
        return "lat"
    if is_digit(ch):
        return "dig"
    if is_punct(ch):
        return "pnc"
    return "oth"


def trans(a: str, b: str) -> str:
    return f"{a}->{b}"


def is_word_char(ch: str) -> bool:
    # как в main.ipynb
    return is_cyr(ch) or is_lat(ch) or is_digit(ch) or ch in "-_"


def local_tokens(text: str, pos: int) -> Tuple[str, str, str]:
    """
    pos — позиция вставки пробела (между pos-1 и pos), 1..len(text)-1.
    Возвращает (left_token, right_token, merged_token) в нижнем регистре.
    """
    n = len(text)
    i = pos - 1
    j = pos
    # влево
    L = i
    while L >= 0 and is_word_char(text[L]):
        L -= 1
    left_tok = text[L+1:i+1]
    # вправо
    R = j
    while R < n and is_word_char(text[R]):
        R += 1
    right_tok = text[j:R]
    merged = (left_tok + right_tok) if (left_tok and right_tok) else ""
    return left_tok.lower(), right_tok.lower(), merged.lower()


def boundary_features(text: str, pos: int, window: int, lexicon=None) -> dict:
    feats = {}
    n = len(text)
    Lch = text[pos-1]
    Rch = text[pos]
    Lc = cat(Lch)
    Rc = cat(Rch)

    feats[f"LC:{Lch}"] = 1
    feats[f"RC:{Rch}"] = 1
    feats[f"LCAT:{Lc}"] = 1
    feats[f"RCAT:{Rc}"] = 1
    feats[f"TRAN:{trans(Lc,Rc)}"] = 1

    feats["L_isupper"] = 1 if Lch.isalpha() and Lch.isupper() else 0
    feats["R_isupper"] = 1 if Rch.isalpha() and Rch.isupper() else 0

    for k in range(1, window+1):
        if pos-1-k >= 0:
            ch = text[pos-1-k]
            feats[f"L{k}:{ch}"] = 1
            feats[f"L{k}C:{cat(ch)}"] = 1
        if pos+k < n:
            ch = text[pos+k]
            feats[f"R{k}:{ch}"] = 1
            feats[f"R{k}C:{cat(ch)}"] = 1

    feats["dig_to_alpha"] = 1 if (
        is_digit(Lch) and (is_cyr(Rch) or is_lat(Rch))) else 0
    feats["alpha_to_dig"] = 1 if (
        (is_cyr(Lch) or is_lat(Lch)) and is_digit(Rch)) else 0
    feats["cyr_to_lat"] = 1 if (is_cyr(Lch) and is_lat(Rch)) else 0
    feats["lat_to_cyr"] = 1 if (is_lat(Lch) and is_cyr(Rch)) else 0
    feats["punct_left"] = 1 if is_punct(Lch) else 0
    feats["punct_right"] = 1 if is_punct(Rch) else 0

    if lexicon is not None:
        lt, rt, merged = local_tokens(text, pos)
        feats["lex_left"] = 1 if lt and lt in lexicon else 0
        feats["lex_right"] = 1 if rt and rt in lexicon else 0
        feats["lex_merged"] = 1 if merged and merged in lexicon else 0
        feats["len_lt"] = len(lt)
        feats["len_rt"] = len(rt)
        feats["len_merged"] = len(merged)
    else:
        feats["len_lt"] = 0
        feats["len_rt"] = 0
        feats["len_merged"] = 0

    feats["rel_pos"] = pos / n
    feats["is_start_near"] = 1 if pos <= 2 else 0
    feats["is_end_near"] = 1 if (n - pos) <= 2 else 0
    return feats

# -----------------------------
# Модель и хешер (n_features берём из модели!)
# -----------------------------


class Predictor:
    def __init__(self, model, window: int, lexicon: Optional[Dict[str, int]]):
        self.model = model
        self.window = window
        self.lexicon = lexicon

        # ожидания модели по числу фич
        if hasattr(model, "coef_"):
            n_features_expected = int(model.coef_.shape[1])
        elif hasattr(model, "n_features_in_"):
            n_features_expected = int(model.n_features_in_)
        else:
            raise ValueError(
                "Не удалось определить число признаков у модели (coef_.shape[1] или n_features_in_)")

        # делаем хешер ТАКИМ ЖЕ, как на обучении
        self.hasher = FeatureHasher(
            n_features=n_features_expected,
            input_type="dict",
            alternate_sign=False  # как в ноутбуке
            # dtype по умолчанию (float64) — совпадает с классическим обучением
        )
        print(f"[info] ожидаемое число признаков: {n_features_expected}")

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def predict_probs_for_text(self, text: str) -> np.ndarray:
        n = len(text)
        if n <= 1:
            return np.zeros((0,), dtype=np.float32)
        feats = [boundary_features(
            text, pos, self.window, self.lexicon) for pos in range(1, n)]
        X = self.hasher.transform(feats)
        dec = self.model.decision_function(X)
        if isinstance(dec, list):
            dec = np.asarray(dec)
        probs = self._sigmoid(dec).astype(np.float32)
        return probs  # длина n-1; p[0] -> позиция 1


# -----------------------------
# Чтение датасета и колонка с текстом
# -----------------------------
PREFERRED_TEXT_COLUMNS = [
    "text_no_spaces", "text", "text_nospace", "query", "title", "input"
]


def _load_two_col_first_comma(path: Path) -> pd.DataFrame:
    """
    Ожидаем заголовок вида: id,<имя_колонки_текста>
    Далее строки: <id>,<весь_остальной_текст_как_есть_включая_запятые>
    """
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:  # utf-8 с авто-съеданием BOM
        header = f.readline()
        if not header:
            raise ValueError(f"{path}: пустой файл")
        header = header.strip().lstrip("\ufeff")
        # имя текстовой колонки забираем из заголовка
        if "," not in header:
            raise ValueError(
                f"{path}: в заголовке нет запятой (ожидали 'id,<text_col>')")
        _, text_col_name = header.split(",", 1)
        text_col_name = text_col_name.strip()

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # делим ТОЛЬКО по первой запятой
            comma = line.find(",")
            if comma < 0:
                # строка без запятой — пропускаем или трактуем как пустой текст
                continue
            id_part = line[:comma].strip()
            text_part = line[comma + 1:]  # всё остальное — текст как есть
            # пытаемся привести id к int, но не настаиваем
            try:
                idx = int(id_part)
            except Exception:
                idx = id_part
            rows.append((idx, text_part))

    if not rows:
        # пусть лучше упадём, чем вернём пустоту
        raise ValueError(
            f"{path}: не удалось прочитать ни одной строки данных")

    df = pd.DataFrame(rows, columns=["id", text_col_name])
    return df


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Унифицированный загрузчик:
      1) Для .txt — всегда используем безопасный парсер по первой запятой.
      2) Для .csv — сначала пробуем pandas autodetect; если распилило на >2 колонок
         и не находится текстовой колонки, откатываемся на безопасный парсер.
    """
    if not path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {path}")

    ext = path.suffix.lower()
    if ext == ".txt":
        return _load_two_col_first_comma(path)

    # .csv (или что-то ещё): сначала пробуем обычный pd.read_csv
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # если парсер не справился — используем безопасный
        return _load_two_col_first_comma(path)

    # если колонка с текстом нашлась — ок
    cols_lower = {c.lower(): c for c in df.columns}
    if any(c in cols_lower for c in PREFERRED_TEXT_COLUMNS):
        return df

    # если распилило на кучу колонок и нет явной текстовой — лечим
    if df.shape[1] > 2:
        return _load_two_col_first_comma(path)

    # 2 колонки — допустим, первая 'id', вторая — текст
    return df


def pick_text_column(df: pd.DataFrame) -> str:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in PREFERRED_TEXT_COLUMNS:
        if cand in cols_lower:
            return cols_lower[cand]
    # если нет знакомых — берём первую object-колонку
    for c in df.columns:
        if pd.api.types.is_object_dtype(df[c]):
            return c
    # иначе берём последнюю колонку (обычно это текст во входных наборах id,text)
    return df.columns[-1]

# -----------------------------
# Формирование predicted_positions
# -----------------------------


def positions_to_string(pos_set: set) -> str:
    """
    Возвращает строку ровно в формате "[5, 8, 13]" (скобки + пробел после запятой).
    Пустое множество -> "[]".
    """
    if not pos_set:
        return "[]"
    return "[" + ", ".join(str(i) for i in sorted(pos_set)) + "]"


def predict_positions_for_series(series: pd.Series, predictor: Predictor, thr: float) -> pd.Series:
    """
    Копия task_data + колонка predicted_positions как СТРОКА "[...]" для каждой строки.
    """
    out = []
    for s in series.astype(str):
        probs = predictor.predict_probs_for_text(s)
        if probs.size == 0:
            out.append("[]")
            continue
        pred_idx = np.where(probs >= thr)[0] + 1  # p[0] -> позиция 1
        if pred_idx.size == 0:
            out.append("[]")
        else:
            out.append(positions_to_string(set(int(i) for i in pred_idx)))
    return pd.Series(out, index=series.index, dtype="string")
# -----------------------------
# CLI
# -----------------------------


def main():
    parser = argparse.ArgumentParser(description="Space restoration inference")
    parser.add_argument("-i", "--input", type=str, default="dataset.csv",
                        help="Входной CSV/TSV с текстами (по умолчанию dataset.csv)")
    parser.add_argument("-o", "--output", type=str, default=str(DEFAULT_OUTPUT),
                        help="Файл сабмита (по умолчанию submission.csv)")
    args = parser.parse_args()

    cfg = load_config()
    model = load_model()
    thr = load_threshold()
    lexicon = load_lexicon(cfg.get("LEXICON_TOPK"))

    predictor = Predictor(
        model=model,
        window=int(cfg["WINDOW"]),
        lexicon=lexicon if lexicon else None
    )

    inp_path = Path(args.input)
    out_path = Path(args.output)

    print(f"[info] читаю датасет: {inp_path}")
    df = load_dataset(inp_path)
    text_col = pick_text_column(df)
    print(f"[info] колонка с текстом: '{text_col}'")
    print(f"[info] размер: {len(df)} строк")
    print(f"[info] порог: {thr:.4f}")

    df = df.copy()
    df["predicted_positions"] = predict_positions_for_series(
        df[text_col], predictor, thr)
    df.to_csv(out_path, index=False)
    print(f"[done] сохранено: {out_path} ({out_path.stat().st_size} байт)")


if __name__ == "__main__":
    main()
