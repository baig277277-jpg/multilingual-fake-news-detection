"""Dataset loading, text preprocessing, label normalisation and leakage-aware splitting."""
from __future__ import annotations

import hashlib
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Public dataset used by default for the real-data run (verified on the Zenodo
# record page): one CSV and its MD5. The licence and MD5 were NOT independently
# re-verified; check the Zenodo record page. A wrong MD5 makes the download fail loudly.
# ---------------------------------------------------------------------------
FAKECOVID_RECORD = "https://zenodo.org/records/3965871"
FAKECOVID_URL = "https://zenodo.org/records/3965871/files/FakeCovid_July2020.csv?download=1"
FAKECOVID_FILENAME = "FakeCovid_July2020.csv"
FAKECOVID_MD5 = "c8f2774f9315f6311b2ab78e41ea9bea"
FAKECOVID_PAPER = "https://arxiv.org/abs/2006.11343"

LABEL_NAMES = {0: "Real", 1: "Potentially Fake"}
MIN_TEXT_CHARS = 10
MIN_SUPERVISED_ROWS = 300

# Column-name aliases (matched after lower-casing and replacing spaces by "_").
TEXT_ALIASES = [
    "text", "content", "article", "article_text", "content_text", "full_text", "body",
    "claim", "statement", "headline", "title", "source_title", "reference_title",
]
LABEL_ALIASES = [
    "label", "class", "fact_check_class", "factcheck_class", "verdict", "rating",
    "target", "truth_label",
]
LANGUAGE_ALIASES = ["language", "lang", "language_code"]
DATE_ALIASES = [
    "date", "published_date", "publish_date", "datetime", "timestamp", "published", "created_at",
]
GROUP_ALIASES = ["claim_id", "group_id", "group"]

LANGUAGE_NAME_TO_CODE = {
    "english": "en", "hindi": "hi", "german": "de", "spanish": "es", "french": "fr",
    "portuguese": "pt", "italian": "it", "russian": "ru", "arabic": "ar", "bengali": "bn",
    "bangla": "bn", "marathi": "mr", "tamil": "ta", "telugu": "te", "malayalam": "ml",
    "urdu": "ur", "gujarati": "gu", "punjabi": "pa", "kannada": "kn", "indonesian": "id",
    "dutch": "nl", "polish": "pl", "turkish": "tr",
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_fakecovid(destination: str | Path, timeout: int = 120, retries: int = 3) -> Path:
    """Download the FakeCovid CSV and verify it against the checksum on the Zenodo record."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        FAKECOVID_URL,
        headers={"User-Agent": "Multilingual-fake-news-course-project/1.0", "Accept": "text/csv,*/*"},
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            digest = hashlib.md5()
            size = 0
            with urllib.request.urlopen(request, timeout=timeout) as response, open(temporary, "wb") as fh:
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size < 1_000_000:
                raise RuntimeError(f"Downloaded file is unexpectedly small ({size} bytes).")
            if digest.hexdigest() != FAKECOVID_MD5:
                raise RuntimeError(
                    f"Checksum mismatch: got {digest.hexdigest()}, the Zenodo record lists {FAKECOVID_MD5}."
                )
            temporary.replace(destination)
            return destination
        except Exception as exc:  # noqa: BLE001 - report the final failure below
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 4))
    if temporary.exists():
        temporary.unlink()
    raise RuntimeError(
        f"Failed to download/validate FakeCovid after {retries} attempts from {FAKECOVID_URL}. "
        f"Last error: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WS = re.compile(r"\s+")


def preprocess_text(text) -> str:
    """Unicode-normalise, replace URLs by a token and collapse whitespace.

    Case is preserved on purpose: cased multilingual encoders expect natural text.
    """
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _URL.sub(" <URL> ", text)
    return _WS.sub(" ", text).strip()


def normalize_language(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unknown"
    key = str(value).strip().lower()
    if not key:
        return "unknown"
    return LANGUAGE_NAME_TO_CODE.get(key, key)


def detect_languages(texts: list[str]) -> list[str] | None:
    """Optional language identification. Returns None when `langdetect` is not installed."""
    try:
        from langdetect import DetectorFactory, detect
    except ImportError:
        return None
    DetectorFactory.seed = 0
    result = []
    for text in texts:
        try:
            result.append(detect(text))
        except Exception:  # noqa: BLE001 - langdetect raises for empty/odd strings
            result.append("unknown")
    return result


# Verdicts that are ambiguous are dropped rather than forced into a class.
_AMBIGUOUS = (
    "partly", "partially", "half", "mixture", "mixed", "unproven", "unverified", "unknown",
    "unclear", "outdated", "misattributed", "satire", "opinion", "not enough", "cannot",
    "can't", "needs context", "missing context", "no evidence",
)
_NEGATED_REAL = ("not true", "untrue", "not correct", "not accurate", "not real", "isn't true")
_FAKE_WORDS = (
    "false", "fake", "fabricat", "misleading", "incorrect", "hoax", "scam", "fraud", "bogus",
    "pants on fire", "distort", "manipulat",
)
_REAL_WORDS = ("true", "real", "correct", "accurate", "genuine", "authentic", "legit")


def normalize_label(value) -> int | None:
    """Map a raw verdict to 1 (potentially fake), 0 (real) or None (dropped).

    Numeric labels are assumed to be 1 = fake and 0 = real. String verdicts follow
    fact-checking usage ("false" -> fake, "true" -> real). Use --label-map to override.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"0", "1", "0.0", "1.0"}:
        return int(float(text))
    if any(word in text for word in _AMBIGUOUS):
        return None
    if any(word in text for word in _NEGATED_REAL):
        return 1
    if any(word in text for word in _FAKE_WORDS):
        return 1
    if any(word in text for word in _REAL_WORDS):
        return 0
    return None


def map_labels(series: pd.Series, label_map: dict | None = None) -> pd.Series:
    """Apply an explicit mapping (unmapped values dropped) or the default heuristics."""
    if label_map:
        lookup = {str(k).strip().lower(): v for k, v in label_map.items()}

        def convert(value):
            mapped = lookup.get(str(value).strip().lower())
            return None if mapped is None else int(mapped)
    else:
        convert = normalize_label
    return pd.to_numeric(series.map(convert), errors="coerce")


def _norm_name(name) -> str:
    return re.sub(r"\s+", "_", str(name).strip().lower())


def _find_column(df: pd.DataFrame, explicit: str | None, aliases: list[str], what: str, required: bool):
    lookup = {_norm_name(c): c for c in df.columns}
    if explicit:
        key = _norm_name(explicit)
        if key not in lookup:
            raise ValueError(
                f"The {what} column '{explicit}' was not found. Available columns: {list(df.columns)}"
            )
        return lookup[key]
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    if required:
        raise ValueError(
            f"Could not find a {what} column (tried {aliases}). "
            f"Available columns: {list(df.columns)}. Pass it explicitly with --{what}-col."
        )
    return None


def _drop_duplicate_texts(frame: pd.DataFrame, info: dict) -> pd.DataFrame:
    conflict = frame.groupby("text")["label"].transform("nunique") > 1
    info["dropped_conflicting_duplicate_texts"] = int(conflict.sum())
    frame = frame[~conflict]
    duplicated = frame.duplicated("text", keep="first")
    info["dropped_exact_duplicate_texts"] = int(duplicated.sum())
    return frame[~duplicated]


def load_dataset_csv(
    path: str | Path,
    text_col: str | None = None,
    label_col: str | None = None,
    language_col: str | None = None,
    date_col: str | None = None,
    group_col: str | None = None,
    label_map: dict | None = None,
    dayfirst: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Load a CSV into the project schema: text, label, language [, date, claim_id]."""
    path = Path(path)
    raw = pd.read_csv(path, encoding="utf-8", encoding_errors="replace", low_memory=False)
    info: dict = {"raw_rows": int(len(raw)), "columns_in_file": [str(c) for c in raw.columns]}

    tcol = _find_column(raw, text_col, TEXT_ALIASES, "text", True)
    lcol = _find_column(raw, label_col, LABEL_ALIASES, "label", True)
    gcol = _find_column(raw, group_col, GROUP_ALIASES, "group", False)
    langcol = _find_column(raw, language_col, LANGUAGE_ALIASES, "language", False)
    dcol = _find_column(raw, date_col, DATE_ALIASES, "date", False)
    info["columns_used"] = {"text": tcol, "label": lcol, "language": langcol, "date": dcol, "group": gcol}

    frame = pd.DataFrame({"text": raw[tcol].map(preprocess_text)})
    info["raw_label_counts"] = {str(k): int(v) for k, v in raw[lcol].astype(str).value_counts().head(40).items()}
    frame["label"] = map_labels(raw[lcol], label_map)

    if langcol is not None:
        frame["language"] = raw[langcol].map(normalize_language)
        info["language_source"] = f"column '{langcol}'"
    else:
        detected = detect_languages(frame["text"].tolist())
        if detected is None:
            frame["language"] = "unknown"
            info["language_source"] = "unknown (no language column and 'langdetect' is not installed)"
        else:
            frame["language"] = detected
            info["language_source"] = "langdetect (automatic; can be wrong on short or code-mixed text)"

    if dcol is not None:
        frame["date"] = pd.to_datetime(raw[dcol], utc=True, errors="coerce", dayfirst=dayfirst)
        info["date_parse_rate"] = float(frame["date"].notna().mean())
    if gcol is not None:
        frame["claim_id"] = raw[gcol].astype(str)

    before = len(frame)
    frame = frame[frame["text"].str.len() >= MIN_TEXT_CHARS]
    info["dropped_short_or_empty_text"] = int(before - len(frame))

    before = len(frame)
    frame = frame.dropna(subset=["label"]).copy()
    info["dropped_unmapped_or_ambiguous_label"] = int(before - len(frame))
    frame["label"] = frame["label"].astype(int)

    frame = _drop_duplicate_texts(frame, info).reset_index(drop=True)
    info["rows_after_cleaning"] = int(len(frame))
    info["label_counts"] = {LABEL_NAMES[int(k)]: int(v) for k, v in frame["label"].value_counts().items()}
    info["language_counts"] = {str(k): int(v) for k, v in frame["language"].value_counts().items()}
    if frame.empty:
        raise ValueError("No usable rows remain after cleaning. Check --label-col / --label-map.")
    return frame, info


def load_parallel_csv(path: str | Path) -> pd.DataFrame:
    """Load a label-free parallel file with columns text, language, claim_id."""
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    lookup = {_norm_name(c): c for c in df.columns}
    missing = {"text", "language", "claim_id"}.difference(lookup)
    if missing:
        raise ValueError(f"Parallel CSV is missing columns {sorted(missing)}; found {list(df.columns)}.")
    out = pd.DataFrame(
        {
            "text": df[lookup["text"]].map(preprocess_text),
            "language": df[lookup["language"]].map(normalize_language),
            "claim_id": df[lookup["claim_id"]].astype(str),
        }
    )
    return out[out["text"].str.len() >= MIN_TEXT_CHARS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def _class_counts(frame: pd.DataFrame) -> dict:
    return {LABEL_NAMES[int(k)]: int(v) for k, v in frame["label"].value_counts().items()}


def _require_both_classes(parts: dict[str, pd.DataFrame], strategy: str) -> None:
    for name, part in parts.items():
        if part["label"].nunique() < 2:
            raise ValueError(
                f"The {name} split contains a single class under the '{strategy}' strategy. "
                "Use a different --seed or --split option, or a larger dataset."
            )


def split_frame(
    frame: pd.DataFrame,
    strategy: str = "auto",
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
):
    """Split into train/validation/test (70/15/15) without leaking related claims.

    auto: grouped by claim_id when translations/duplicates share an id, otherwise
    chronological when dates are available, otherwise a stratified random split.
    Rows from one claim_id never appear in two splits, so a translation of a test
    claim cannot be memorised from the training set.
    """
    n = len(frame)
    if n < MIN_SUPERVISED_ROWS:
        raise ValueError(f"Only {n} labelled rows remain. At least {MIN_SUPERVISED_ROWS} are required.")
    has_groups = "claim_id" in frame and frame["claim_id"].nunique() < n
    has_dates = "date" in frame and frame["date"].notna().mean() >= 0.9
    if strategy == "auto":
        strategy = "grouped" if has_groups else ("chronological" if has_dates else "stratified")
    notes: list[str] = []

    if strategy == "grouped":
        if "claim_id" not in frame:
            raise ValueError("The grouped split needs a claim_id column (or --group-col).")
        rng = np.random.default_rng(seed)
        groups = np.array(sorted(frame["claim_id"].unique()), dtype=object)
        rng.shuffle(groups)
        sizes = frame.groupby("claim_id").size()
        n_train, n_val = fractions[0] * n, fractions[1] * n
        assignment, running = {}, 0
        for group in groups:
            assignment[group] = "train" if running < n_train else ("val" if running < n_train + n_val else "test")
            running += int(sizes[group])
        part = frame["claim_id"].map(assignment)
        train, val, test = (frame[part == name] for name in ("train", "val", "test"))
    elif strategy == "chronological":
        if "date" not in frame:
            raise ValueError("The chronological split needs a date column (or --date-col).")
        dated = frame.dropna(subset=["date"]).sort_values("date", kind="stable")
        notes.append(f"{n - len(dated)} rows without a parseable date were excluded from this split.")
        i_train, i_val = int(len(dated) * fractions[0]), int(len(dated) * (fractions[0] + fractions[1]))
        train, val, test = dated.iloc[:i_train], dated.iloc[i_train:i_val], dated.iloc[i_val:]
    elif strategy == "stratified":
        stratify = frame["label"] if frame["label"].value_counts().min() >= 3 else None
        train, temp = train_test_split(frame, test_size=fractions[1] + fractions[2], stratify=stratify, random_state=seed)
        stratify_temp = temp["label"] if temp["label"].value_counts().min() >= 2 else None
        val, test = train_test_split(temp, test_size=fractions[2] / (fractions[1] + fractions[2]), stratify=stratify_temp, random_state=seed)
        notes.append(
            "Random split: near-duplicate or translated versions of the same claim can land in different "
            "splits, which can inflate scores. Prefer a claim_id or date column when available."
        )
    else:
        raise ValueError(f"Unknown split strategy '{strategy}'.")

    _require_both_classes({"training": train, "validation": val, "test": test}, strategy)
    info = {
        "strategy": strategy,
        "fractions": list(fractions),
        "counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "class_counts": {"train": _class_counts(train), "validation": _class_counts(val), "test": _class_counts(test)},
        "language_counts": {
            name: {str(k): int(v) for k, v in part["language"].value_counts().items()}
            for name, part in (("train", train), ("validation", val), ("test", test))
        },
        "notes": notes,
    }
    if strategy == "chronological":
        info["test_period"] = [str(test["date"].min()), str(test["date"].max())]
    return train.copy(), val.copy(), test.copy(), info


# ---------------------------------------------------------------------------
# Synthetic demonstration data (software smoke test only)
# ---------------------------------------------------------------------------
# (English, Hindi, romanised Hindi). Hindi subjects are stored in the form that
# is grammatical before "के बारे में".
_SUBJECTS = [
    ("the city water supply", "शहर की पानी की आपूर्ति", "shehar ki paani ki aapurti"),
    ("the new vaccine", "नए टीके", "naye teeke"),
    ("the local train service", "स्थानीय रेल सेवा", "sthaaneeya rail seva"),
    ("the national highway", "राष्ट्रीय राजमार्ग", "raashtreeya raajmaarg"),
    ("the school admission process", "स्कूल प्रवेश प्रक्रिया", "school pravesh prakriya"),
    ("the electricity connection", "बिजली कनेक्शन", "bijli connection"),
    ("the village hospital", "गाँव के अस्पताल", "gaon ke aspataal"),
    ("the bank loan scheme", "बैंक ऋण योजना", "bank rin yojana"),
    ("the ration card scheme", "राशन कार्ड योजना", "ration card yojana"),
    ("the metro project", "मेट्रो परियोजना", "metro pariyojana"),
    ("the flood relief fund", "बाढ़ राहत कोष", "baadh raahat kosh"),
    ("the pension scheme", "पेंशन योजना", "pension yojana"),
    ("the public library", "सार्वजनिक पुस्तकालय", "saarvajanik pustakaalay"),
    ("the mobile network service", "मोबाइल नेटवर्क सेवा", "mobile network seva"),
    ("the property tax system", "संपत्ति कर प्रणाली", "sampatti kar pranaali"),
    ("the garbage collection service", "कचरा संग्रहण सेवा", "kachra sangrahan seva"),
    ("the scholarship programme", "छात्रवृत्ति कार्यक्रम", "chhaatravritti kaaryakram"),
    ("the digital payment app", "डिजिटल भुगतान ऐप", "digital bhugtaan app"),
    ("the bus pass system", "बस पास प्रणाली", "bus pass pranaali"),
    ("the farm subsidy scheme", "कृषि सब्सिडी योजना", "krishi subsidy yojana"),
    ("the community health camp", "सामुदायिक स्वास्थ्य शिविर", "saamudaayik svaasthya shivir"),
    ("the airport expansion", "हवाई अड्डे के विस्तार", "havaai adde ke vistaar"),
    ("the traffic fine system", "यातायात जुर्माना प्रणाली", "yaataayaat jurmaana pranaali"),
    ("the voter registration drive", "मतदाता पंजीकरण अभियान", "matdaata panjeekaran abhiyaan"),
]

_FAKE_STYLE = [
    (
        "Forwarded message claims the truth about {s} is being hidden, share with everyone before it is deleted!",
        "फॉरवर्ड संदेश का दावा है कि {s} के बारे में सच छिपाया जा रहा है, डिलीट होने से पहले सबको शेयर करें!",
        "Forward message ka daava hai ki {s} ke baare mein sach chhipaya ja raha hai, delete hone se pehle sabko share karein!",
    ),
    (
        "SHOCKING secret about {s} that doctors and officials do not want you to know, 100% proven!",
        "{s} के बारे में चौंकाने वाला राज़ जो डॉक्टर और अधिकारी आपको नहीं बताना चाहते, 100% सिद्ध!",
        "{s} ke baare mein chaunkane wala raaz jo doctor aur adhikari aapko nahi batana chahte, 100% siddh!",
    ),
    (
        "Anonymous source says that {s} will change overnight and everyone will lose money, no official confirmation needed.",
        "अज्ञात सूत्र का कहना है कि {s} के बारे में रातों-रात सब बदल जाएगा और सबका पैसा डूबेगा, किसी आधिकारिक पुष्टि की ज़रूरत नहीं।",
        "Agyaat sutra ka kehna hai ki {s} ke baare mein raaton-raat sab badal jayega aur sabka paisa doobega, kisi aadhikarik pushti ki zaroorat nahi.",
    ),
]
_REAL_STYLE = [
    (
        "The government department published an official notice about {s} and shared the full document on its website.",
        "सरकारी विभाग ने {s} के बारे में आधिकारिक सूचना जारी की और पूरा दस्तावेज़ अपनी वेबसाइट पर साझा किया।",
        "Sarkari vibhag ne {s} ke baare mein aadhikarik soochna jaari ki aur poora dastavez apni website par saajha kiya.",
    ),
    (
        "A news report about {s} cites the official statistics and includes comments from named experts.",
        "{s} के बारे में एक समाचार रिपोर्ट आधिकारिक आँकड़ों का हवाला देती है और नामित विशेषज्ञों की टिप्पणियाँ शामिल करती है।",
        "{s} ke baare mein ek samachar report aadhikarik aankdon ka hawala deti hai aur naamit visheshagyon ki tippaniyan shaamil karti hai.",
    ),
    (
        "Officials confirmed the schedule for {s} at a press briefing and published the dates in the public record.",
        "अधिकारियों ने प्रेस ब्रीफिंग में {s} के बारे में कार्यक्रम की पुष्टि की और तारीखें सार्वजनिक रिकॉर्ड में प्रकाशित कीं।",
        "Adhikariyon ne press briefing mein {s} ke baare mein kaaryakram ki pushti ki aur taareekhein saarvajanik record mein prakaashit keen.",
    ),
]
DEMO_LANGUAGES = ("en", "hi", "hi-Latn")


def generate_synthetic_demo(path: str | Path, seed: int = 42, label_noise: float = 0.06) -> Path:
    """Write a clearly labelled, template-generated parallel dataset (English, Hindi, romanised Hindi).

    Every claim exists in all three languages under one claim_id. Labels are decided by the
    template style (sensational vs. official wording), then a fraction is flipped per claim
    to avoid a trivially perfect score. This is only for checking that the software runs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows, claim_number = [], 0
    for subject in _SUBJECTS:
        for label, templates in ((1, _FAKE_STYLE), (0, _REAL_STYLE)):
            for template in templates:
                claim_id = f"demo_{claim_number:04d}"
                claim_number += 1
                final = 1 - label if rng.random() < label_noise else label
                for index, language in enumerate(DEMO_LANGUAGES):
                    rows.append(
                        {
                            "claim_id": claim_id,
                            "text": template[index].format(s=subject[index]),
                            "label": "fake" if final == 1 else "real",
                            "language": language,
                            "data_origin": "synthetic_demo",
                        }
                    )
    frame = pd.DataFrame(rows).iloc[rng.permutation(len(rows))].reset_index(drop=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path
