"""ECB Hawkish-Dovish Monitor.

Classifies policy-relevant sentences in ECB Executive Board speeches as hawkish /
dovish / neutral with a RoBERTa-base model fine-tuned on annotated ECB communication
(LorenzoAleCon29/roberta-base-dapt-ecb-hawkish-dovish, reported test accuracy 0.650; up to 10
random policy sentences per speech), aggregates them
into a speech-level and monthly hawkishness index, and tests whether the index
(1) anticipates deposit-facility-rate changes and (2) moves short-term euro yields.

    python hawkdove.py sentences   # extract policy sentences   -> data/sentences.csv
    python hawkdove.py classify    # label them (resumable)     -> data/labels.csv
    python hawkdove.py analyze     # index, regressions, charts -> results/
"""
import re
import sys
import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).parent / "data"
RESULTS = pathlib.Path(__file__).parent / "results"
START = "2004-09-06"  # first date of the ECB AAA yield-curve series
MODEL = "LorenzoAleCon29/roberta-base-dapt-ecb-hawkish-dovish"  # trained on ECB text; reported test accuracy 0.650
MAX_PER_SPEECH = 10  # random sample per speech: the large model ran at 1.4 sentences/s on CPU (~10 h for all)
POLICY = re.compile(
    r"inflation|price stability|interest rate|policy rate|monetary policy|tighten|accommodat|"
    r"asset purchase|deposit facility|rate hike|rate cut|wage", re.I)


# ---------- text ----------
def policy_sentences(text: str) -> list[str]:
    """Sentences of 8-80 words that mention monetary-policy vocabulary."""
    out = []
    text = re.sub(r"\[\d+\]", "", text or "")  # footnote markers ("term.[1] The") would block the split
    for s in re.split(r"(?<=[.!?])\s+", text):
        s = s.strip()
        if 8 <= len(s.split()) <= 80 and POLICY.search(s):
            out.append(s)
    return out


def hawkishness(labels: pd.Series) -> float:
    """(hawkish - dovish) / all classified sentences, in [-1, 1] (measure used in the paper)."""
    n = len(labels)
    return 0.0 if n == 0 else ((labels == "hawkish").sum() - (labels == "dovish").sum()) / n


# ---------- pipeline steps ----------
def extract() -> None:
    d = pd.read_csv(DATA / "all_ECB_speeches.csv", sep="|", quoting=3, dtype=str,
                    on_bad_lines="skip", encoding="utf-8").dropna(subset=["contents"])
    d = d[d["date"] >= START].reset_index(drop=True)
    rows = [(i, r.date, r.speakers, s) for i, r in d.iterrows() for s in policy_sentences(r.contents)]
    out = pd.DataFrame(rows, columns=["speech_id", "date", "speaker", "sentence"])
    sample = out.sample(frac=1, random_state=0).groupby("speech_id").head(MAX_PER_SPEECH).sort_index()
    sample.reset_index(drop=True).to_csv(DATA / "sentences.csv", index=False)
    print(f"{d.shape[0]} speeches -> {len(out)} policy sentences -> {len(sample)} sampled (max {MAX_PER_SPEECH}/speech)")


def classify(chunk: int = 256) -> None:
    from transformers import pipeline

    sents = pd.read_csv(DATA / "sentences.csv")
    out = DATA / "labels.csv"
    done = len(pd.read_csv(out)) if out.exists() else 0  # resume after interruption
    clf = pipeline("text-classification", model=MODEL, device=-1)
    for start in range(done, len(sents), chunk):
        part = sents.iloc[start:start + chunk]
        part = part.loc[part["sentence"].str.len().sort_values().index]  # similar lengths per batch = less padding
        res = clf(part["sentence"].tolist(), batch_size=32, truncation=True, max_length=128)
        assert {r["label"] for r in res} <= {"dovish", "hawkish", "neutral"}, "unexpected label names in model config"
        pd.DataFrame({"row": part.index, "label": [r["label"] for r in res],
                      "score": [r["score"] for r in res]}).to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"{start + len(part)}/{len(sents)}", flush=True)


def ecb_series(name: str) -> pd.Series:
    d = pd.read_csv(DATA / f"{name}.csv", usecols=["TIME_PERIOD", "OBS_VALUE"])
    return pd.Series(d["OBS_VALUE"].values, index=pd.to_datetime(d["TIME_PERIOD"]), name=name).sort_index()


def analyze() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import statsmodels.api as sm

    RESULTS.mkdir(exist_ok=True)
    sents = pd.read_csv(DATA / "sentences.csv")
    labels = pd.read_csv(DATA / "labels.csv").set_index("row")
    assert len(labels) == len(sents), f"classification incomplete: {len(labels)}/{len(sents)}"
    sents["label"] = labels["label"]
    sents["date"] = pd.to_datetime(sents["date"])

    speech = sents.groupby(["speech_id", "date", "speaker"])["label"].agg(hawkishness).rename("score").reset_index()
    speech.to_csv(RESULTS / "speech_scores.csv", index=False)
    mix = sents["label"].value_counts(normalize=True).round(3).to_dict()

    # --- monthly index vs policy ---
    month = speech.set_index("date")["score"].resample("ME").mean().rename("index")
    dfr = ecb_series("dfr").resample("ME").last()
    hicp = ecb_series("hicp_yoy").resample("ME").last()
    m = pd.concat([month, dfr, hicp], axis=1).loc[START:]
    m["dfr_next3"] = m["dfr"].shift(-3) - m["dfr"]
    m["dfr_past3"] = m["dfr"] - m["dfr"].shift(3)
    reg = m.dropna()
    X = sm.add_constant(reg[["index", "hicp_yoy", "dfr_past3"]])
    policy = sm.OLS(reg["dfr_next3"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    # --- event study: speech-day change in the 2y yield ---
    y2, y10 = ecb_series("yield_2y"), ecb_series("yield_10y")
    dy = pd.concat([y2.diff().rename("d2y"), y10.diff().rename("d10y")], axis=1) * 100  # basis points
    # The series holds the date a new rate takes EFFECT; the decision is announced up to a week earlier.
    effective = ecb_series("dfr").diff().loc[lambda s: s != 0].index
    near_decision = {e - pd.Timedelta(days=k) for e in effective for k in range(8)}
    day = speech.groupby("date")["score"].mean().to_frame()
    ev = day.join(dy, how="inner").dropna()
    ev = ev[~ev.index.isin(near_decision)]
    ev["score_z"] = (ev["score"] - ev["score"].mean()) / ev["score"].std()
    event = {k: sm.OLS(ev[k], sm.add_constant(ev[["score_z"]])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
             for k in ("d2y", "d10y")}

    # --- charts ---
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(m.index, m["index"].rolling(3).mean(), color="#b03a2e", label="Hawkishness index (3-month avg)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("Hawkish − dovish share")
    ax2 = ax.twinx()
    ax2.plot(m.index, m["dfr"], color="#1f4e79", label="Deposit facility rate (%)")
    ax2.plot(m.index, m["hicp_yoy"], color="#7f8c8d", ls="--", lw=1, label="HICP inflation (% y/y)")
    ax2.set_ylabel("%")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=8)
    ax.set_title("ECB speeches: hawkishness index vs policy rate and inflation")
    fig.tight_layout(); fig.savefig(RESULTS / "index_vs_policy.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(ev["score"], ev["d2y"], s=6, alpha=0.4, color="#1f4e79")
    b = event["d2y"].params
    xs = np.linspace(ev["score"].min(), ev["score"].max(), 50)
    ax.plot(xs, b["const"] + b["score_z"] * (xs - ev["score"].mean()) / ev["score"].std(), color="#b03a2e")
    ax.set_xlabel("Speech hawkishness score"); ax.set_ylabel("Same-day change in 2y yield (bp)")
    ax.set_title("Speech tone vs 2-year euro yield (speech days)")
    fig.tight_layout(); fig.savefig(RESULTS / "event_study_2y.png", dpi=150); plt.close(fig)

    corr = m[["index", "hicp_yoy"]].dropna().corr().iloc[0, 1]
    summary = f"""# Results (generated by `python hawkdove.py analyze`)

Sample: {speech['date'].min():%Y-%m-%d} to {speech['date'].max():%Y-%m-%d}; {len(speech)} speeches; {len(sents)} policy sentences.
Sentence mix: {mix}

## 1. Does speech tone anticipate rate changes?
OLS, dependent variable = change in deposit facility rate over the next 3 months (pp), monthly, HAC(3) s.e., n = {int(policy.nobs)}.
```
{policy.summary2().tables[1].round(4).to_string()}
```
R² = {policy.rsquared:.3f}. Correlation of monthly index with HICP inflation: {corr:.2f}.

## 2. Do yields react on speech days?
OLS, dependent variable = same-day change in AAA euro yield (bp), regressor = standardised speech score, HAC(5) s.e.; days in the week before a deposit-rate change excluded (decision announcements); n = {int(event['d2y'].nobs)} speech days.
| Yield | Coef. per 1 s.d. of tone (bp) | t-stat | p-value | R² |
|---|---|---|---|---|
""" + "\n".join(f"| {k} | {event[k].params['score_z']:.3f} | {event[k].tvalues['score_z']:.2f} | {event[k].pvalues['score_z']:.3f} | {event[k].rsquared:.4f} |"
                 for k in ("d2y", "d10y")) + "\n"
    (RESULTS / "results.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    {"sentences": extract, "classify": classify, "analyze": analyze}[sys.argv[1]]()
