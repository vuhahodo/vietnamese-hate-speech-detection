"""End-to-end CPU latency of the T1 classical pipelines + export of the artifacts.

Fits BASE / FULL / EA on the training split (same configs as final_eval.py), checks that the
end-to-end pipeline reproduces the dev predictions of the search code, measures latency
(raw text -> prediction) and writes results/latency_t1.json plus artifacts/*.joblib
(artifacts/ is git-ignored). Test is not read.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib  # noqa: E402
import numpy as np  # noqa: E402

from vispam.data import load_splits  # noqa: E402
from vispam.evaluate import latency_ms, macro_f1  # noqa: E402
from vispam.pipeline import SpamPipeline  # noqa: E402
from vispam.search.fitness import SearchContext  # noqa: E402
from vispam.search.space import decode, encode  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from final_eval import ea_knee_vector  # noqa: E402


def main() -> None:
    x_ea, _ = ea_knee_vector()
    cands = {"BASE": decode(encode((0, 1, 0, 0), None, 0.0, "lr")),
             "FULL": decode(encode((1, 1, 1, 1), None, -1.0, "svc")),
             "EA": decode(x_ea)}
    splits = load_splits()[0]
    tr, dev = splits["train"], splits["dev"]
    ctx = SearchContext("t1")
    texts_dev = dev["text"].tolist()
    sample = texts_dev[:300]
    (ROOT / "artifacts").mkdir(exist_ok=True)
    out = {}
    for name, cfg in cands.items():
        t0 = time.time()
        pipe = SpamPipeline(cfg).fit(tr["text"].tolist(), tr["label"].to_numpy())
        fit_s = time.time() - t0
        pred = pipe.predict(texts_dev)
        ref = ctx.predict(cfg, "dev")
        agree = float((pred == ref).mean())
        batch_ms = latency_ms(pipe.predict, sample, repeats=3)
        singles = []
        for t in sample[:100]:
            t1 = time.perf_counter()
            pipe.predict([t])
            singles.append((time.perf_counter() - t1) * 1000)
        joblib.dump(pipe, ROOT / "artifacts" / f"t1_{name.lower()}.joblib")
        size_mb = (ROOT / "artifacts" / f"t1_{name.lower()}.joblib").stat().st_size / 1e6
        out[name] = {"n_features": pipe.n_features, "dev_macro_f1": macro_f1(dev["label"].to_numpy(), pred),
                     "agreement_with_search_code_dev": agree, "fit_seconds": fit_s,
                     "latency_ms_batch300": batch_ms,
                     "latency_ms_single_median": float(np.median(singles)),
                     "latency_ms_single_p95": float(np.percentile(singles, 95)),
                     "artifact_mb": size_mb}
        print(name, json.dumps(out[name]), flush=True)
    (ROOT / "results" / "latency_t1.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
