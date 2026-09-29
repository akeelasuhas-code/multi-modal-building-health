# Building Health Auditing System, v2

Replace your GitHub repo's `app.py` and `requirements.txt` with these files (keep them all in the repo root).

| File | Job |
|---|---|
| `app.py` | Streamlit UI only |
| `bhas_core.py` | All the maths: segmentation, crack width/length, severity, InSAR fit, construction-year detector, fusion, clIoU/clDice |
| `bhas_pipeline.py` | Runs one audit; drops missing data instead of faking it; writes the warnings |
| `bhas_gee.py` | Earth Engine (the only file that imports `ee`) |
| `bhas_db.py` | SQLite log + time-bound escalation |
| `bhas_report.py` | Two-page PDF (plain language + technical/contractor) |
| `evaluate.py`, `synth.py` | Measure accuracy (synthetic regression test, or a real labelled dataset) |
| `test_bhas.py` | 30 tests: `pytest -q test_bhas.py` |

## Run

```
pip install -r requirements.txt -r requirements-dev.txt
pytest -q test_bhas.py
python evaluate.py --synthetic
streamlit run app.py
```

In Colab: upload the files (or `git clone` your repo), `!pip install -r requirements.txt -r requirements-dev.txt`, then `!pytest -q test_bhas.py`. In Colab, `ee.Authenticate()` once and `bhas_gee.init_ee(project="suhas-proj1")` will pick up your credentials.

## Earth Engine on Streamlit Cloud

Your Colab login does not exist on Streamlit's servers, so a hosted app needs a service account. Per Google's docs: the Cloud project must be registered for Earth Engine with the Earth Engine API enabled; the service account needs the **Earth Engine Resource Viewer** role, and possibly **Service Usage Consumer**.

1. Cloud Console, project `suhas-proj1`: IAM & Admin > Service Accounts > Create. Grant the role(s) above.
2. Keys > Add key > JSON. Download it.
3. Streamlit Cloud > your app > Settings > Secrets: paste the JSON fields in the format of `secrets.toml.example`.
4. Never commit the key. (`.gitignore` excludes `*.json`.)

If you skip this, the app still works: it just shows "Earth Engine not connected" and the age step is disabled. Nothing is mocked.

## Score real accuracy (the number a paper needs)

Download the test subsets of OmniCrack30k (or CRACK500 / DeepCrack), then:

```
python evaluate.py --images path/to/images --masks path/to/masks --size 256 --out results.csv
```

It reports clIoU (4 px tolerance), clDice, IoU and F1, plus your old method as a baseline on the same images. Masks must share the image's file name (a `_mask` suffix is fine). `--size 256` matches how OmniCrack30k results are published.

## Provisional (not fitted, not validated)

- Critical crack width 0.3 mm, density 1 m/m2, anomaly fraction 10%, and all weights and status thresholds. Check the width against IS 456 / ACI 224R for your exposure class, then calibrate the rest on expert grades (`fit_logistic` and `auroc` are in `bhas_core.py`).
- `k = 5` crack sensitivity was tuned on synthetic concrete.
- Construction-year thresholds were set from simulated noise. Real Landsat series have autocorrelation and will false-alarm more.
- Escalation writes notices to the log; it does not send email. Streamlit Cloud's disk is wiped on restart, so export the CSV.
