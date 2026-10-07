# BHAS: Building Health Auditing System

BHAS ranks buildings for inspection using two cheap sources of evidence: a close-up photograph of a wall, and the Landsat satellite record of the plot the building stands on. It finds and measures cracks in the photo, estimates when the building was constructed from 40 years of Landsat images, and writes a short inspection report for the owner and the engineer.

The project was built for buildings in Hyderabad, India, and is described in our paper (see [Citation](#citation)).

## What it does

- **Crack detection.** A classical image-processing pipeline (the default in the web app) and a ResNet-18 U-Net trained on public crack datasets.
- **Crack measurement.** Width in millimetres along every crack, crack length, and dark patches such as spalling or damp. A size reference in the photo (an A4 sheet or an ID card) converts pixels to millimetres.
- **Severity score.** A 0 to 100 score with three bands: no significant damage, needs monitoring, needs urgent inspection.
- **Construction year.** Dry-season Landsat 5/7/8/9 composites from 1985 to 2025, read through Google Earth Engine. Each site is compared against a 10 km background so that city-wide effects cancel out, and the first lasting change is taken as the construction year.
- **Ground motion (optional).** A velocity fit for InSAR displacement data, if you have it.
- **Reports and follow-up.** A two-page PDF report, an audit log, and an escalation ladder for public buildings that stay unattended.

## Results

Crack segmentation on the 237 DeepCrack test images (clIoU with a 4 px tolerance):

| Method | clIoU | Pixel F1 |
|---|---|---|
| First prototype (adaptive threshold) | 0.122 | 0.285 |
| Classical pipeline | 0.593 | 0.718 |
| U-Net (ResNet-18) | 0.737 | 0.661 |

The U-Net finds cracks better; the classical pipeline draws crack edges more accurately.

Construction year on 7 Hyderabad landmarks with documented dates: 5 of 7 estimates fell inside the construction window, against 1 of 7 for our first detector. This is a small pilot test, not a measured accuracy for the city.

## Repository layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit web application |
| `bhas_core.py` | Crack detection, measurement, severity score, construction-year detector, fusion |
| `bhas_pipeline.py` | Runs one full audit from the inputs |
| `bhas_gee.py` | Landsat access through Google Earth Engine |
| `bhas_report.py` | PDF report generation |
| `bhas_db.py` | Audit log and escalation |
| `evaluate.py` | Accuracy evaluation on labelled crack datasets |
| `synth.py` | Synthetic test images with exact ground truth |
| `test_bhas.py` | Automated tests |

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

To run the tests and the synthetic benchmark:

```bash
pip install -r requirements-dev.txt
pytest -q test_bhas.py
python evaluate.py --synthetic
```

To evaluate on a labelled dataset (image and mask folders with matching file names):

```bash
python evaluate.py --images path/to/images --masks path/to/masks --size 0
```

### Google Earth Engine

The construction-year feature needs Earth Engine access. Locally or in Colab, run `ee.Authenticate()` once. On Streamlit Cloud, create a service account in a Google Cloud project registered for Earth Engine, give it the Earth Engine Resource Viewer role, and paste its key into the app's secrets in the format of `secrets.toml.example`. Without Earth Engine the app still works; only the construction-year option is switched off.

## Data

- Training data: the merged crack collection by K. Ha ([khanhha/crack_segmentation](https://github.com/khanhha/crack_segmentation)), with every DeepCrack-derived image removed before training.
- Test data: [DeepCrack](https://github.com/yhlleo/DeepCrack) (Liu et al., 2019).
- Satellite data: Landsat Collection 2 surface reflectance, accessed through Google Earth Engine.

The trained U-Net weights (ONNX, 57 MB) are not included in this repository yet.

## Limitations

- The U-Net has not yet been tested on labelled photos of Hyderabad facades.
- The severity constants are engineering defaults and have not yet been calibrated against engineers' grades.
- Construction years can only be estimated for buildings that came up after about 1988, on land that was open before.
- The InSAR branch has been checked only on simulated data.

## Citation

If you use this code, please cite our paper:

A. Suhas, D. Umesh Chandra, K. Sindhu and V. Srinu, "Multi-Modal Building Health Screening Using Crack Segmentation and Landsat-Derived Construction Year: A Case Study in Hyderabad," *International Research Journal of Modernization in Engineering Technology and Science (IRJMETS)*, 2026 (under review).

GitHub's "Cite this repository" button in the sidebar gives the same citation in APA and BibTeX formats.

## Authors

A. Suhas, D. Umesh Chandra, K. Sindhu and V. Srinu (guide), Department of CSE (AI&ML), CMR Technical Campus, Hyderabad.

## License

MIT. See [LICENSE](LICENSE).
