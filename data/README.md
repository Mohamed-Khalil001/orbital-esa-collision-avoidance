# Data

The repository does **not** contain the full ESA files: `train_data.csv` is about 230 MB and GitHub rejects files above 100 MB.
Only a small demo sample is committed.

| Path | In git? | What it is |
|---|---|---|
| `data/sample/demo_events.csv` | yes (2.4 MB) | 150 events from the ESA test file, so the app works right after cloning |
| `data/train_data.csv` or `data/train_data.zip` | no | official training CDMs (162,634 rows, 13,154 events) |
| `data/test_data.csv` | no | official test CDMs (24,484 rows, 2,167 events), inputs only |
| `data/test_data_private.csv` | no | test outcomes released after the competition, needed only by the research notebooks |

## Get the full data

The dataset is public on Zenodo (record 4463683) and on the Kelvins competition page.

```bash
python tools/download_data.py
```

The script downloads the Zenodo archive, checks its MD5 checksum, and copies the files it recognises into `data/`.
If the archive layout differs from what the script expects, it tells you which files it found; copy the three files above into `data/` by hand.

## Rebuild the demo sample

```bash
python tools/make_demo_sample.py --input data/test_data.csv
```

The sample is deterministic (seed 42): about a quarter of the events contain at least one high-risk CDM.

## Source and terms

ESA Advanced Concepts Team and ESA Space Debris Office, *Collision Avoidance Challenge dataset*, Zenodo, 2021,
<https://zenodo.org/records/4463683>. Competition page: <https://kelvins.esa.int/collision-avoidance-challenge/>.
Check the terms on the Zenodo page before redistributing the data in any other form.
