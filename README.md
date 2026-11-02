# Blood Cell YOLO26

Fine-tune a pretrained YOLO26 object detector on the supplied TXL-PBC blood-smear
dataset, run a held-out test evaluation, and review the dataset and model reports
in a local HTML dashboard.

The project expects the original dataset at `TXL-PBC/`. It uses YOLO detection
labels with class IDs `0 = WBC`, `1 = RBC`, and `2 = Platelets`.

## Dataset report

Create a dataset health report and validate the images and labels:

```bash
python3 scripts/analyze_dataset.py
```

The report is written to `reports/dataset_report.json`.

## Kaggle GPU training

The training notebook is configured for a private Kaggle dataset and a GPU
accelerator. The API token is read from `.env` only by the local submission
script; it is not copied into the notebook, dataset, or Git history.

1. Install the project dependencies:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

2. Keep the provided `KAGGLE_API_TOKEN` in the root `.env`. Add
   `KAGGLE_USERNAME=your-kaggle-username` if the account name is not discovered
   automatically.
3. Submit the private dataset and notebook:

   ```bash
   python3 scripts/submit_kaggle.py
   ```

   Add `--reuse-dataset` when resubmitting unchanged data and training source.
   The submission starts a Kaggle Notebook run with GPU enabled. The notebook
   page provides live epoch output. To stream Kaggle logs into the local
   dashboard while training, run this in another terminal:

   ```bash
   python3 scripts/submit_kaggle.py --watch
   ```

   The default configuration fine-tunes
   `yolo26n.pt` for up to 100 epochs at 640px with early stopping (patience 20).
4. When Kaggle reports the run as complete, download its output:

   ```bash
   python3 scripts/submit_kaggle.py --download
   ```

   Reports and weights are downloaded under `outputs/`; large model weights and
   training artifacts are excluded from Git by default.

Kaggle may prompt you to accept the relevant model/dataset terms or enable
internet access for pretrained weight and package downloads. The dataset source
license is not included in the provided files, so the Kaggle dataset upload is
private and marked with an unknown license rather than assigning a license.

## Dashboard

Serve the project root and open
[dashboard/index.html](dashboard/index.html) in a browser:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000/dashboard/`. The page refreshes its JSON reports
every 10 seconds. While training is active, use the Kaggle notebook page for
live epoch logs; after the run, download the reports to populate the local
performance charts and class metrics.

## Outputs

- `reports/dataset_report.json`: split sizes, class distribution, image sizes,
  and annotation checks.
- `outputs/reports/training_progress.json`: training state and current epoch.
- `outputs/reports/model_performance.json`: held-out test metrics and per-class
  precision, recall, and AP.
- `outputs/runs/yolo26n/weights/best.pt`: best validation checkpoint.
- `outputs/runs/test/`: test plots and confusion matrix from Ultralytics.

This project is an educational research aid, not a medical device or clinical
diagnostic system. Predictions should not be used to make clinical decisions.
