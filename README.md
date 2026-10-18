# ♻️ Arabic Smart Waste Sorter (فرز النفايات الذكي)

Streamlit app: webcam or uploaded photo → predicted item + recommended bin in Arabic → spoken Arabic result
(browser speech). Local inference on CPU (MobileNetV3-Small), no paid services, no API keys.

## IMPORTANT: what is and is NOT included / tested

| Item | Status |
|---|---|
| Source code, config, scripts, docs | Included |
| Trained model (`models/waste_model.pt`) | **NOT included.** No internet in the build environment, so no dataset was downloaded and nothing was trained. You create it in steps 2-3 below. |
| Test accuracy / confusion matrix | **None yet.** `train.py` produces them on your machine (`models/classification_report.txt`, `confusion_matrix.png`, `metrics.json`). Do not quote any accuracy before you have these files. |
| Tested here | Decision logic (uncertain/unknown), Arabic message building, config validation, stratified split (8 unit tests pass), Python syntax of all files |
| NOT tested here | Streamlit UI, webcam, browser speech, PyTorch training/inference, dataset download, Windows `.bat` scripts. Please do a full dry run before the conference. |

## Windows setup (needs internet once)

1. Install **Python 3.11 or 3.12** from python.org (tick "Add Python to PATH").
2. Double-click `setup_windows.bat` (creates `.venv`, installs CPU PyTorch + requirements).
3. Double-click `prepare_data.bat` – downloads the public TrashNet dataset (cardboard, glass, metal, paper, plastic, trash) from Hugging Face into `data/trashnet/`. If it fails, download a "garbage classification" dataset manually and arrange it as `data/<name>/<class>/*.jpg`.
4. **Add your real objects** (strongly recommended, see below), then double-click `train_model.bat`. On CPU expect roughly 10-40 minutes depending on the laptop. For a quick trial: `train_model.bat --max-per-class 150 --epochs-head 2 --epochs-ft 3`.
5. Read the printed test accuracy and per-class precision/recall. Open `models/confusion_matrix.png`.
6. Double-click `run_app.bat` – the app opens in the browser (use **Microsoft Edge or Chrome**).

Manual commands (inside the project folder, PowerShell):
```
py -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements-data.txt
python prepare_dataset.py
python train.py
streamlit run app.py
```

## Why you must add your own photos

TrashNet has no chips bags, paper cups, laban containers or notebooks, and its pictures are taken on a plain background.
A model trained only on it will mislabel your demo objects. Use the app's **«علّم النموذج»** tab (or copy photos into `data/custom/<class>/`) with **≥ 15–30 photos per class** in different angles, lighting and hands holding the item:

`water_bottle, juice_bottle, laban_container, chips_bag, paper_cup, book_notebook, paper/cardboard, metal`, plus **`background`** (empty table, empty hand, room) so the app can say "no item". Large and small paper cups share one class (size doesn't change the material); split them in `classes.json` if your school's rules differ.
Then re-run `train_model.bat`. Inference never retrains; the app just loads `models/waste_model.pt`.

Caveat on evaluation: random splits of photos of the *same* objects (and TrashNet's near-duplicate shots) give optimistic test accuracy. For an honest number, photograph some items on a different day/place and test only those.

## How the app avoids forcing an answer

`decide()` in `waste_core.py` returns **"غير متأكد"** when the top probability is below the confidence threshold or too close to the runner-up (sliders in the sidebar; defaults in `classes.json`). Paper vs cardboard (same bin) are treated as one group. Softmax confidence is *not* calibrated probability – the UI says so.

## Local recycling rules (edit `classes.json`)

Bins and advice are **default assumptions, not verified rules**. Confirm with your school supervisor and edit `bin` / `advice` per class. The model sees shape and texture, not material, so for laban containers and similar items the app asks the user to check the packaging symbol.

## Arabic speech

Uses the browser's `speechSynthesis`. If no Arabic voice exists you'll see a message: use Edge/Chrome, and add an Arabic voice in Windows (Settings → Time & language → Speech / Language & region). Automatic speech may be blocked by the browser until the page has been clicked; the 🔊 / 🔁 buttons always work. Test sound and voice on the demo laptop beforehand.

## Files

`app.py` UI · `waste_core.py` logic · `model_utils.py` model · `train.py` fine-tune + evaluation · `prepare_dataset.py` dataset download · `classes.json` Arabic names/bins/thresholds · `tests/test_core.py` (`python -m unittest discover -s tests`).

## Troubleshooting

- `streamlit` not found → run `setup_windows.bat` first. · Camera blank → allow camera permission in the browser. · "no trained model" → steps 3–4.
- If a newer Streamlit changes `components.html`, pin `streamlit==1.40.*` in requirements.txt.
