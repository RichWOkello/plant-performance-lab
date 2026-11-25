# Plant Performance Lab

I picked this project because it brings together control engineering and data science. A power plant is an interesting place to study regression: a few sensor readings can tell us quite a lot about expected generation, but they cannot tell us everything. I wanted to build something that goes beyond a notebook and gives an operator a way to explore the readings and try different conditions.

## Why this dataset?

The UCI Combined Cycle Power Plant dataset has 9,568 hourly averaged observations from 2006 to 2011, collected while one plant operated at full load. I use temperature, exhaust vacuum, ambient pressure and relative humidity to estimate net electrical output in MW. Exhaust vacuum is a process measurement, not simply a weather variable.

## What I found

<!-- RESULTS START -->

I worked with 9,568 records. Average output was 454.37 MW, and the temperature-output correlation was -0.948. The linear baseline had a test RMSE of 4.50 MW. After selecting and tuning Random Forest using training cross-validation, the final test RMSE was 3.24 MW, MAE was 2.32 MW and R² was 0.9639. The largest permutation importance belonged to AT, with a 17.86 MW increase in error when shuffled. That is a measure of model dependence, not a causal effect.

<!-- RESULTS END -->

## What I had to watch out for

The direct dataset download was unavailable during the initial build. I used the original UCI workbook supplied with the project instead, and saved the first of its five shuffled sheets as a CSV. I did not stack the sheets because they are permutations of the same observations. Another concern was leakage. I put the scaler inside each model pipeline and used training cross-validation to choose the model instead of picking the one with the best test score. Exact duplicate rows are reported rather than silently removed, since the dataset has no timestamps to establish whether they are repeated measurements. The random split placed 14 test observations with an exact match in training, so the reported performance may be optimistic. A grouped or time-based validation would be a useful follow-up.

## Presentation

The [six-slide presentation](reports/Plant_Performance_Lab_Presentation.pptx) explains the project without assuming a technical background. It includes the test results and speaker notes. I use Google Slides for presenting it.

To set it up, download the PPTX, open [Google Slides](https://slides.google.com), use the file picker to upload the presentation, and choose **File > Save as Google Slides**. Check the layout and the final slide's Open dashboard link after conversion. Copy the sharing link. In Git Bash, create a local settings file if it does not already exist:

```bash
cp -n .env.example .env
```

Open `.env` and paste the link after `GOOGLE_SLIDES_URL=`. Restart Streamlit. The sidebar will now open your actual Google Slides presentation. Until then, it offers a download and a link to the Google Slides homepage. The online copy is not part of this repository; it belongs to your Google account. The dashboard link points to `http://localhost:8501`, so start Streamlit on the same computer before the live demonstration.

## Dashboard

The sidebar has a download button for the [analysis notebook](notebooks/capstone.ipynb), a download button for the presentation, and a link to Google Slides. The notebook and the final slide link back to the running dashboard. I kept these in one folder so the project is easy to share and present.

Capture the Overview and Predict pages once the application is running and save them in `docs/` as `dashboard.png` and `prediction.png`. These are intended screenshot paths, not existing images.

## Running it in VS Code (Windows, Git Bash)

Open the extracted project folder in VS Code and select Git Bash as your terminal. Python 3.12 or 3.13 is recommended. The included model was trained with Python 3.13 and scikit-learn 1.8.0.

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.data
python -m jupyter notebook notebooks/capstone.ipynb
```

The supplied notebook has already been executed on the original data, and the saved model and evaluation artifacts are included. The scikit-learn version is pinned to match the fitted model. Run all notebook cells in order if you want to reproduce the experiment. The original UCI CSV is bundled, so the first data fetch is not required. Once the CSV and model artifacts exist, the local analysis and predictions can run without internet. The notebook also updates the findings in this README. Alternatively, execute the notebook from Git Bash:

```bash
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks/capstone.ipynb
```

Then run the dashboard from the project root:

```bash
streamlit run app/dashboard.py
```

Open http://localhost:8501. The links in the notebook and presentation use this address. Keep the Streamlit terminal running during the presentation. Use Ctrl+C to stop the server.

## Chat assistant

The default assistant uses Ollama on my own computer. I chose `qwen2.5:3b` because it is a relatively small local model. It receives the actual plant statistics and model results, and prediction requests are passed to the fitted regression pipeline. No API key is required for Ollama. The language model can still make mistakes, so check its answers against the dashboard.

In Git Bash, check that the model is installed:

```bash
ollama list
```

If it is missing, run `ollama pull qwen2.5:3b`. Start the Ollama desktop application, or run `ollama serve` in a separate terminal if the service is not running. Select Local Ollama in the Ask the Data page. The connection is checked against the local Ollama service at `http://127.0.0.1:11434`.

The offline reference assistant also works without Ollama. Claude is an optional third choice. To configure it, copy `.env.example` to `.env` and enter your own `ANTHROPIC_API_KEY` from https://console.anthropic.com/. Claude sends the context and conversation to Anthropic. Local Ollama does not require that cloud service. Never commit `.env`.

## Project layout

```text
notebooks/capstone.ipynb   Analysis and training
src/                     Data, models, reporting and chatbot helpers
app/dashboard.py         Streamlit application
data/                    Local UCI CSV (included)
models/                  Trained model and evaluation files
reports/figures/         Notebook plots
.streamlit/config.toml   Dashboard theme
docs/                    Dashboard screenshots
```

## Putting it on GitHub

Create an empty repository called `plant-performance-lab` on GitHub. In Git Bash, from this project folder, run:

```bash
git init
git add .
git commit -m "Add plant power prediction capstone"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/plant-performance-lab.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username. The CSV, executed notebook, model and figures are included. The virtual environment and local settings file are excluded. GitHub can render the notebook, but it does not run Streamlit or Ollama. Run those on your computer for the live demo.

## Limitations and next steps

This is data from one plant at full load. I would not assume the model will work at another plant or during startup, shutdown or part-load operation. There are no timestamps, so a random split cannot establish future forecasting performance or account for temporal dependence. The inputs must be available at prediction time, including exhaust vacuum. The temperature scenario holds the other variables fixed, which is not necessarily physically realistic. I would collect time-stamped operating data, check performance across operating regimes, test on another plant and add prediction intervals before considering operational use. The dashboard is not a safety or control system.

## Dataset citation

Tufekci, P. & Kaya, H. (2014). Combined Cycle Power Plant. UCI Machine Learning Repository. https://doi.org/10.24432/C5002N (CC BY 4.0).
