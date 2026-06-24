Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

# Environment Set up (pip or conda)
* Option 1: use the supplied file `environment.yml` to create a new environment with conda
* Option 2: use the supplied file `requirements.txt` to create a new environment with pip
    
## Repositories
* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

# Data
* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model
* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation
*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.

---

# Project Submission

**GitHub repository:** https://github.com/Sahara65/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

## How to run this project

1. Create the environment and install dependencies:
   ```
   conda env create -f environment.yml      # option 1 (conda)
   conda activate fastapi
   # or, with pip:
   python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
   pip install -r requirements.txt
   ```
2. Train the model and generate slice metrics (creates `model/model.pkl`,
   `model/encoder.pkl`, and `slice_output.txt`):
   ```
   python train_model.py
   ```
3. Run the unit tests:
   ```
   pytest test_ml.py -v
   ```
4. Start the API, then exercise it locally from a second terminal:
   ```
   uvicorn main:app --reload          # terminal 1
   python local_api.py                # terminal 2
   ```
   Expected `local_api.py` output:
   ```
   Status Code: 200
   Result: Welcome to the Census Income Prediction API!
   Status Code: 200
   Result: <=50K
   ```

## Repository contents (graded artifacts)

* `ml/model.py`, `train_model.py` — completed training pipeline.
* `test_ml.py` — three unit tests on the ML code.
* `main.py` — FastAPI app with a GET (welcome) and a POST (inference).
* `local_api.py` — GET + POST requests against the running API.
* `model/model.pkl`, `model/encoder.pkl` — saved model and encoder artifacts.
* `model_card.md` — completed model card with metrics.
* `slice_output.txt` — performance on data slices.
* `screenshots/local_api.png`, `screenshots/continuous_integration.png`,
  `screenshots/unit_test.png` — required screenshots.
* `.github/workflows/manual.yml` — CI running `flake8` and `pytest` on push.
