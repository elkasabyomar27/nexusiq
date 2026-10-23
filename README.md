# NexusIQ — League of Legends Draft Prediction

NexusIQ is an interactive machine learning application that predicts the likely winning side of a League of Legends match based on the selected champions and summoner spells for both teams.

## Features

- Interactive Blue Side and Red Side draft builder
- Champion selection and summoner spell configuration
- Machine learning-based match outcome predictions
- Draft insights and lane matchup analysis
- Probability-based prediction results
- Input validation to prevent duplicate champion picks

## Tech Stack

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Joblib

## Model Evaluation

The selected Logistic Regression model was evaluated on a held-out test set of 42,399 historical matches.

| Metric | Result |
|---|---:|
| Accuracy | 53.95% |
| ROC-AUC | 0.556 |
| Average Precision | 0.566 |

## Important Limitations

The model uses historical League of Legends data from 2016–2017. Its predictions do not represent current champion strength or current-patch win probabilities. Performance on modern matches has not been validated, and the model should be treated as an experimental demonstration rather than a competitive draft recommendation system.

## Running Locally

1. Clone this repository.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run app.py
   ```

## Project Purpose

NexusIQ demonstrates an end-to-end machine learning workflow, from historical match data and model evaluation to interactive inference through a user-facing application.
