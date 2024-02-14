# NebulaPredictors

## Description
NebulaPredictors is a project focused on predicting pips in the euro/dolar (EUR/USD) pair to assist a trading bot in making real-world trading decisions. The project explores various predictive models and techniques to forecast the movement of the EUR/USD pair.

## Goals
- Develop accurate and reliable predictive models for forex pips using machine learning and statistical analysis.
- Train a trading bot to utilize the predictions for informed decision-making in real-time trading scenarios.
- Continuously analyze and improve the models to enhance performance and predictive capabilities.

## Folder Structure
- `models`: Contains the machine learning models utilized for prediction of every trial.
- `notebooks`: Contains Jupyter notebooks used for data exploration, model training, evaluation, and experimentation in every trial. Aditionally contain .py files with 
               flask.
- `enviroment.yml`: File specifying the Anaconda environment dependencies required for the project.

## Usage
- Clone this repository to your local machine.
- Set up the required environment using the specifications in `enviroment.yml`.
- Explore the notebooks in the `notebooks` directory for data analysis, model training, and evaluation.
- Use the trained models in the `models` directory to make predictions for the EUR/USD pair within your trading platform.

## FLASK
Inside conda environment:
- flask --app C:/Users/david/OneDrive/Documents/NebulaPredictors/notebooks/deaf_reload_flask_01272024 run
- flask --app C:/Users/david/OneDrive/Documents/NebulaPredictors/notebooks/trial_2/deaf_reload_flask_signed run


## Trials Descriptions

### trial 1: 
- The notebook for this trial is: SuperLearnerModel_trial1_signed_p95.ipynb located in notebook folder with name trial_1
- 4 models with diferent time periods (see images folder:(trial_1image1.png))
- 2 p95 trained models too
- The metamodel is a Lightgbm regressor
- databases are located in data folder too

### trial 2: 
- The notebook for this trial is: SuperLearnerModel_trial2.ipynb located in notebook folder with name trial_2
- 4 models with diferent time periods (see images folder:(trial_2image1.png))
- The metamodel is a Lightgbm regressor
- databases are located in data folder too

## Simulations Outcomes
- Folder simulations has the main outcomes for diferent trials combinations, and important notes!

## Contribution
Contributions to this project are welcome. Feel free to raise issues, suggest improvements, or submit pull requests.

