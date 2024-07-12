ANALYSIS OF THE TERRORIST THREAT IN SAHEL
==============================

Presentation
------------
We have developed a tool to predict events involving terrorist groups in the Sahel over the next 6 months.
We also want to study the impact of Russian PMCs (Wagner, Redut, Africa Corps, etc.) in the fight against terrorism since their arrival in these three Sahelian countries.

Based on data (from the last 3 years) from the NGO [ACLED](https://acleddata.com/) on events of political violence in the Sahel over the past three years, we want to focus on terrorist attacks in Mali, Burkina Faso and Niger, three countries with a common border that is the scene of terrorist-related violence.


Methodology
------------
After an exploratory study of the data, we set out to solve our problem using a non-linear regression machine learning model.

We first built a `past_data` dataframe containing historical datas (from the last 3 years), then we created a `future_data` dataframe which will be used to make our predictions for the next 6 months. We used a method based on calculating averages from the datas in `past_data` to fill in `future_data`.

We then trained several non-linear regression models, compared their performance metrics and concluded that the best performing model was a Random Forest model with LDA-type dimension reduction.

Finally, we trained this model on `past_data`, then calculate predictions on the same dataset in order to reuse this model on `future_data`.


Results
------------
During the data-mining phase, we generated a number of graphs to capture the subtleties of our problem and the general context.

Following the predictions calculated by the model on `future_data`, we also generated several graphs that visualize the values observed over the last 3 years as well as the predicted values over the next 6 months concerning events linked to terrorist organizations.

We have also been able to generate a map that displays month-by-month, for each target country, by sub-region, the predictions of events linked to terrorist organizations (with an associated color code according to the number).


How to locally execute this project
------------
You need to first download ACLED datas from their website (an account is needed), then rename in `config.yaml` the `paths.raw_data` property with the name of the file you have downloaded.

Make sure that before compiling each files, you set up the PYTHONPATH var env:\ 
`export PYTHONPATH="absolute_path/to/this_repository"`

Then to compile each file, be sure to be at the root of this repository and use this command (e.g.):\ 
`python3 -m src.features.data_preprocessing`\ 
or `python -m src.features.data_preprocessing`

Then you need to compile files in this order:\ 
files in`/src/features` path\ 
files in the `/src/models` path\ 

A trained model will be generated in `/models` and datas (csv format) in `/data/processed`.

If you want to generate the figures and the map, you have to compile files in `/src/visualization`.\ 
The figures will be generated and saved in `/reports/figures`.\ 
The map will be opened in your default web browser using your localhost on a dedicated port.


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained models
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that we made during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── data_preprocessing.py
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── predict_train_model.py
    │   │ 
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize_dataframe.py
    │   │   └── visualize_map.py
    │   │   └── visualize_predictions.py

--------
