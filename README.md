
# McKinsey Analytics Datahack 2018

This is my code for Datahack 2018.

It employs various modular classifiers including XGBoost, AdaBoost, SVM, 
simple neural nets (MLPs), naive Bayes, and QDA.  These are trained individually
and then combined into a weighted soft-voting ensemble.

Model params and ensemble weighting has been tuned to the dataset via grid
search and some manual adjustment.

XGBoost is doing most of the work here (as usual), so the weights on the other
models are typically much smaller.  The final submission ensemble consisted of
(approximately):

    0.6 * XGBoost + 0.3 * KNN + 0.06 * QDA + 0.03 * AdaBoost


## Setup and running

Using conda, set up an env with Python 3.6:

    conda create -n datahack python=3.6

Then install the required packages:

    pip install -r requirements.txt

The submission can then be generated using:

    python datahack/main.py

Submissions using other prediction models can be generated with:

    python datahack/main.py [MODEL_ID]

where the model ID is one of `ensemble`, `xgb`, `svm`, `mlp`, `gnb`, `qda`, etc.


## Task specification

See `TASK.md`. 