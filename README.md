# Financial Transactions Classification

The goal this project is classifying financial transactions with respect to the kind of establishment in which the operation was performed.

## Repository Organization

The repository was organized to share characteristics from experimental and production environments. Usually, both environments would be organized in separated repositories, to isolate the unstable and unstested experimental code from the production code. For this assessment test, however, a common structure was created for simplicity.

The structure is organized as follows:
 - `application`: The main directory of the project with code and notebooks.
   - `code`: Functions and classes that are used by notebooks or other applications.
   - `data`: The location of `external`, `raw` and `processed` datasets used on experiments.
   - `notebooks`: Jupyter notebooks used to analyze and pre-process data and perform experiments.
 - `containers`: Directory with the definition of containers (`application` and `MLflow`).
 - `tests`: Directory with `unit` and `integration` tests.
 

## Dataset

The dataset used for experiments is [Brazilian Credit Card Spending](https://www.kaggle.com/datasets/sufyant/brazilian-real-bank-dataset), with about 5.000 records about credit card transactions from a mid-sized Brazilian bank.

## Comands

There is a `Makefile` with commands to make it easier to perform some tasks. The main commands are: 
 - `build`: Build the docker images for the application and MLflow.
 - `up`: Run Jupyter Lab with the application and MLflow.
 - `unit-tests`: Run all unit tests.
 - `integration-tests`: Run all integration tests.
 - `all-tests`: Run all kinds of tests.

The commands above should be used preceeded by `make`. For instance:
  > make all-tests