# Financial Transactions Classification

This project aims to classify financial transactions with respect to the kind of establishment in which the operation was performed.

## Repository Organization

The repository was organized to share characteristics from experimental and production environments. Usually, both environments would be contained in separated repositories to isolate the unstable and untested experimental code from the production code. For this assessment test, however, a common structure was created for simplicity.

The structure is organized as follows:
 - `application`: The main directory of the project with code and notebooks.
   - `code`: Functions and classes that are used by notebooks or other applications.
   - `data`: The location of `external`, `raw` and, `processed` datasets used in experiments.
   - `notebooks`: Jupyter notebooks used to analyze and pre-process data and perform experiments.
 - `containers`: Directory with the definition of containers (`application` and `MLflow`).
 - `tests`: Directory with `unit` and `integration` tests.
 

### Dataset

The dataset used for experiments is [Brazilian Credit Card Spending](https://www.kaggle.com/datasets/sufyant/brazilian-real-bank-dataset), with about 5.000 records about credit card transactions from a mid-sized Brazilian bank.


### Notebooks

The notebooks created are:
 - [01 Split Dataset](application/notebooks/01_SplitDataset.ipynb): Notebook to read the raw dataset, identify the split criterion, and split dataset into `training` and `test` sets. That is the first step to avoid leacking information about the main assessment dataset.
 - [02 Explore Dataset](application/notebooks/02_ExploreDataset.ipynb): Notetebook to explore the dataset data, understand the problem and identify potential features for the model.
 - [03.1 Train Base Model](application/notebooks/03.1_TrainBaseModel.ipynb): Notebook to perform experiments for the base model. There is a preliminary assessment based on the validation dataset. Afterward, the final model is created and published on MLFlow server.
 - [03.2 Train Modified Model](application/notebooks/03.2_TrainModifiedModel.ipynb): Notebook to perform experiments for the modified model, which gives higher importance for clothing category. There is also a preliminary assessment based on the validation dataset. Afterward, the final modified model is created and published on MLFlow server.
 - [04 Evaluate Models](application/notebooks/04_EvaluateModels.ipynb): Notebook to retrieve the most recent version of the published models (`base` and `modified`) are evaluated on the `test` set, to provide a final assessment based on unseen data.

## Commands

There is a `Makefile` with commands to make it easier to perform some tasks. The main commands are: 
 - `build`: Build the docker images for the application and MLflow.
 - `up`: Run Jupyter Lab with the application and MLflow.
 - `unit-tests`: Run all unit tests.
 - `integration-tests`: Run all integration tests.
 - `all-tests`: Run all kinds of tests.

The commands above should be used preceded by `make`. For instance:
  > make all-tests

## How to Reproduce the Steps:

To reproduce all procedures performed, follow the steps:
 - Run `make build` to build the *application* and *MLFlow* images.
 - Run `make up` to run the *application* and *MLFlow* containers.
   - From now on, it is possible to access the applications at your browser:
     - Jupyter: http://localhost:8888
     - MLFlow: http://localhost:5002
 - At Jupyter, execute every notebook based on the numeric order.

## Requirements and Dependencies

To be able to run the application, it is necessary to have:
 - docker (20.10.17)
 - docker-compose (1.29.2)

It should not be necessary to have the same version, but both of them were specified for eventual issues found.
 
