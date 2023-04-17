# HCM-Personalised-FederatedLearning-Predictor
A classifier for Hypertrophic Cardiomyopathy (HCM) which uses personalised federated learning. Program based off https://github.com/Linardos/federated-HCM-diagnosis .

Files & directories written by Linardos:
- misc
- models
- variables
- config.yaml
- data_loader.py
- from_confusion_matrices.py
- train.py

data_loader.py and config.yaml are altered slightly by me to tune the amount of synthetic data used in the personalisation layer.
Any file paths have been changed to suit my local system.

Files & directiories written by me:
- config_personalisation.yml
- environment.yml
- fine_tune.py


Install conda
Run 'conda env create -f environment.yml' to replicate virtual environment
