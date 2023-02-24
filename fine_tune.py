import importlib
import torch
import yaml
from pathlib import Path
from data_loader import MultiDataLoader 
from train_copy import local_train
from tqdm import tqdm
from from_confusion_matrices import metrics_from_confusion_matrices
import pandas as pd
# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

config_file = Path('config.yaml')
with open(config_file) as file:
  config = yaml.safe_load(file)

model_storage = "../Model Epochs/"
without_vall = 'epoch_39_lco_vall'
fold = 0 #0 Vall 1 Sag 2 ACDC 3 San

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'


#Start Model
def import_class(name):
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def initialize_model(device, state_dict=None):
    Model = import_class(config['model']['arch']['function'])
    if config['model']['arch']['args']:
        model_copy = Model(**config['model']['arch']['args'])
    else:
        model_copy = Model()
    if state_dict != None:
        model_copy.load_state_dict(state_dict)
    else:
        model_copy.load_state_dict(model.state_dict())
    model_copy.to(device)
    return model_copy

def load_model():
    checkpoint = torch.load(model_storage + without_vall)['state_dict']
    model = initialize_model(device, checkpoint)
    return model

def load_data():
    data = {} 
    fold_splits = []
    for centre in config['data']['centres']:
        worker = centre # sy.VirtualWorker(hook, id=centre) # remove comment when they fix the bug
        dl = MultiDataLoader([centre]) # replace with two way split instead
        data[centre]=(worker, dl)
    for i, centre in enumerate(data.keys()): # will be used to index over centers 
        test_indices = centre
        train_indices = [x for j,x in enumerate(data.keys()) if j!=i] # The validation comes out of the centers we train on
        fold_splits.append((train_indices, test_indices))
    return data, fold_splits

def set_crit_opt(model):
    #Criterion & Optimiser
    Criterion = import_class(config['hyperparameters']['criterion']) #torch.nn.BCELoss
    criterion = Criterion()
    Opt = import_class(config['hyperparameters']['optimizer'])
    learning_rate = float(config['hyperparameters']['lr'])
    if config['model']['arch']['args']['early_layers_learning_rate']: # if zero then it's freeze
        # Low shallow learning rates instead of freezing
        low_lr_list = []
        high_lr_list = []
        for name,param in model.named_parameters():
            if 'fc' not in name: # Only works for ResNets
                low_lr_list.append(param)
            else:
                high_lr_list.append(param)
        print(f"Initializing optimizer with learning rates {config['model']['arch']['args']['early_layers_learning_rate']} for the early and {learning_rate} for the final layers")
        opt = Opt([
            {'params': low_lr_list, 'lr': float(config['model']['arch']['args']['early_layers_learning_rate'])},
            {'params': high_lr_list, 'lr': learning_rate}
        ], lr=learning_rate)
    else:
        print(f"Layer learning mode set to frozen")
        opt = Opt(model.parameters(), lr=learning_rate)
    return criterion, opt

def fine_tune(model,data,opt,criterion,fold_splits):
    #Test
    dataset = data
    fold_splits = fold_splits[fold]
    if not config['federated']['type'] in ['CIIL', 'SWA']:
        model.to('cpu')
    if config['cvtype'] == 'LCO':
        test_index = fold_splits[1]
        print(test_index)
        test_set = {test_index: dataset[test_index]}
    return local_train(test_set, model, opt, criterion, device, fold=fold, test=False)

def test(data,model,opt,criterion,fold_splits):
    #Test
    dataset = data
    fold_splits = fold_splits[fold]
    if config['cvtype'] == 'LCO':
        test_index = fold_splits[1]
        print(test_index)
        test_set = {test_index: dataset[test_index]}
    return local_train(test_set, model, opt, criterion, device, fold=fold, test=True)


if __name__=='__main__':
    model = load_model()
    data, fold_splits = load_data()
    criterion, opt = set_crit_opt(model)
    model_new, _, _, _, _, _, _, _ = fine_tune(model,data,opt,criterion,fold_splits)
    model_new = model_new[0]
    _, _, _, test_predictions, _, _, test_accuracy_avg, test_confusion_matrix = test(data,model_new,opt,criterion,fold_splits)
    print(test_predictions)
    print(test_confusion_matrix)
    _, _, _, test_predictions, _, _, test_accuracy_avg, test_confusion_matrix = test(data,model,opt,criterion,fold_splits)
    print(test_predictions)
    print(test_confusion_matrix)
    #acc_list, precision, recall, f1_scores = metrics_from_confusion_matrices(test_confusion_matrix)

