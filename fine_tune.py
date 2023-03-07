import importlib
import torch
import yaml
import numpy as np
from pathlib import Path
from data_loader import MultiDataLoader 
from train_copy import local_train,run_epoch
#from train_copy import run_epoch
from tqdm import tqdm
import pandas as pd
# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


config_file = Path('config_personalisation.yaml')
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
    if config['train_private']: 
        Model = import_class(config['model']['arch']['function'])
        model = Model(**config['model']['arch']['args'])
        print("Training without Global Model")
    else: 
        checkpoint = torch.load(model_storage + without_vall)['state_dict']
        model = initialize_model(device, checkpoint)
    return model

def load_data():
    data = {} 
    fold_splits = []
    for centre in config['data']['centres']:
        worker = centre 
        dl = MultiDataLoader([centre])
        data[centre]=(worker, dl)
    for i, centre in enumerate(data.keys()):
        test_indices = centre
        train_indices = [x for j,x in enumerate(data.keys()) if j!=i] 
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



def log_results(epoch,train_loss,val_loss,val_accuracy,confusion_matrix, best_model_results,best_model,early_stop_counter):
    if train_loss != None: # Last epoch for federated is None
        print(f'''
        ========================================================
        Epoch {epoch} finished
        Training loss: {train_loss:0.3f}
        Validation loss: {val_loss:0.3f}, accuracy score:
        {val_accuracy:0.3f}
        ========================================================''')
    print("Confusion Matrix: ")
    print(confusion_matrix)

    if val_loss < best_model_results['val_loss']:
        best_model_results['epoch'] = epoch
        best_model_results['val_loss']= val_loss
        best_model_results['train_loss'] = train_loss
        best_model_results['val_accuracy'] = val_accuracy
        best_model_results['confusion_matrix'] = confusion_matrix
        best_model = initialize_model(device, model.state_dict())
    else:
        early_stop_counter+=1
    return best_model_results, best_model, early_stop_counter
    


def save_best_model(best_model, best_epoch, results_path):
    torch.save({'epoch': best_epoch,
                'state_dict': best_model.cpu().state_dict(),
                    },results_path.joinpath(f'epoch_{best_epoch}'))


#Fine-tunes the model on a center
def fine_tune_local_train(model,data,opt,criterion,fold_splits):
    fold_splits = fold_splits[fold]
    test_index = fold_splits[1]
    test_set = {test_index: data[test_index]}
    return local_train(test_set, model, opt, criterion, device, fold=fold, test=False)


#Tests model on a full center
def test_on_center(data,model,opt,criterion,fold_splits):
    fold_splits = fold_splits[fold]
    test_index = fold_splits[1]
    test_set = {test_index: data[test_index]}
    return local_train(test_set, model, opt, criterion, device, fold=fold, test=True)


if __name__=='__main__':
    model = load_model()
    data, fold_splits = load_data()
    criterion, opt = set_crit_opt(model)
    model_new = model
    opt_new = opt
    num_epochs = config['num_epochs']
    early_stop_counter, best_loss = 0, 10000
    best_model_results = dict()
    best_model_results['val_loss'] = best_loss
    best_model = model

    #_, _, _, test_predictions, _, _, test_accuracy_avg, test_confusion_matrix = test_on_center(data,model_new,opt_new,criterion,fold_splits) #Tests on a full center

   for epoch in range(0,num_epochs):
        model_new,opt_new,_, _, train_loss, val_loss, val_accuracy, confusion_m = fine_tune_local_train(model_new,data,opt_new,criterion,fold_splits)
        model_new = model_new[0]
        opt_new = opt_new[0]
        best_model_results, best_model,early_stop_counter = log_results(epoch,train_loss,val_loss,val_accuracy,confusion_m,best_model_results,best_model,early_stop_counter)
        print(best_model_results)
        if early_stop_counter == config['early_stop_checkpoint']: 
            print("Reached early stop checkpoint")
            break
    print("The Best model is")
    print(best_model_results)
    #save_best_model(best_model, best_model_results['epoch'], config['output_model_storage'])
        

