import importlib
import torch
import yaml
import numpy as np
from pathlib import Path
from data_loader import MultiDataLoader 
from train_copy import local_train,run_epoch
from from_confusion_matrices import metrics_from_confusion_matrices
from tqdm import tqdm
import pandas as pd
import time
import warnings
# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


CONFIG_FILE = Path('CONFIG_personalisation.yaml')
with open(CONFIG_FILE) as file:
  CONFIG = yaml.safe_load(file)
ALTERNATE_CONFIG_FILE = Path('config.yaml')
with open(ALTERNATE_CONFIG_FILE) as file:
  ALTERNATE_CONFIG = yaml.safe_load(file)
NUM_TRANSFORMATIONS = ALTERNATE_CONFIG['data']['num_transformations']

MODEL_STORAGE = CONFIG['model_storage']
RESULTS_STORAGE = Path(CONFIG['output_model_storage'])
GLOBAL_MODEL = CONFIG['global_model']
FOLD = CONFIG['center_fold']
CENTER = CONFIG['data']['centres'][FOLD]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_MODELS = CONFIG['save_models']
TRAIN_PRIVATE = CONFIG['train_private']

NUM_FOLDS = CONFIG['num_folds']
NUM_EPOCHS = CONFIG['num_epochs']


#Start Model
def import_class(name):
    module_name, class_name = name.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


#Loads model weights from pretrained model
def initialize_model(state_dict=None):
    Model = import_class(CONFIG['model']['arch']['function'])
    if CONFIG['model']['arch']['args']:
        model_copy = Model(**CONFIG['model']['arch']['args'])
    else:
        model_copy = Model()
    if state_dict != None:
        model_copy.load_state_dict(state_dict)
    else:
        model_copy.load_state_dict(model_copy.state_dict())
    model_copy.to(DEVICE)
    return model_copy

#Loads the pretrained the model or initialises a new one
def load_model():
    if TRAIN_PRIVATE: 
        model = initialize_model(None)
        print("Training without Global Model")
    else: 
        checkpoint = torch.load(MODEL_STORAGE + GLOBAL_MODEL)['state_dict']
        model = initialize_model(checkpoint)
    return model


#Uses data_loader.py to load datasets into objects seperated by center
def load_data():
    data = {} 
    fold_splits = []
    for centre in CONFIG['data']['centres']:
        worker = centre 
        dl = MultiDataLoader([centre])
        data[centre]=(worker, dl)
    for i, centre in enumerate(data.keys()):
        test_indices = centre
        train_indices = [x for j,x in enumerate(data.keys()) if j!=i] 
        fold_splits.append((train_indices, test_indices))
    return data, fold_splits


#Sets the criterion and optimiser for the model
def set_crit_opt(model):
    #Criterion & Optimiser
    Criterion = import_class(CONFIG['hyperparameters']['criterion']) #torch.nn.BCELoss
    criterion = Criterion()
    Opt = import_class(CONFIG['hyperparameters']['optimizer'])
    learning_rate = float(CONFIG['hyperparameters']['lr'])
    if CONFIG['model']['arch']['args']['early_layers_learning_rate']: # if zero then it's freeze
        # Low shallow learning rates instead of freezing
        low_lr_list = []
        high_lr_list = []
        for name,param in model.named_parameters():
            if 'fc' not in name: # Only works for ResNets
                low_lr_list.append(param)
            else:
                high_lr_list.append(param)
        print(f"initializing optimizer with learning rates {CONFIG['model']['arch']['args']['early_layers_learning_rate']} for the early and {learning_rate} for the final layers")
        opt = Opt([
            {'params': low_lr_list, 'lr': float(CONFIG['model']['arch']['args']['early_layers_learning_rate'])},
            {'params': high_lr_list, 'lr': learning_rate}
        ], lr=learning_rate)
    else:
        print(f"Layer learning mode set to frozen")
        opt = Opt(model.parameters(), lr=learning_rate)
    return criterion, opt


#Prints the result metrics every epoch, updates the optimal model and updates early stop counter
def print_results(epoch,train_loss,val_loss,val_accuracy,confusion_matrix, best_model_results,best_model,early_stop_counter,model_m):
    if train_loss != None: # Last epoch for federated is None
        print(f'''
        ========================================================
        Epoch {epoch} finished
        Training loss: {train_loss:0.3f}
        Validation loss: {val_loss:0.3f}, accuracy score:
        {val_accuracy:0.3f}
        ========================================================''')
    #print("Confusion Matrix: ")
    #print(confusion_matrix)

    if val_loss < best_model_results['val_loss']: #and train_loss < CONFIG['train_loss_threshold']: #and val_loss > 0.09:
        best_model_results['epoch'] = epoch
        best_model_results['val_loss']= val_loss
        best_model_results['train_loss'] = train_loss
        best_model_results['val_accuracy'] = val_accuracy
        best_model_results['confusion_matrix'] = confusion_matrix
        best_model = initialize_model(model_m.state_dict())
        print("Best model updated")
    else:
        early_stop_counter+=1
    return best_model_results, best_model, early_stop_counter
    

#Saves the model produced by the epoch with the best generalisation
def save_best_model(best_model, best_epoch,fold,center,cross_val):
    torch.save({'epoch': best_epoch,
                'center': center,
                'cross_val': cross_val,
                'state_dict': best_model.cpu().state_dict(),
                    },RESULTS_STORAGE.joinpath(f'cross_val_{cross_val}_center_{center}_fold_{fold}'))

def calculate_time(elapsed_time):
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    return "Train Time: {:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def fine_tune(test_fold,model,data,opt,criterion):
    dl = data[CENTER][1]

    early_stop_counter, best_loss = 0, 10000
    best_model_results = dict()
    best_model_results['val_loss'] = best_loss
    best_model = model
    best_opt = opt
    training_loader, validation_loader, test_loader = dl.load(fold_index=test_fold) 
    global_test_confusion_matrix = []

    if (CONFIG['train_private']!=True):
        #Accuracy to beat
        print("Testing Global Model. Fold: " + str(test_fold))
        _, _, global_test_predictions, _, _, global_test_confusion_matrix = run_epoch(FOLD,test_loader, model, opt, criterion, DEVICE, is_training = False)
        print("Global Model Confusin Matrix:")
        print(global_test_confusion_matrix)
        #print(global_test_predictions)

    for epoch in range(0, NUM_EPOCHS):
        #Train
        model, opt_new, _, epoch_losses, _, _ = run_epoch(FOLD,training_loader, model, opt, criterion, DEVICE,is_training = True)
        train_loss_avg = np.mean(epoch_losses)

        #Validate
        _, _, predictions, epoch_losses, epoch_accuracies, confusion_m = run_epoch(FOLD,validation_loader, model, opt, criterion, DEVICE,is_training = False)
        val_loss_avg = np.mean(epoch_losses)
        val_accuracy_avg = np.mean(epoch_accuracies)

        #Log
        best_model_results,best_model,early_stop_counter = print_results(epoch,train_loss_avg,val_loss_avg,val_accuracy_avg,confusion_m, best_model_results,best_model,early_stop_counter,model)
        if early_stop_counter == CONFIG['early_stop_checkpoint']: 
            print("Reached early stop checkpoint")
            break
    
    #Test
    print("Testing Personalised Model. Fold: " + str(test_fold))
    _, _, test_predictions, _, epoch_accuracies, test_confusion_matrix = run_epoch(FOLD,test_loader, best_model, best_opt, criterion, DEVICE, is_training = False)
    print("Personalised Model Confusion Matrix")
    print(test_confusion_matrix)
            
    return best_model, best_model_results, test_confusion_matrix, global_test_confusion_matrix


if __name__=='__main__':
    warnings.filterwarnings("ignore")  
    warnings.simplefilter('ignore')

    data, _ = load_data()
    print(len(data[CENTER][1]))

    confusion_matrices_personalised = []
    confusion_matrices_global = []
    cross_val = 'lco' if "lco" in GLOBAL_MODEL else 'ccv'

    print("Running " + GLOBAL_MODEL + " on " + CENTER + " with " + cross_val + " Cross Validation ")
    train_time_start = time.time()
    for test_fold in range(0,NUM_FOLDS):
        model = load_model()
        criterion, opt = set_crit_opt(model)

        best_model,best_model_results,confusion_matrix_personalised,confusion_matrix_global = fine_tune(test_fold, model,data,opt,criterion)
        print("Best Epoch: " + str(best_model_results['epoch']) + "\n" +
              "Validation Loss: " + str(best_model_results['val_loss']) + "\n" +
              "Train Loss: " + str(best_model_results['train_loss']) + "\n" +
              "Test Accuracy: " + str(best_model_results['val_accuracy']))
        confusion_matrices_personalised.append(confusion_matrix_personalised)
        confusion_matrices_global.append(confusion_matrix_global)
        #print("The Best model is")
        #print(best_model_results)
        if SAVE_MODELS:
            save_best_model(best_model, best_model_results['epoch'], test_fold, CENTER,cross_val)
    train_time_end = time.time()
    train_time = calculate_time(train_time_end-train_time_start)
    
    #Log
    tl = 'fresh_model' if TRAIN_PRIVATE else 'transfer_learn'
    file_name = cross_val + "_" + CENTER + "_"+ tl + "_" + str(NUM_TRANSFORMATIONS) + "_"+ str(NUM_EPOCHS) +"_accuracies"
    file = open(file_name, 'w')
    accuracies,_,_,_ = metrics_from_confusion_matrices(confusion_matrices_personalised)
    average_acc_score_personalised = np.mean(accuracies)
    file.write("Centre: " + CENTER + " Input Model CV: " + cross_val + " " + tl  + "\n")
    if not TRAIN_PRIVATE:
        accuracies,_,_,_ = metrics_from_confusion_matrices(confusion_matrices_global)
        average_acc_score_global = np.mean(accuracies)
        file.write("Global: " + str(average_acc_score_global) + "\n")
    file.write("Personalised: " + str(average_acc_score_personalised) + "\n")
    file.write(train_time)
    file.close()