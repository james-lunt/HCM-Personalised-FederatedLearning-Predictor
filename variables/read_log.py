import pickle
import pandas as pd
# Permanently changes the pandas settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
with open("LCO_FL_VallD_Seed2.pkl", "rb") as f:
    data = pickle.load(f)
print("Final Train:")
print(data['final_train'])
print("Final Val:")
print(data['final_val'])
print("Final confusion matrices:")
print(data['final_confusion_matrices'])
print("Test Predictions:")
print(data['test_predictions'])
print("Next Fold:")
print(data['next_fold'])