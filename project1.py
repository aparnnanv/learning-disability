# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 11:46:15 2024

@author: USER
"""
import tkinter as tk
from tkinter import *
from tkinter import fieldialog,messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import mathplotlib.pyplot as plt

root=tk.Tk()
root.config(bg="yellow")
root.title("learning disability")
df=None
model = None 
accuracy = None
x_test,y_test=None,None

def load_dataset():
file_path =filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
 if not file_path:
return
    try:
data = pd.read_csv(file_path)
messagebox.showinfo("Success", "Dataset loaded successfully!")
print("Dataset Preview:")
print(data.head())
print("\ndataset information")
print(data.info())
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {e}")

required_columns=['response_time','error_rate','word_accuracy','label']
if not all(columns in data.columns for columns in required_columns):
    raise ValueError(f"the dataset must contain the following columns:{required}")
messagebox.showerror("Error", f"The dataset must contain the following columns: {required_columns}")
return
    
def train_model():
   if data is None:
   messagebox.showwarning("Warning", "Please load a dataset first!")
   return
try:
    
x=data[['response_time','error_rate','word_accuracy']]
y=data['label']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

model=RandomForestClassifier(random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
acc=accuracy_score(y_test,y_pred)
report=classification_report(y_test,y_pred)
print("Confusion Matrix:")
print(cm)
print(f"Accuracy Score: {acc}")
print("\nclassification_report:")
print(report)
messagebox.showinfo("Success", f"Model trained successfully!\nAccuracy: {acc:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")

def show_confusion_matrix():
    if model is None or df is None:
messagebox.showerror("error","model must be trained before displaying the confusion matrix!")
return
  try:
      y_pred=model.predict(x_test)
      cm=confusion_matrix(y_test,y_pred)
      fig,

new_sample=np.array([[0.65,12,0.35]])
prediction=model.predict(new_sample)
print(f"Prediction for new sample: {'Potential issue' if prediction[0] == 1 else 'No issue'}")

