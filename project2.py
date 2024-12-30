# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 13:27:50 2024

@author: USER
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Initialize global variables
data = None
model = None
accuracy = None
x_test, y_test = None, None

# Create the main window
root = tk.Tk()
root.config(bg="yellow")
root.title("Learning Disability Prediction")
root.geometry("600x400")

def load_dataset():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not file_path:
        return
    try:
        data = pd.read_csv(file_path)
        messagebox.showinfo("Success", "Dataset loaded successfully!")
        print("Dataset Preview:")
        print(data.head())
        print("\nDataset Information:")
        print(data.info())

        # Validate required columns
        required_columns = ['response_time', 'error_rate', 'word_accuracy', 'label']
        if not all(col in data.columns for col in required_columns):
            messagebox.showerror("Error", f"The dataset must contain the following columns: {required_columns}")
            data = None
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset: {e}")

def train_model():
    global model, accuracy, x_test, y_test
    if data is None:
        messagebox.showwarning("Warning", "Please load a dataset first!")
        return

    try:
        # Split data into features and target
        x = data[['response_time', 'error_rate', 'word_accuracy']]
        y = data['label']

        # Train-test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        # Train the Random Forest model
        model = RandomForestClassifier(random_state=42)
        model.fit(x_train, y_train)

        # Evaluate the model
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Display results in the console
        print("Confusion Matrix:")
        print(cm)
        print(f"Accuracy Score: {accuracy}")
        print("\nClassification Report:")
        print(report)

        messagebox.showinfo("Success", f"Model trained successfully!\nAccuracy: {accuracy:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to train model: {e}")

def show_confusion_matrix():
    if model is None:
        messagebox.showerror("Error", "Model must be trained before displaying the confusion matrix!")
        return

    try:
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)

        # Display the confusion matrix
        fig, ax = plt.subplots()
        cax = ax.matshow(cm, cmap='Blues')
        plt.colorbar(cax)
        ax.set_xticklabels([''] + ['No Issue', 'Potential Issue'])
        ax.set_yticklabels([''] + ['No Issue', 'Potential Issue'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        canvas=FigureCanvasTkAgg(fig)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH,expand=True)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to display confusion matrix: {e}")

def predict_new_sample():
    if model is None:
        messagebox.showwarning("Warning", "Please train the model first!")
        return

    try:
        # Collect user inputs for prediction
        response_time = float(response_time_entry.get())
        error_rate = float(error_rate_entry.get())
        word_accuracy = float(word_accuracy_entry.get())

        # Prepare the input for prediction
        new_sample = np.array([[response_time, error_rate, word_accuracy]])
        prediction = model.predict(new_sample)
        result = "Potential Issue" if prediction[0] == 1 else "No Issue"

        # Display the prediction result
        result_label.config(text=f"Prediction: {result}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict: {e}")

# UI Elements
tk.Button(root, text="Load Dataset", command=load_dataset, width=20).pack(pady=30)
tk.Button(root, text="Train Model", command=train_model, width=20).pack(pady=10)
tk.Button(root, text="Show Confusion Matrix", command=show_confusion_matrix, width=20).pack(pady=10)

tk.Label(root, text="Enter Response Time:").pack()
response_time_entry = tk.Entry(root)
response_time_entry.pack()

tk.Label(root, text="Enter Error Rate:").pack()
error_rate_entry = tk.Entry(root)
error_rate_entry.pack()

tk.Label(root, text="Enter Word Accuracy:").pack()
word_accuracy_entry = tk.Entry(root)
word_accuracy_entry.pack()

tk.Button(root, text="Predict", command=predict_new_sample, width=20).pack(pady=10)

result_label = tk.Label(root, text="", fg="blue", bg="yellow")
result_label.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()