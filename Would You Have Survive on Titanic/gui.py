from tkinter import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


# Load the passenger data
passengers = pd.read_csv('passengers.csv')


# # Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = passengers['Survived']
# # Perform train, test, split
x_train, x_test, y_train, y_test = train_test_split(features, survival, train_size = 0.8, random_state = 4)

# # Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
train_feature = scaler.fit_transform(x_train)
test_feature = scaler.transform(x_test)


# # Create and train the model
model = LogisticRegression()
model.fit(train_feature, y_train)

root=Tk()
root.geometry("700x600")
root.maxsize(700, 600)
root.minsize(700, 600)
# root.configure(background="grey11")
root.title("Would You ")

def click(age, passenger_class, gender):
	first_class = 1 if passenger_class == 1 else 0
	sencond_class = 1 if passenger_class == 2 else 0
	new_list = np.array([[gender, age, first_class, sencond_class]])
	sample_passengers = scaler.transform(new_list)
	predict = model.predict(sample_passengers)
	print(new_list)
	print(predict)
	classification_label1['text'] = 'You Would Be' if predict[0]==1 else ''
	classification_label2['text'] = 'Alive ' if predict[0]==1 else 'Sorry, you would be dead'
	classification_label1['fg'] = 'red' if predict[0] == 0 else 'green'
	classification_label2['fg'] = 'red' if predict[0] == 0 else 'green'

title_frame = Frame(root, width = 600, height = 100)
title_frame.grid(row = 0, column = 0)
input_frame = Frame(root, width=600, height=300)
input_frame.grid(row = 1, column = 0)
output_frame =Frame(root, width=600, height=300)
output_frame.grid(row = 2, column = 0)

title = Label(title_frame, text = 'Would you have survived on Titanic🤨', font = ('arial 20 bold'))
title.grid(row = 0, column = 0)

age_label = Label(input_frame, text = 'Age')
age_label.grid(row = 0, column = 0)
age_input=Entry(input_frame, font=("arial 20 bold"), bd=2, width=14, justify=LEFT)
age_input.grid(row=0, column=1, sticky=W+E+S)

class_label = Label(input_frame, text = 'Passenger Class(1/2/3)')
class_label.grid(row = 1, column = 0)
class_input=Entry(input_frame, font=("arial 20 bold"), bd=2, width=14, justify=LEFT)
class_input.grid(row=1, column=1, sticky=W+E+S)


gender_label = Label(input_frame, text = 'Gender(1 for female and 0 for male)')
gender_label.grid(row = 2, column = 0)
gender_input=Entry(input_frame, font=("arial 20 bold"), bd=2, width=14, justify=LEFT)
gender_input.grid(row=2, column=1, sticky=W+E+S)


convert = Button(input_frame, text = 'Analyze', command = lambda: click(int(age_input.get()), int(class_input.get()), int(gender_input.get())))
convert.grid(row = 3, column = 0, columnspan = 2, padx = 300)


classification_label1 = Label(output_frame, text = '??', font = ('arial 30 bold'))
classification_label1.grid(row = 0, column = 0)
classification_label2 = Label(output_frame,  font = ('arial 30 bold'))
classification_label2.grid(row = 1, column = 0)

root.mainloop()