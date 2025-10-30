# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the packages.
2.Analyse the data. 
3.Use modelselection and Countvectorizer to preditct the values. 
4.Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HASMITHA V NANCY
RegisterNumber:  212224040111
*/
```
~~~
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from IPython.display import display, Markdown

data = pd.read_csv(r"C:\Users\admin\Downloads\spam.csv", encoding='Windows-1252')
display(Markdown("## DATA:"))
display(data.head())

display(Markdown(f"## data.shape():\n```text\n{data.shape}\nimage\n```"))

x, y = data['v2'].values, data['v1'].values
display(Markdown(f"## x.shape():\n```text\n{x.shape}\nimage\n```"))
display(Markdown(f"## y.shape():\n```text\n{y.shape}\nimage\n```"))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
display(Markdown(f"## x_train:\n```text\nimage\n```"))
display(Markdown(f"## x_train.shape():\n```text\n{x_train.shape}\nimage\n```"))

cv = CountVectorizer()
x_train, x_test = cv.fit_transform(x_train), cv.transform(x_test)
svc = SVC()
svc.fit(x_train, y_train)

y_pred = svc.predict(x_test)
display(Markdown(f"## y_pred:\n```text\nimage\n```"))

acc = accuracy_score(y_test, y_pred)
display(Markdown(f"## acc (Accuracy):\n```text\n{acc}\nimage\n```"))

con = confusion_matrix(y_test, y_pred)
display(Markdown(f"## con (Confusion Matrix):\n```text\n{con}\nimage\n```"))

cl = classification_report(y_test, y_pred)
display(Markdown(f"## cl (Classification Report):\n```text\n{cl}\nimage\n```"))

~~~

## Output:
## DATA:

<img width="861" height="299" alt="image" src="https://github.com/user-attachments/assets/0436a340-ab85-4175-9da9-70621ac49f25" />

## data.shape():

<img width="424" height="68" alt="image" src="https://github.com/user-attachments/assets/65b868da-b82f-4a11-b2fd-a41b4cb779d2" />

## y.shape():

<img width="362" height="62" alt="image" src="https://github.com/user-attachments/assets/b3fdfef8-911d-4acf-a609-d95c4778fbf1" />

## x_train:

<img width="544" height="232" alt="image" src="https://github.com/user-attachments/assets/b2ad83f0-8cc3-4e33-8311-c7ff9c2d8d47" />

## x_train_shape:

<img width="440" height="62" alt="image" src="https://github.com/user-attachments/assets/561d2d59-12bf-46b9-8a1e-ace5a6c4789f" />

## Accuracy:

<img width="450" height="59" alt="image" src="https://github.com/user-attachments/assets/3ec7501c-e152-471b-81f6-da6bc14d1139" />

## Confusion matrix:

<img width="475" height="83" alt="image" src="https://github.com/user-attachments/assets/859b11bb-842e-4b5c-aac8-667f43ddfe47" />

## Classification Report:

<img width="620" height="256" alt="image" src="https://github.com/user-attachments/assets/c1e0ae52-a73d-4068-b89a-a54561eedc93" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
