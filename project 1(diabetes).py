import sklearn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics  import accuracy_score
dataset= r"C:\Users\BIT\Downloads\diabetes.csv"
diabetes_dataset= pd.read_csv(dataset)
print(diabetes_dataset.head())
print(diabetes_dataset.shape)
print(diabetes_dataset.describe())
print(diabetes_dataset['Outcome'].value_counts())
print(diabetes_dataset.groupby('Outcome').mean())
X= diabetes_dataset.drop(columns='Outcome',axis=1)
Y = diabetes_dataset['Outcome']
print(X)
print(Y)
scaler= StandardScaler()
standardized_data= scaler.fit_transform(X)
print(standardized_data)
X= standardized_data
Y= diabetes_dataset['Outcome']
print(X)
print(Y)
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
classifier= svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)
X_train_prediction= classifier.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction,Y_train)
print('accuracy score of training data:',training_data_accuracy)
X_test_prediction= classifier.predict(X_test)
testing_data_accuracy= accuracy_score(X_test_prediction,Y_test)
print('accuracy score of testing data:',testing_data_accuracy)
def predict_diabetes():
    print("Enter the following details:")
    pregnancies = float(input("Number of Pregnancies: "))
    glucose = float(input("Glucose Level: "))
    blood_pressure = float(input("Blood Pressure: "))
    skin_thickness = float(input("Skin Thickness: "))
    insulin = float(input("Insulin Level: "))
    bmi = float(input("BMI: "))
    diabetes_pedigree_function = float(input("Diabetes Pedigree Function: "))
    age = float(input("Age: "))
    
    # Create a numpy array with the user input
    user_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]], 
                             columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])   # Standardize the user input data
    user_data = scaler.transform(user_data)
    
    # Make a prediction
    prediction = classifier.predict(user_data)
    
    if prediction[0] == 0:
        print("The person is non-diabetic.")
    else:
        print("The person is diabetic.")

# Call the function to get user input and make a prediction
predict_diabetes()