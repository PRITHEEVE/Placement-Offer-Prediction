import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random


data = pd.read_csv("/content/placement-dataset.csv")

X = data[["cgpa", "iq"]]  
y = data["placement"]  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


model = GaussianNB()


model.fit(X_train, y_train)

y_pred = model.predict(X_test)  

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


new_cgpa = float(input("Enter CGPA for new data: "))
new_iq = float(input("Enter IQ for new data: "))


new_data = [[new_cgpa, new_iq]] 
prediction = model.predict(new_data)

print("Predicted:", prediction[0])  
if prediction[0] == 0:
    a = [
        "Your performance may currently fall short of industry expectations. It is essential to identify areas for improvement and take proactive steps to enhance your skills and contributions in order to thrive in the professional landscape.",
        "Your current performance level may not meet the industry standards. It is crucial to recognize areas that require improvement and take proactive measures to enhance your skills and capabilities, ensuring a more successful presence in the professional arena."
    ]
    b = random.choice(a)
    print(b)
else:
    print("Your performance is sufficient for survival within the company.")
