import matplotlib.pyplot as plt  
import pandas as pd  
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 


#import Dataset
drug=pd.read_csv("C:/Users/dell/Desktop/Drug prediction/drug1.csv")

#shape
print ("Number of Instances and Attributes",drug.shape)

#peek dataset
print (drug.head(20))

# class distribution
print(drug.groupby('Alcohol').size())

#preprossesing
X = drug.drop('Class', axis=1)  
y = drug['Class']


#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#svm  
svclassifier = SVC(kernel='linear', degree=8)  
svclassifier.fit(X_train, y_train) 

#making Prediction
y_pred = svclassifier.predict(X_test)  

#evaluate
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

# histograms
drug.hist()
plt.show()

# scatter plot matrix
scatter_matrix(drug)
plt.show()



"""
output of the model will be near to 0.92 or 92 % .. 
so we can predict a good svm model 
"""







