import warnings
warnings.filterwarnings('ignore')

#import numpy as np
import pandas as pd
from sklearn import metrics
from joblib import dump
#import svm
dataset = pd.read_csv("/Users/Arpana/pythonProject/daisee7.csv")

X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


from sklearn.svm import SVC

clf = SVC(kernel='linear', C=1.0, random_state=0)
clf.fit(X_train, y_train)

# generate evaluation metrics
print("Train - Accuracy :", metrics.accuracy_score(y_train, clf.predict(X_train)))
print( "Train - Confusion matrix :",metrics.confusion_matrix(y_train, clf.predict(X_train)))
print( "Train - classification report :", metrics.classification_report(y_train , clf.predict(X_train)))

print( "Test - Accuracy :", metrics.accuracy_score(y_test, clf.predict(X_test)))
print( "Test - Confusion matrix :",metrics.confusion_matrix(y_test, clf.predict(X_test)))
print( "Test - classification report :", metrics.classification_report(y_test, clf.predict(X_test)))
dump(clf,"/Users/Arpana/pythonProject/engagement.joblib");
