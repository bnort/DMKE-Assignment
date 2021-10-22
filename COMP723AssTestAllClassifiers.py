import pickle    
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

#Load classifiers
with open('NBClassifier', 'rb') as y_testdata:
    nbclf = pickle.load(y_testdata)

with open('NNClassifier', 'rb') as x_traindata:
    nnclf = pickle.load(x_traindata)

with open('SVMClassifier', 'rb') as x_testdata:
    svmclf = pickle.load(x_testdata)

#Load testing data    
with open('x_testfile', 'rb') as x_testdata:
    X_test = pickle.load(x_testdata)
    
with open('y_testfile', 'rb') as y_testdata:
    y_test = pickle.load(y_testdata)

#Test data on NB Classifier
y_pred = nbclf.predict(X_test)

#Display results as confusion matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
#Display statistical results
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

#Test data on NN Classifier
y_pred = nnclf.predict(X_test)

#Display results as confusion matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
#Display statistical results
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

#Test data on SVM Classifier
y_pred = svmclf.predict(X_test)

#Display results as confusion matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
#Display statistical results
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
