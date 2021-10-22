import pickle
from sklearn import svm

#Load training data
with open('x_trainfile', 'rb') as x_traindata:
    X_train = pickle.load(x_traindata)

with open('y_trainfile', 'rb') as y_traindata:
    y_train = pickle.load(y_traindata)

#Build Support Vector Machine classifier
clf = svm.SVC()
#Train classifier
clf.fit(X_train, y_train)

#Save classifier to PC
with open('SVMClassifier', 'wb') as picklefile:
    pickle.dump(clf,picklefile)
