import pickle
from sklearn.naive_bayes import GaussianNB

#Load training data
with open('x_trainfile', 'rb') as x_traindata:
    X_train = pickle.load(x_traindata)

with open('y_trainfile', 'rb') as y_traindata:
    y_train = pickle.load(y_traindata)

#Build Gaussian Naive Bayes classifier
gnb = GaussianNB()
#Train classifier
gnb.fit(X_train, y_train)

#Save classifier to PC
with open('NBClassifier', 'wb') as picklefile:
    pickle.dump(gnb,picklefile)