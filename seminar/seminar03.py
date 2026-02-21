import numpy as np
import sklearn.naive_bayes as naive_bayes
import sklearn.linear_model as linear_model
from sklearn.metrics import accuracy_score

data = np.load("mnist.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]

x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test = x_test.reshape(x_test.shape[0], -1) / 255.0

y_train = y_train.astype(int)
y_test = y_test.astype(int)

cls_lr = linear_model.LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
cls_lr.fit(x_train, y_train)
pred_lr = cls_lr.predict(x_test)
print('Logistic Regression Accuracy: %.4f' % accuracy_score(y_test, pred_lr))

cls_nb = naive_bayes.MultinomialNB(alpha=0.5)
cls_nb.fit(x_train, y_train)
pred_nb = cls_nb.predict(x_test)
print('Naive Bayes Accuracy: %.4f' % accuracy_score(y_test, pred_nb))
