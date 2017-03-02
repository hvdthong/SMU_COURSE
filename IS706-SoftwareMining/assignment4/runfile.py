from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
    for i in range(len(y_hat)):
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
    for i in range(len(y_hat)):
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 0:
            TN += 1

    return TP, FP, FN, TN


def getfeatures(feature):
    x = []
    for v in feature:
        v_split = v.split(':')
        x.append(float(v_split[1]))
    return x


def fileprocessing(content):
    X, y = [], []
    for line in content:
        split_ = line.split()
        x_ = [split_[i] for i in range(1, len(split_))]
        X.append(getfeatures(x_))
        if split_[0] == '+1':
            y.append(1)
        else:
            y.append(0)
    X_scaled = preprocessing.scale(np.array(X))
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X_scaled = min_max_scaler.fit_transform(np.array(X))
    return X_scaled, np.array(y)


def clf_algorithms(X, y, folds, clf):
    kf = StratifiedKFold(n_splits=folds)
    TP, FP, FN = 0, 0, 0
    for train, test in kf.split(X, y):
        # print len(train), len(test)
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]

        if clf == 'LR':
            model = LogisticRegression(class_weight='balanced')
        elif clf == 'SVM':
            model = svm.LinearSVC(class_weight='balanced', max_iter=10000)
            # model = svm.NuSVC(max_iter=1000)
        elif clf == 'NB':
            model = GaussianNB()
        elif clf == 'DT':
            model = tree.DecisionTreeClassifier()

        model.fit(X_train, y_train)
        predict_ = model.predict(X_test)

        TP_, FP_, FN_, TN_ = perf_measure(y_test, predict_)
        TP += TP_
        FP += FP_
        FN += FN_

    P = TP / float(TP + FP)
    R = TP / float(TP + FN)
    F = 2 * P * R / (P + R)
    print 'Classification %s KFold %i Precision %f Recall %f F1 %f' % (clf, folds, P, R, F)


if __name__ == '__main__':
    fname = "./features.dat"
    with open(fname) as f:
        content = f.readlines()
    X, y = fileprocessing(content)
    folds = [2, 10]
    clf = ['LR', 'SVM', 'NB', 'DT']
    # folds, clf = 10, 'LR'
    # folds, clf = 10, 'SVM'
    # folds, clf = 10, 'NB'
    # folds, clf = 10, 'DT'
    print 'Number of positive instances:%i ' % (len([i for i in y if i == 1]))
    print 'Number of negative instances:%i ' % (len([i for i in y if i == 0]))
    for f in folds:
        print 'Applying %i fold cross-validation' % (f)
        for c in clf:
            clf_algorithms(X, y, f, c)
