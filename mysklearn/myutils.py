# TODO: your reusable general-purpose functions here
import numpy as np
import mysklearn.mypytable
from mysklearn.mypytable import MyPyTable 




import mysklearn.myevaluation
import mysklearn.myevaluation as myevaluation

def randomize_in_place(alist, parallel_list=None,ran_seed=None):
    for i in range(len(alist)):
        # generate a random index to swap this value at i with
        if ran_seed != None:
            np.random.seed(ran_seed)
        rand_index = np.random.randint(0, len(alist)) # rand int in [0, len(alist))
        # do the swap
        alist[i], alist[rand_index] = alist[rand_index], alist[i]
        if parallel_list is not None:
            parallel_list[i], parallel_list[rand_index] =\
                parallel_list[rand_index], parallel_list[i]

def normalize(X_train, X_test):
    for i in range(len(X_train[0])):
        maximum = max([l[i] for l in X_train])
        minimum = min([l[i] for l in X_train])
        for j in range(len(X_train)):
            X_train[j][i] = (X_train[j][i] - minimum)/maximum
        for h in range(len(X_test)):
            X_test[h][i] = (X_test[h][i] - minimum)/maximum
    return X_train, X_test

def DOE_discrete(lst):
    y_train=[]
    for i in range(len(lst)):
        if (lst[i]) >=45: 
            y_train.append(9)
        elif (lst[i]) >= 37:
            y_train.append(8)
        elif (lst[i]) >= 31: 
            y_train.append(7)
        elif (lst[i]) >= 27: 
            y_train.append(6)
        elif (lst[i]) >= 24: 
            y_train.append(5)
        elif (lst[i]) >= 20: 
            y_train.append(4)
        elif (lst[i]) >= 17: 
            y_train.append(3)
        elif (lst[i]) >= 15: 
            y_train.append(2)
        elif (lst[i]) == 14: 
            y_train.append(1)
        else: 
            y_train.append(0)
    return y_train

def get_rows(indexes,lst):
    new_lst = []
    for index in indexes:
        new_lst.append(lst[index])
    return new_lst

def compute_clf_stats(clf,X,y,clf_name):
    X_train_folds,X_test_folds = myevaluation.stratified_kfold_cross_validation(X,y,n_splits=10)
    y_pred_clf = []
    y_test = []
    for i in range(len(X_train_folds)):
        X_train = get_rows(X_train_folds[i],X)
        X_test = get_rows(X_test_folds[i],X)
        y_train = get_rows(X_train_folds[i],y)
        y_test += get_rows(X_test_folds[i],y)
        clf.fit(X_train,y_train)
        y_pred_clf += clf.predict(X_test)
    
    accuracy = myevaluation.accuracy_score(y_test,y_pred_clf)
    print()
    print("Stratified 10-Fold Cross Validation")
    print(clf_name + " : accuracy = ", accuracy, " error = ", 1 - accuracy)

    precision = myevaluation.binary_precision_score(y_test,y_pred_clf)
    recall = myevaluation.binary_recall_score(y_test,y_pred_clf)
    f1_score = myevaluation.binary_f1_score(y_test,y_pred_clf)
    print("binary precision: ", precision)
    print("binary recall: ", recall)
    print("f1 score: ", f1_score)
    labels = [y_pred_clf[0]]
    for value in y_pred_clf:
            if labels.count(value) > 0:
                pass
            else:
                labels.append(value)
    for value in y_test:
        if labels.count(value) > 0:
                pass
        else:
            labels.append(value)
    matrix = myevaluation.confusion_matrix(y_test,y_pred_clf,labels)
    print("Confusion Matrix")
    print(matrix)
