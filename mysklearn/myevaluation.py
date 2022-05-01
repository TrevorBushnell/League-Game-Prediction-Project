
from mysklearn import myutils

# TODO: copy your myevaluation.py solution from PA5 here
from math import ceil
from mysklearn import myutils
import numpy as np
import copy

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!
    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if type(test_size) == float:
        test_len = ceil(test_size*len(X))
    else:
        test_len = test_size
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    if shuffle == True:
        if random_state == None:
            myutils.randomize_in_place(X,y)
        else:
            myutils.randomize_in_place(X,y,random_state)
    for i in range(len(X)):
        if i < len(X)-test_len:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])
    return X_train,X_test,y_train,y_test # TODO: fix this

def kfold_cross_validation(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.
    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_rain_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold
    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_indexes = [ i for i in range(len(X))]
    if shuffle == True:
        if random_state == None:
            myutils.randomize_in_place(X_indexes)
        else:
            myutils.randomize_in_place(X_indexes,ran_seed=random_state)
    X_test_folds = [ [] for i in range(n_splits)]
    for i in range(len(X)):
        X_test_folds[i%n_splits].append(X_indexes[i])
    X_train_folds = []
    for j in range(len(X_test_folds)):
        indexes = []
        for h in range(len(X_test_folds)):
            if h !=j:
                indexes = indexes + X_test_folds[h]
        X_train_folds.append(indexes)
    return X_train_folds, X_test_folds
        

        

    return [], [] # TODO: fix this

def stratified_kfold_cross_validation(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.
    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.
    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    X_indexes = [ i for i in range(len(X))]
    if shuffle == True:
        if random_state == None:
            myutils.randomize_in_place(X_indexes,y)
        else:
            myutils.randomize_in_place(X_indexes,y,random_state)
    grouped_X =[]
    grouped_y = []
    for i in range(len(X)):
        if grouped_y.count(y[i]) >0:
            grouped_X[grouped_y.index(y[i])].append(X_indexes[i])
        else:
            grouped_y.append(y[i])
            grouped_X.append([X_indexes[i]])
    
    X_test_folds = [ [] for i in range(n_splits)]
    current_index = 0
    for i in range(len(grouped_X)):
        for j in range(len(grouped_X[i])):
            current_index +=1
            X_test_folds[current_index%n_splits].append(grouped_X[i][j])
    
    X_train_folds = []
    for j in range(len(X_test_folds)):
        indexes = []
        for h in range(len(X_test_folds)):
            if h !=j:
                indexes = indexes + X_test_folds[h]
        X_train_folds.append(indexes)
    return X_train_folds, X_test_folds

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.
    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results
    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
    """
    if random_state !=None:
        np.random.seed(random_state)
    if n_samples==None:
        n_samples = len(X)
    X_sample = []
    X_out_of_bag = []
    rand_index_list = []
    for i in range(n_samples):
        rand_index = np.random.randint(0,len(X))
        rand_index_list.append(rand_index)
        X_sample.append(copy.deepcopy(X[rand_index]))
    for h in range(len(X)):
        if rand_index_list.count(h)==0:
            X_out_of_bag.append(copy.deepcopy(X[h]))
    if y == None:
        return X_sample,X_out_of_bag,None,None
    else:
        y_sample = []
        y_out_of_bag = []
        for j in range(len(y)):
            if rand_index_list.count(j)== 0:
                y_out_of_bag.append(copy.deepcopy(y[j]))
        for value in rand_index_list:
            y_sample.append(copy.deepcopy(y[value]))
        return X_sample,X_out_of_bag,y_sample,y_out_of_bag

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix
    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class
    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = [ [0 for i in range(len(labels))] for i in range(len(labels))]
    for i in range(len(y_pred)):
        matrix[labels.index(y_true[i])][labels.index(y_pred[i])] += 1

    return matrix # TODO: fix this

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.
    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).
    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    score = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            score +=1
    if normalize == False:
        return score 
    else: 
        return score / len(y_pred)

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        precision(float): Precision of the positive class
    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels == None:
        labels = [y_true[0]]
        for value in y_true:
            if labels.count(value) > 0:
                pass
            else:
                labels.append(value)
    matrix = confusion_matrix(y_true,y_pred,labels)
    if pos_label == None:
        if matrix[0][0] + matrix[1][0] ==0:
            return 0.0
        else:
            return matrix[0][0]/ (matrix[0][0] + matrix[1][0])
    else: 
        index = labels.index(pos_label)
        if (matrix[index][index] + matrix[len(labels)-1-index][index]) ==0:
            return 0.0
        else:
            return matrix[index][index] / (matrix[index][index] + matrix[len(labels)-1-index][index])

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        recall(float): Recall of the positive class
    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels == None:
        labels = [y_true[0]]
        for value in y_true:
            if labels.count(value) > 0:
                pass
            else:
                labels.append(value)
    matrix = confusion_matrix(y_true,y_pred,labels)
    if pos_label == None:
        if matrix[0][0] + matrix[0][1] ==0:
            return 0.0
        else:
            return matrix[0][0]/ (matrix[0][0] + matrix[0][1])
    else: 
        index = labels.index(pos_label)
        if (matrix[index][index] + matrix[index][len(labels)-1-index]) ==0:
            return 0.0
        else:
            return matrix[index][index] / (matrix[index][index] + matrix[index][len(labels)-1-index])
    return 0.0 # TODO: fix this

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)
    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels
    Returns:
        f1(float): F1 score of the positive class
    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true,y_pred,labels,pos_label)
    recall = binary_recall_score(y_true,y_pred,labels,pos_label)
    if precision + recall == 0:
        return 0
    else:
        return 2* (precision*recall)/(precision+recall)
