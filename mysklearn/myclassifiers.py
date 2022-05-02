##############################################
# Programmer: Ben Lombardi
# Class: CptS 322-01, Spring 2022
# Programming Assignment #7
# 4/14/2022
# I did not attempt the bonus...
# 
# Description: This program completes functions and unit tests
# for the Decision Tree Classifier
##############################################
import copy

from matplotlib.style import available

from mysklearn import myevaluation, myutils

# TODO: copy your myclassifiers.py solution from PA4-6 here
import operator
from turtle import distance

from sympy import maximum, minimum
from mysklearn import myutils
import numpy as np
import math

def compute_euclidean_distance(v1, v2):
    if type(v1[0]) != str:
        return np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    else: 
        dist = []
        for i in range(len(v1)):
            if v1[i] == v2[i]:
                dist.append(0)
            else:
                dist.append(1)
        return dist

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).
    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data
    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.
        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        if regressor == None:
            regressor = MySimpleLinearRegressor()
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor.fit(X_train,y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred_num = self.regressor.predict(X_test)
        y_pred_clf = self.discretizer(y_pred_num)
        return y_pred_clf # TODO: fix this

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.
    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.
        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """

        self.X_train = X_train
        self.y_train = y_train
    

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices =[]
        for j in range(len(X_test)):
            distances.append([])
            neighbor_indices.append([])
            k_dist = []
            for i in range(len(self.X_train)):
                dist = compute_euclidean_distance(self.X_train[i],X_test[j])
                k_dist.append([i,dist])
            k_dist.sort(key=operator.itemgetter(-1))
            k_near = k_dist[:self.n_neighbors]
            for h in range(len(k_near)):
                distances[j].append(k_near[h][-1])
                neighbor_indices[j].append(k_near[h][0])
        return distances, neighbor_indices
    

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances,neighbor_indexes = self.kneighbors(X_test)
        y_predicted = []
        for i in range(len(X_test)):
            y_vals = []
            for value in neighbor_indexes[i]:
                y_vals.append(self.y_train[value])
            vote = y_vals[0]
            vote_count = y_vals.count(y_vals[0])
            for value in y_vals:
                if value == vote:
                    pass
                elif vote_count < y_vals.count(value):
                    vote = value
                    vote_count = y_vals.count(value)
            y_predicted.append(vote)
            
        return y_predicted 

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.
    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()
    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.
        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        vote = y_train[0]
        vote_count = y_train.count(vote)
        for value in y_train:
                if value == vote:
                    pass
                elif vote_count < y_train.count(value):
                    vote = value
                    vote_count = y_train.count(value)
        self.most_common_label = vote


    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.most_common_label for i in range(len(X_test))] # TODO: fix this


class MySimpleLinearRegressor:
    """Represents a simple linear regressor.
    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b
    Notes:
        Loosely based on sklearn's LinearRegression:
            https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.
        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.
        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        X_train = [x[0] for x in X_train] # convert 2D list with 1 col to 1D list
        self.slope, self.intercept = MySimpleLinearRegressor.compute_slope_intercept(X_train,
            y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.
        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list
                    with one element e.g. [[0], [1], [2]]
        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        predictions = []
        if self.slope is not None and self.intercept is not None:
            for test_instance in X_test:
                predictions.append(self.slope * test_instance[0] + self.intercept)
        return predictions

    @staticmethod # decorator to denote this is a static (class-level) method
    def compute_slope_intercept(x, y):
        """Fits a simple univariate line y = mx + b to the provided x y data.
        Follows the least squares approach for simple linear regression.
        Args:
            x(list of numeric vals): The list of x values
            y(list of numeric vals): The list of y values
        Returns:
            m(float): The slope of the line fit to x and y
            b(float): The intercept of the line fit to x and y
        """
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) \
            / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
        # y = mx + b => y - mx
        b = mean_y - m * mean_x
        return m, b

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.
    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.
        X_attributes: list containing each x_attribute and its index 
        labels: list containing each possible class attribute
    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.posteriors = None
        self.X_attributes = None
        self.labels = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.
        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        count = [0]
        labels = [y_train[0]]
        for i in range(len(y_train)):
            if labels.count(y_train[i]) >0:
                count[labels.index(y_train[i])] += 1
            else:
                labels.append(y_train[i])
                count.append(1)
        self.priors = [value / len(y_train) for value in count]
        self.labels =labels
        X_attributes = []
        for i in range(len(X_train[0])):
            for j in range(len(X_train)):
                if X_attributes.count([X_train[j][i],i]) > 0:
                    pass
                else:
                    X_attributes.append([X_train[j][i],i])
        self.X_attributes = X_attributes
        grouped_X =[]
        grouped_y = []
        for i in range(len(X_train)):
            if grouped_y.count(y_train[i]) >0:
                grouped_X[grouped_y.index(y_train[i])].append(X_train[i])
            else:
                grouped_y.append(y_train[i])
                grouped_X.append([X_train[i]])
        posteriors = [[[] for i in range(len(labels))] for j in range(len(X_attributes))]
        for i in range(len(labels)):
            for j in range(len(X_attributes)):
                total = 0
                for h in range(len(grouped_X[i])):
                    if grouped_X[i][h][X_attributes[j][-1]] == X_attributes[j][0]:
                        total +=1
                posteriors[j][i] = total/count[i]
        self.posteriors = posteriors

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.
        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = []
        for entry in X_test:
            prob_list = [1 for i in range(len(self.priors))]
            for value in self.priors:
                for att in entry:
                    prob_list[self.priors.index(value)] *= self.posteriors[self.X_attributes.index([att,entry.index(att)])][self.priors.index(value)]
                prob_list[self.priors.index(value)] *= value
            y_pred.append(self.labels[prob_list.index(max(prob_list))])
        return y_pred

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.header = None
        self.attribute_domains = None
        self.attribute_range = None

    def fit(self, X_train, y_train,available_atts = 0):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.header = ["att" + str(i) for i in range(len(X_train[0]))]
        if available_atts != 0:
            temp_header = []
            for i in range(len(available_atts)):
                temp_header.append("att" + str(available_atts[i]))
            self.header = temp_header
            del_atts = [i for i in range(len(X_train[0]))]
            for i in sorted(del_atts,reverse=True):
                if available_atts.count(i) !=0:
                    del_atts.remove(i)
            for index in sorted(del_atts,reverse=True):
                for row in X_train:
                    del row[index]

        att_domain = dict()
        for i in range(len(self.header)):
            attributes = [X_train[0][i]]
            for j in range(len(X_train)):
                if attributes.count(X_train[j][i])==0:
                    attributes.append(X_train[j][i])
                else: 
                    pass
            att_domain[self.header[i]] = attributes
        self.attribute_domains = att_domain

        att_range = [y_train[0]]
        for i in range(len(y_train)):
            if att_range.count(y_train[i]) ==0:
                att_range.append(y_train[i])
            else:
                pass
        self.attribute_range = att_range

        # next, make a copy of your header... tdidt() is going
        # to modify the list
        train = [X_train[i]+[y_train[i]] for i in range(len(X_train))]
        available_attributes = self.header.copy()
        instances = train.copy()
        # also: recall that python is pass by object reference
        tree = self.tdidt(instances, available_attributes)
        # note: unit test is going to assert that tree == interview_tree_solution
        # (mind the attribute domain ordering)
        self.tree = tree # TODO: fix this

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        X_test1 = copy.deepcopy(X_test)
        if len(self.header) !=len(X_test1[0]):
            available_atts = [int(value[3:]) for value in self.header]
            del_atts = [i for i in range(len(X_test1[0]))]
            for i in sorted(del_atts,reverse=True):
                if available_atts.count(i) !=0:
                    del_atts.pop(i)
            for index in sorted(del_atts,reverse=True):
                for row in X_test1:
                    del row[index]
  
        y_predicted = []
        for entry in X_test1:
            tree = copy.deepcopy(self.tree)   
            while tree[0] != "Leaf":
                att_index = self.header.index(tree[1])
                if self.attribute_domains[tree[1]].count(entry[att_index]) ==0:
                    print(tree[1])
                    print(entry[att_index])
                    #print(entry)
                    print(self.attribute_domains[tree[1]])
                    break
                for i in range(len(tree)-2):
                    if entry[att_index] == tree[i+2][1]:
                        tree = tree[i+2][2]
                        break
            y_predicted.append(tree[1])
        return y_predicted # TODO: fix this

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        tree = self.tree.copy()
        if attribute_names == None:
            attribute_names = self.header
        self.traverse_tree(attribute_names,tree,"",class_name)
        pass # TODO: fix this

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

    def tdidt(self,current_instances, available_attributes):
        # basic approach (uses recursion!!):

        # select an attribute to split on
        attribute = self.select_attribute(current_instances, available_attributes)
        available_attributes.remove(attribute)
        tree = ["Attribute", attribute]
        # group data by attribute domains (creates pairwise disjoint partitions)
        partitions = self.partition_instances(current_instances, attribute)
        # for each partition, repeat unless one of the following occurs (base case)
        for att_value, att_partition in partitions.items():
            value_subtree = ["Value", att_value]

            #    CASE 1: all class labels of the partition are the same => make a leaf node
            if len(att_partition) > 0 and self.all_same_class(att_partition):
                value_subtree.append(["Leaf",att_partition[0][-1],len(att_partition),len(current_instances)])
                tree.append(value_subtree)
            #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
            elif len(att_partition) > 0 and len(available_attributes) == 0:
                count = [ 0 for i in range(len(self.attribute_range))]
                for i in range(len(att_partition)):
                    count[self.attribute_range.index(att_partition[i][-1])]+=1
                labels = self.attribute_range.copy()
                if max(count)/len(current_instances) == 1/2:
                    labels = self.attribute_range.copy()
                    labels.sort
                    value_subtree.append(["Leaf",labels[0],max(count),len(current_instances)])
                else:
                    value_subtree.append(["Leaf",self.attribute_range[count.index(max(count))],max(count),len(current_instances)])
                tree.append(value_subtree)
            #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
            elif len(att_partition) == 0:
                count = [ 0 for i in range(len(self.attribute_range))]
                for i in range(len(current_instances)):
                        count[self.attribute_range.index(current_instances[i][-1])]+=1
                labels = self.attribute_range.copy()
                if max(count)/len(current_instances) == 1/2:
                    labels.sort
                    return ["Case 3", labels[0],len(current_instances)]
                else:
                    return ["Case 3", self.attribute_range[count.index(max(count))],len(current_instances)]
            else: # the previous conditions are all false... recurse!!
                subtree = self.tdidt(att_partition, available_attributes.copy())
                if subtree[0] == "Case 3":
                    subtree = ["Leaf", subtree[1],subtree[2], len(current_instances)]
                value_subtree.append(subtree)
                tree.append(value_subtree)
                # note the copy
                # TODO: append subtree to value_subtree and to tree
                # appropriately
        return tree

    def select_attribute(self,instances, attributes):
        enew = []
        for i in range(len(attributes)):
            total_count = [ 0 for i in range(len(self.attribute_domains[attributes[i]]))]
            att_count = [[0 for i in range(len(self.attribute_range))] for j in range(len(self.attribute_domains[attributes[i]]))]
            avg_entropy = 0
            for j in range(len(instances)):
                total_count[self.attribute_domains[attributes[i]].index(instances[j][self.header.index(attributes[i])])] +=1
                att_count[self.attribute_domains[attributes[i]].index(instances[j][self.header.index(attributes[i])])][self.attribute_range.index(instances[j][-1])] +=1
            for h in range(len(self.attribute_domains[attributes[i]])):
                entropy = 0
                for k in range(len(att_count[h])):
                    if att_count[h][k] ==0:
                        pass
                    else:
                        entropy += -((att_count[h][k])/total_count[h] * math.log(att_count[h][k]/total_count[h],2))
                entropy = entropy* (total_count[h]) / len(instances)
                avg_entropy+=entropy
            enew.append(avg_entropy)          
        return attributes[enew.index(min(enew))]

    def partition_instances(self,instances, split_attribute):
        # lets use a dictionary
        partitions = {} # key (string): value (subtable)
        att_index = self.header.index(split_attribute) # e.g. 0 for level
        att_domain = self.attribute_domains[split_attribute] # e.g. ["Junior", "Mid", "Senior"]
        att_domain.sort()
        for att_value in att_domain:
            partitions[att_value] = []
            for instance in instances:
                if instance[att_index] == att_value:
                    partitions[att_value].append(instance)

        return partitions

    def all_same_class(self,partition):
        labels = [partition[0][-1]]
        for i in range(len(partition)):
            if labels.count(partition[i][-1])==0:
                labels.append(partition[i][-1])
            else:
                pass
        if len(labels) ==1:
            return True
        else:
            return False
    
    def recurse_decision_rules(self,tree,attribute_names,class_name,start,list):
        if tree[0] == "Leaf":
            return " THEN " + class_name + " = " + tree[1]
        else:
            for i in range(len(tree)-2):
                if start ==0:
                    list.append("IF " + attribute_names[int(tree[1][-1])] + " = " + tree[2][1] + self.recurse_decision_rules(tree[i+2][2],attribute_names,class_name,1,list))
                else:
                    return (" AND " + attribute_names[int(tree[1][-1])] + " = " + tree[2][1] + self.recurse_decision_rules(tree[i+2][2],attribute_names,class_name,1,list))
        return list
    
    def traverse_tree(self,header,tree,rule,class_name):
        info_type = tree[0]
        if info_type == "Leaf":
            print("IF " + rule[:-5] + " THEN " + class_name + "=" + tree[1])
        else:
            for i in range(2, len(tree)):
                value_list = tree[i]
                self.traverse_tree(header, value_list[2], rule + header[int(tree[1][-1])] + "=" + value_list[1] + " AND ",class_name)


class MyRandomForestClassifier:

    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.M = 0
        self.N = 0
        self.F = 0
        self.forest = None
        self.attribute_range = None
    
    def fit(self,X,y,M,N,F,random_state=0):
        att_range = [y[0]]
        for i in range(len(y)):
            if att_range.count(y[i]) ==0:
                att_range.append(y[i])
            else:
                pass
        self.attribute_range = att_range
        self.M = M
        self.N = N
        self.F = F
        np.random.seed(random_state)
        att_indexes = list(range(len(X[0])))
        temp_forest = []
        temp_accuracy = []

        for i in range(M):
            X_train,X_test,y_train,y_test = myevaluation.bootstrap_sample(X,y,random_state=random_state)
            np.random.seed(i)
            selected_att = self.compute_random_subset(att_indexes,F)
            selected_att.sort()
            tree_clf = MyDecisionTreeClassifier()
            tree_clf.fit(X_train,y_train,available_atts=selected_att)
            temp_forest.append(tree_clf)
            y_pred = tree_clf.predict(X_test)
            temp_accuracy.append(myevaluation.accuracy_score(y_test,y_pred))
        for i in range(M-N):
            temp_forest.pop(temp_accuracy.index(min(temp_accuracy)))
            temp_accuracy.pop(temp_accuracy.index(min(temp_accuracy)))
        print("fitted")
        self.forest = temp_forest
    
    def predict(self,X_test):
        print("predicting2")
        y_pred = []
        votes = []
        for i in range(len(self.forest)):
            tree_clf = self.forest[i]
            votes.append(tree_clf.predict(X_test))
        print("predicing 3")
        for i in range(len(X_test)):
            vote = []
            vote_count = []
            for j in range(len(votes)):
                vote.append(votes[j][i])
            for value in self.attribute_range:
                vote_count.append(vote.count(value))
            y_pred.append(self.attribute_range[vote_count.index(max(vote_count))])
        return y_pred             
                
    def compute_random_subset(self,values, num_values):
        # there is a function np.random.choice()
        values_copy = values[:] # shallow copy
        np.random.shuffle(values_copy) # in place shuffle
        return values_copy[:num_values]
            
