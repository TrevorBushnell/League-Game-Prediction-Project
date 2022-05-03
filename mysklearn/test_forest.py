import numpy as np

from mysklearn.myclassifiers import MyRandomForestClassifier

def test_random_forest_fit():
    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    tree_solutions = [['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 4, 10]], ['Value', 'yes', ['Leaf', 'True', 3, 10]]]], ['Value', 'yes', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 1, 4]], ['Value', 'yes', ['Leaf', 'True', 2, 4]]]]],
    ['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 4, 10]], ['Value', 'yes', ['Leaf', 'True', 3, 10]]]], ['Value', 'yes', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 1, 4]], ['Value', 'yes', ['Leaf', 'True', 2, 4]]]]],
     ['Attribute', 'att3', ['Value', 'no', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 4, 10]], ['Value', 'yes', ['Leaf', 'True', 3, 10]]]], ['Value', 'yes', ['Attribute', 'att2', ['Value', 'no', ['Leaf', 'True', 1, 4]], ['Value', 'yes', ['Leaf', 'True', 2, 4]]]]], 
     ['Attribute', 'att1', ['Value', 'Java', ['Leaf', 'True', 2, 14]], ['Value', 'Python', ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'True', 5, 8]], ['Value', 'yes', ['Leaf', 'True', 1, 8]]]], ['Value', 'R', ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'True', 1, 4]], ['Value', 'yes', ['Leaf', 'True', 2, 4]]]]], 
     ['Attribute', 'att0', ['Value', 'Junior', ['Attribute', 'att3', ['Value', 'no', ['Leaf', 'True', 5, 7]], ['Value', 'yes', ['Leaf', 'False', 2, 7]]]], ['Value', 'Mid', ['Leaf', 'True', 4, 14]], ['Value', 'Senior', ['Leaf', 'False', 3, 14]]]]
    forest_clf = MyRandomForestClassifier()
    forest_clf.fit(X_train_interview,y_train_interview,7,5,2,random_state=0)
    for i in range(len(forest_clf.forest)):
        assert forest_clf.forest[i].tree == tree_solutions[i]


def test_random_forest_predict():
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    X_test_interview = [["Junior","Java","yes","no"],["Junior","Java","yes","yes"]]
    forest_clf = MyRandomForestClassifier()
    forest_clf.fit(X_train_interview,y_train_interview,20,7,2)
    y_pred = forest_clf.predict(X_test_interview)
    assert y_pred == ["True","True"]
