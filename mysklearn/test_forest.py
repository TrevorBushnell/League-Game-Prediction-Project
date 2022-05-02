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
    forest_clf = MyRandomForestClassifier()
    forest_clf.fit(X_train_interview,y_train_interview,20,7,2,random_state=0)
    assert len(forest_clf.forest) == 7
    print(forest_clf.forest[0].tree)

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
    #assert y_pred == ["True","False"]


from mysklearn.myclassifiers import MyDecisionTreeClassifier

# TODO: copy your test_myclassifiers.py solution from PA4-6 here

def test_decision_tree_classifier_fit():
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

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_interview = \
            ["Attribute", "att0",
                ["Value", "Junior", 
                    ["Attribute", "att3",
                        ["Value", "no", 
                            ["Leaf", "True", 3, 5]
                        ],
                        ["Value", "yes", 
                            ["Leaf", "False", 2, 5]
                        ]
                    ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Senior",
                    ["Attribute", "att2",
                        ["Value", "no",
                            ["Leaf", "False", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "True", 2, 5]
                        ]
                    ]
                ]
            ]
    tree_clf = MyDecisionTreeClassifier()
    tree_clf.fit(X_train_interview,y_train_interview)
    assert tree_clf.tree == tree_interview
    # bramer degrees dataset
    header_degrees = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    X_train_degrees = [
        ['A', 'B', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'A', 'B', 'B'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'A', 'B', 'B', 'A'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B'],
        ['A', 'A', 'A', 'A', 'A'],
        ['B', 'A', 'A', 'B', 'B'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'B', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'B', 'B', 'B'],
        ['B', 'B', 'B', 'B', 'B'],
        ['A', 'A', 'B', 'A', 'A'],
        ['B', 'B', 'B', 'A', 'A'],
        ['B', 'B', 'A', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['B', 'A', 'B', 'A', 'B'],
        ['A', 'B', 'B', 'B', 'A'],
        ['A', 'B', 'A', 'B', 'B'],
        ['B', 'A', 'B', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B']
    ]
    y_train_degrees = ['SECOND', 'FIRST', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                    'SECOND', 'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND',
                    'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND', 'FIRST',
                    'SECOND', 'SECOND', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                    'SECOND', 'SECOND']

    tree_degrees = ['Attribute','att0', 
                        ['Value', 'A',
                            ['Attribute', 'att4',
                                ['Value', 'A',
                                    ['Leaf', 'FIRST',5,14]
                                ],
                                ['Value', 'B',
                                    ['Attribute','att3',
                                        ['Value', 'A',
                                            ['Attribute','att1',
                                                ['Value', 'A',
                                                    ['Leaf','FIRST',1,2 ]
                                                ],
                                                ['Value','B',
                                                    ['Leaf','SECOND',1,2]
                                                ]
                                            ]
                                        ],
                                        ['Value','B',
                                            ['Leaf', 'SECOND',7,9]
                                        ]
                                    ]
                                ]
                            ]
                        ],
                        ['Value','B',
                            ['Leaf', 'SECOND',12,26]
                        ]
                    ]
    tree_clf.fit(X_train_degrees,y_train_degrees)
    assert tree_clf.tree == tree_degrees

    X_train_iphone = [
        ["1","3","fair"],
        ["1","3","excellent"],
        ["2","3","fair"],
        ["2","2","fair"],
        ["2","1","fair"],
        ["2","1","excellent"],
        ["2","1","excellent"],
        ["1","2","fair"],
        ["1","1","fair"],
        ["2","2","fair"],
        ["1","2","excellent"],
        ["2","2","excellent"],
        ["2","3","fair"],
        ["2","2","excellent"],
        ["2","3","fair"]
    ]
    y_train_iphone = ["no","no","yes","yes","yes","no","yes","no","yes","yes","yes","yes","yes","no","yes"]

    tree_iphone = ["Attribute", "att0",
                        ["Value","1",
                            ["Attribute","att1",
                                ["Value", "1",
                                    ["Leaf","yes",1,5]                                
                                ],
                                ["Value", "2",
                                    ["Attribute","att2",
                                        ["Value","excellent",
                                            ["Leaf","yes",1,2]
                                        ],
                                        ["Value","fair",
                                            ["Leaf","no",1,2]
                                        ]
                                    ]
                                ],
                                ["Value","3",
                                    ["Leaf","no",2,5]
                                ]
                            ]
                        ],
                        ["Value","2",
                            ["Attribute","att2",
                                ["Value","excellent",
                                    ["Leaf","no",4,10]
                                ],
                                ["Value","fair",
                                    ["Leaf","yes",6,10]
                                ]
                            ]
                        ]

                ]
    
    ['Attribute', 'att0', 
        ['Value', '1', 
            ['Attribute', 'att1', 
                ['Value', '1', 
                    ['Leaf', 'yes', 1, 5]
                ], 
                ['Value', '2', 
                    ['Attribute', 'att2', 
                        ['Value', 'excellent', 
                            ['Leaf', 'yes', 1, 2]
                        ], 
                        ['Value', 'fair', 
                            ['Leaf', 'no', 1, 2]
                        ]
                    ]
                ], 
                ['Value', '3', 
                    ['Leaf', 'no', 2, 5]
                ]
            ]
        ], 
        ['Value', '2', 
            ['Attribute', 'att2', 
                ['Value', 'excellent', 
                    ['Leaf', 'no', 4, 10]
                ], 
                ['Value', 'fair', 
                    ['Leaf', 'yes', 6, 10]
                ]
            ]
        ]
    ]

    tree_clf.fit(X_train_iphone,y_train_iphone) 
    assert tree_clf.tree == tree_iphone

def test_decision_tree_classifier_predict():
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

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    
    X_test_interview = [["Junior","Java","yes","no"],["Junior","Java","yes","yes"]]
    tree_clf = MyDecisionTreeClassifier()
    tree_clf.fit(X_train_interview,y_train_interview)
    assert tree_clf.predict(X_test_interview) == ["True","False"]

    # bramer degrees dataset
    header_degrees = ["SoftEng", "ARIN", "HCI", "CSA", "Project", "Class"]
    X_train_degrees = [
        ['A', 'B', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'A', 'B', 'B'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'A', 'B', 'B', 'A'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B'],
        ['A', 'A', 'A', 'A', 'A'],
        ['B', 'A', 'A', 'B', 'B'],
        ['B', 'A', 'A', 'B', 'B'],
        ['A', 'B', 'B', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'B', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['A', 'A', 'B', 'B', 'B'],
        ['B', 'B', 'B', 'B', 'B'],
        ['A', 'A', 'B', 'A', 'A'],
        ['B', 'B', 'B', 'A', 'A'],
        ['B', 'B', 'A', 'A', 'B'],
        ['B', 'B', 'B', 'B', 'A'],
        ['B', 'A', 'B', 'A', 'B'],
        ['A', 'B', 'B', 'B', 'A'],
        ['A', 'B', 'A', 'B', 'B'],
        ['B', 'A', 'B', 'B', 'B'],
        ['A', 'B', 'B', 'B', 'B']
    ]
    y_train_degrees = ['SECOND', 'FIRST', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                    'SECOND', 'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND',
                    'SECOND', 'FIRST', 'SECOND', 'SECOND', 'SECOND', 'FIRST',
                    'SECOND', 'SECOND', 'SECOND', 'SECOND', 'FIRST', 'SECOND',
                    'SECOND', 'SECOND']
    
    X_test_degrees = [["B", "B", "B", "B", "B"], ["A", "A", "A", "A", "A"], ["A", "A", "A", "A", "B"]]
    tree_clf.fit(X_train_degrees,y_train_degrees)
    assert tree_clf.predict(X_test_degrees) == ["SECOND","FIRST","FIRST"]

    X_train_iphone = [
        ["1","3","fair"],
        ["1","3","excellent"],
        ["2","3","fair"],
        ["2","2","fair"],
        ["2","1","fair"],
        ["2","1","excellent"],
        ["2","1","excellent"],
        ["1","2","fair"],
        ["1","1","fair"],
        ["2","2","fair"],
        ["1","2","excellent"],
        ["2","2","excellent"],
        ["2","3","fair"],
        ["2","2","excellent"],
        ["2","3","fair"]
    ]
    y_train_iphone = ["no","no","yes","yes","yes","no","yes","no","yes","yes","yes","yes","yes","no","yes"]

    X_test_iphone = [["2","2","fair"],["1","1","excellent"]]
    tree_clf.fit(X_train_iphone,y_train_iphone)
    assert tree_clf.predict(X_test_iphone) == ["yes","yes"]
