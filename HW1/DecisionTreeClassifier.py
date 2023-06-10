
import numpy as np
import pandas as pd
import math
import os




class TreeNode:
    def __init__(self, data, output):
        self.data = data
        self.children = {}
        self.output = output
        self.index = -1

    def add_child(self, feature_value, obj):
        self.children[feature_value] = obj





class DecisionTreeClassifier:
    def __init__(self):
        self.__root = None



    def entropy(self, Y):

        mydict = {}
        for i in Y:
            if i not in mydict:
                mydict[i] = 1
            else:
                mydict[i] += 1
        ans= 0
        for i in mydict:
            ans += (-(mydict[i] / len(Y))) * math.log2((mydict[i] / len(Y)))
        return ans

    def gain_ratio(self, X, Y, selected_feature):
        info_orig = self.entropy(Y)
        info_f = 0
        split_info = 0
        values = set(X[:, selected_feature])
        df = pd.DataFrame(X)

        df[df.shape[1]] = Y
        for i in values:
            df1 = df[df[selected_feature] == i]
            info_f += (df1.shape[0] / df.shape[0]) * self.entropy(df1[df1.shape[1] - 1])
            split_info += (-df1.shape[0] / df.shape[0]) * math.log2(df1.shape[0] / df.shape[0])

        if split_info == 0:
            return math.inf

        info_gain = info_orig - info_f
        gain_ratio = info_gain / split_info
        return gain_ratio

    def gini_index(self, Y):
        mydict = {}
        for i in Y:
            if i not in mydict:
                mydict[i] = 1
            else:
                mydict[i] += 1
        gini_index_ = 1
        for i in mydict:
            gini_index_ -= (mydict[i] / len(Y)) ** 2
        return gini_index_

    def gini_gain(self, X, Y, s_feature):
        gini_orig = self.gini_index(Y)
        gini_split_f = 0
        df = pd.DataFrame(X)
        df[df.shape[1]] = Y
        for i in set(X[:, s_feature]):
            df1 = df[df[s_feature] == i]
            current_size = df1.shape[0]
            gini_split_f += (current_size / df.shape[0]) * self.gini_index(df1[df1.shape[1] - 1])

        gini_gain_ = gini_orig - gini_split_f
        return gini_gain_

    def decision_tree(self, X, Y, features, level, metric, classes):

        if len(features) == 0:
            mydict = {}
            for i in Y:
                if i not in mydict:
                    mydict[i] = 1
                else:
                    mydict[i] += 1
            output = None
            max_count = -math.inf
            for i in classes:
                if i not in mydict:
                    i
                else:
                    if mydict[i] > max_count:
                        output = i
                        max_count = mydict[i]

            if metric == "gain_ratio":
                self.entropy(Y)
            elif metric == "gini_index":
                self.gini_index(Y)

            print()
            return TreeNode(None, output)

        max_gain = -math.inf
        final_feature = None
        for f in features:
            if metric == "gain_ratio":
                current_gain = self.gain_ratio(X, Y, f)
            elif metric == "gini_index":
                current_gain = self.gini_gain(X, Y, f)

            if current_gain > max_gain:
                max_gain = current_gain
                final_feature = f

        print("Level", level)
        mydict = {}
        for i in Y:
            if i not in mydict:
                mydict[i] = 1
            else:
                mydict[i] += 1
        output = None
        max_count = -math.inf

        for i in classes:
            if i not in mydict:

                print("Count of", i, "=", 0)
            else:
                if mydict[i] > max_count:
                    output = i
                    max_count = mydict[i]
                print("Count of", i, "=", mydict[i])

        if metric == "gain_ratio":
            print("Current Entropy is =", self.entropy(Y))
            print("Splitting on feature  X[", final_feature, "] with gain ratio ", max_gain, sep="")
            print()
        elif metric == "gini_index":
            print("Current Gini Index is =", self.gini_index(Y))
            print("Splitting on feature  X[", final_feature, "] with gini gain ", max_gain, sep="")
            print()


        df = pd.DataFrame(X)
        df[df.shape[1]] = Y

        current_node = TreeNode(final_feature, output)

        index = features.index(final_feature)
        features.remove(final_feature)
        for i in set(X[:, final_feature]):
            df1 = df[df[final_feature] == i]

            node = self.decision_tree(df1.iloc[:, 0:df1.shape[1] - 1].values, df1.iloc[:, df1.shape[1] - 1].values,
                                      features, level + 1, metric, classes)
            current_node.add_child(i, node)

        features.insert(index, final_feature)

        return current_node

    def fit(self, X, Y, metric="gain_ratio"):

        features = [i for i in range(len(X[0]))]
        classes = set(Y)
        level = 0
        if metric != "gain_ratio":
            if metric != "gini_index":
                metric = "gain_ratio"

        self.__root = self.decision_tree(X, Y, features, level, metric, classes)

    def predict_for(self, data, node):

        if len(node.children) == 0:
            return node.output

        val = data[node.data]
        if val not in node.children:
            return node.output

        return self.predict_for(data, node.children[val])

    def predict(self, X):

        Y = np.array([0 for i in range(len(X))])


        for i in range(len(X)):
            Y[i] = self.predict_for(X[i], self.__root)
        return Y

    def score(self, X, Y):
        Y_pred = self.predict(X)
        count = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y[i]:
                count += 1
        return count / len(Y_pred)




'''
attribute = ['SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','FATIGUE ','ALLERGY ',' WHEEZING',
'ALCOHOL CONSUMING','COUGHING','SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN,LUNG_CANCER']

'''

#LUNG_CANCER
dataset = pd.read_csv('C:/Users/jahannama/Downloads/lungcancer.csv')
dataset['LUNG_CANCER'] = dataset['LUNG_CANCER'].replace(['YES'],2)
dataset['LUNG_CANCER'] = dataset['LUNG_CANCER'].replace(['NO'],1)

dataset['GENDER'] = dataset['GENDER'].replace(['M'],1)
dataset['GENDER'] = dataset['GENDER'].replace(['F'],2)

X_train=dataset.sample(frac = 0.75,replace = False)
X_test = pd.DataFrame(dataset, index = set(dataset.index).difference(set(X_train.index))).reset_index()
X_train__=X_train.iloc[:, 0:X_train.shape[1] - 1].values
Y_train__=X_train.iloc[:, X_train.shape[1] - 1].values

X_test__=X_test.iloc[:, 0:X_test.shape[1] - 1].values
Y_test__=X_test.iloc[:, X_test.shape[1] - 1].values




for i in range(0,len(X_train__)):
    if X_train__[i][1] >= 60:
        X_train__[i][1] = 2
    else :
        X_train__[i][1]=1


for i in range(0,len(X_test__)):
    if X_test__[i][2] >= 60:
        X_test__[i][2] = 2
    else :
        X_test__[i][2]=1

print(len(Y_train__))




X_test__=np.delete(X_test__, 0, 1)

X=X_train__
Y=Y_train__

clf2 = DecisionTreeClassifier()
clf2.fit(X, Y, metric='gini_index')
Y_pred2 = clf2.predict(X)
print()
our_score_on_train = clf2.score(X, Y)
print()
X=X_test__
Y=Y_test__
Y_pred2 = clf2.predict(X)
our_score_on_test = clf2.score(X, Y)
X=X_train__
Y=Y_train__

import sklearn.tree
clf3 = sklearn.tree.DecisionTreeClassifier()
clf3.fit(X, Y)
Y_pred3 = clf3.predict(X)
sklearn_score = clf3.score(X, Y)

print("Score of our model on train :", our_score_on_train)
print("Score of inbuilt sklearn's decision tree on the same data :", sklearn_score)


X=X_test__
Y=Y_test__

clf3 = sklearn.tree.DecisionTreeClassifier()
clf3.fit(X, Y)
Y_pred3 = clf3.predict(X)
sklearn_score = clf3.score(X, Y)


print("Score of our model on test :", our_score_on_test)
print("Score of inbuilt sklearn's decision tree on the same data :", sklearn_score)


