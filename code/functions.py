from copy import deepcopy
from random import randint, sample
from itertools import chain
from math import ceil
import sys
sys.setrecursionlimit(1000000)

train_file = sys.argv[1]
test_file = sys.argv[2]

# Read in and format the data
def read_input(file, train = True):
    with open(file) as f:
        dat = f.readlines()
    dat = [x.rstrip() for x in dat if x.strip()]  # Remove end-of-line and empty lines
    for i in range(len(dat)):
        dat[i] = dat[i].split(" ")
    
    n_attr = len(dat[1]) - 1
    attr_n_level = [1] * n_attr
    
    labels = [0] * len(dat)
    features = [ [ 0 for x in range(n_attr) ] for y in range(len(dat)) ]
    
    for i in range(len(dat)):
        labels[i] = int(dat[i][0])
        for j in range(1, len(dat[i])):
            dat[i][j] = dat[i][j].split(":")
            attr = int(dat[i][j][0])
            level = int(dat[i][j][1])
            features[i][j - 1] = level
            if level > attr_n_level[attr - 1]:
                attr_n_level[attr - 1] = level
    
    n_label = len(set(labels))
    if train:
        return (n_attr, attr_n_level, n_label, labels, features)
    else:
        return (labels, features)

# Calculate Gini Index. "attr" is the attribute under consideration. attr = 0 if considering no partition.
def get_gini_index(n_attr, attr_n_level, n_label, labels, features, attr):
    if attr == 0:
        gini_index = 1.0
        for label in range(1, n_label + 1):
            gini_index -= (float(labels.count(label)) / len(labels)) ** 2
        return gini_index
    else:
        gini_index_a = 0.0
        gini_index_d = [1.0 for x in range(attr_n_level[attr - 1])]
        nd = [0 for x in range(len(gini_index_d))]
        for level in range(1, len(gini_index_d) + 1):
            counts = [0 for x in range(n_label)]
            for obs in range(len(labels)):
                if features[obs][attr - 1] == level:
                    counts[labels[obs] - 1] += 1
                    nd[level - 1] += 1
            if(nd[level - 1] != 0):
                for i in range(len(counts)):
                    gini_index_d[level - 1] -= (float(counts[i]) / nd[level - 1]) ** 2
        for level in range(1, len(gini_index_d) + 1):
            gini_index_a +=  float(nd[level - 1]) * gini_index_d[level - 1] / len(labels)
        return gini_index_a

# Majority vote in a leaf node to determine the leaf label
def leaf_node_vote(n_label, labels):
    labels_count = [0 for x in range(n_label)]
    for i in range(len(labels)):
        labels_count[labels[i] - 1] += 1
    return labels_count.index(max(labels_count)) + 1

# Build tree
def build_tree(n_attr, attr_n_level, n_label, labels, features, used_attr, rf = False):
    if len(labels) == 0:
        return -1
    if len(labels) == 1:
        return labels[0]
    if used_attr.count(True) == n_attr:
        return leaf_node_vote(n_label, labels)
    gini_index = get_gini_index(n_attr, deepcopy(attr_n_level), n_label, deepcopy(labels), deepcopy(features), 0)
    #print "Building subtree. Used attr before building: " + str(used_attr)
    #print "GI: " + str(gini_index)
    reduce_impurity = [-1 for x in range(n_attr)]
    reduce_impurity_max = -1
    reduce_impurity_which_max = -1
    end_flag = True
    
    if rf:
        cand_attr = []
        for attr in range(len(used_attr)):
            if not used_attr[attr]:
                cand_attr.append(attr + 1)
        n_select_attr = min(max([int(ceil(float(n_attr) / 2)), 2]), n_attr)
        if(len(cand_attr) > n_select_attr):
            select_attr_index = sample(range(len(cand_attr)), n_select_attr)
            masked_attr = [True for x in range(n_attr)]
            for i in range(len(masked_attr)):
                if i in select_attr_index:
                    masked_attr[i] = False
        else:
            masked_attr = deepcopy(used_attr)
        used_attr_store = used_attr
        used_attr = masked_attr
    
    for attr in range(n_attr):
        if not used_attr[attr]:
            gini_index_a = get_gini_index(n_attr, deepcopy(attr_n_level), n_label, deepcopy(labels), deepcopy(features), attr + 1)
            reduce_impurity[attr] = gini_index - gini_index_a
            if reduce_impurity[attr] > reduce_impurity_max:
                reduce_impurity_max = reduce_impurity[attr]
                reduce_impurity_which_max = attr
    if reduce_impurity_max <= 0:
        return leaf_node_vote(n_label, labels)
    else:
        subtree = [reduce_impurity_which_max + 1]
        subtree.extend([ [] for x in range(attr_n_level[reduce_impurity_which_max])])
        if(rf):
            used_attr_store[reduce_impurity_which_max] = True
        else:
            used_attr[reduce_impurity_which_max] = True
        for level in range(1, len(subtree)):
            labels_level = []
            features_level = []
            for obs in range(len(labels)):
                if features[obs][reduce_impurity_which_max] == level:
                    labels_level.extend([labels[obs]])
                    features_level.extend([features[obs]])
            if(rf):
                subtree[level] = build_tree(n_attr, deepcopy(attr_n_level), n_label, deepcopy(labels_level), deepcopy(features_level), deepcopy(used_attr_store), rf)
            else:
                subtree[level] = build_tree(n_attr, deepcopy(attr_n_level), n_label, deepcopy(labels_level), deepcopy(features_level), deepcopy(used_attr), rf)
        return deepcopy(subtree)
        
# Single-tree predictions
def predict_labels_tree(tree, features, n_label):
    pred = [0 for x in range(len(features))]
    for obs in range(len(features)):
        subtree = deepcopy(tree)
        while not (type(subtree) is int):
            goto = features[obs][subtree[0] - 1]
            subtree = deepcopy(subtree[goto])
        if subtree == -1:
            pred[obs] = randint(1, n_label)
        else:
            pred[obs] = subtree
    return deepcopy(pred)

# Ensemble prediction
def predict_labels_forest(forest, features, n_label):
    labels = [-1 for x in range(len(forest))]
    for i in range(len(labels)):
        labels[i] = predict_labels_tree(forest[i], features, n_label)
    pred = [0 for x in range(len(features))]
    for obs in range(len(features)):
        labels_count = [0 for x in range(n_label)]
        for i in range(len(labels)):
            labels_count[labels[i][obs] - 1] += 1
        pred[obs] = labels_count.index(max(labels_count)) + 1
    return deepcopy(pred)

# Get confusion matrix
def get_CM(pred, labels, n_label):
    CM = [ [0 for x in range(n_label)] for y in range(n_label) ]
    for obs in range(len(labels)):
        CM[labels[obs] - 1][pred[obs] - 1] += 1
    return deepcopy(CM)

# Print confusion matrix to standard output
def print_CM(CM):
    print "Confusion matrix:"
    for i in range(len(CM)):
        for j in range(len(CM)):
            if j < len(CM) - 1:
                print str(CM[i][j]) + " ",
            else:
                print str(CM[i][j])
    print ""

# Model evaluation
def evaluate(CM):
    overall_accuracy = 0
    TP = [0 for x in range(len(CM))]
    TN = [0 for x in range(len(CM))]
    FP = [0 for x in range(len(CM))]
    FN = [0 for x in range(len(CM))]
    for i in range(len(CM)):
        # Rows
        TP[i] = CM[i][i]
        FN[i] = sum(CM[i]) - TP[i]
    for i in range(len(CM)):
        # Columns
        for j in range(len(CM)):
            if i != j:
                FP[i] += CM[j][i]
    total_n = sum(list(chain.from_iterable(CM)))
    for i in range(len(CM)):
        TN[i] = total_n - TP[i] - FP[i] - FN[i]
    overall_accuracy = float(sum(TP)) / total_n
    print "Overall accuracy: " + str(overall_accuracy)
    print ""
    for i in range(len(CM)):
        if TP[i] == 0:
            sensitivity = precision = recall = f1 = f05 = f2 = 0
        else:
            sensitivity = float(TP[i]) / (float(TP[i]) + FN[i])
            precision = float(TP[i]) / (float(TP[i]) + FP[i])
            recall = float(TP[i]) / (float(TP[i]) + FN[i])
            f1 = 2 * float(TP[i]) / (2 * float(TP[i]) + float(FP[i]) + FN[i])
            beta = 0.5
            f05 = (1 + beta ** 2) * (float(precision) * recall / (beta ** 2 * float(precision)) + recall)
            beta = 2
            f2 = (1 + beta ** 2) * (float(precision) * recall / (beta ** 2 * float(precision)) + recall)
        if TN[i] == 0:
            specificity = 0
        else:
            specificity = float(TN[i]) / (float(FP[i]) + TN[i])

        print "For label level " + str(i + 1) + ":"
        print "Sensitivity: " + str(sensitivity)
        print "Specificity: " + str(specificity)
        print "Precision: " + str(precision)
        print "Recall: " + str(recall)
        print "F-1 score: " + str(f1)
        print "F-0.5 score: " + str(f05)
        print "F-2 score: " + str(f2)
        print ""
