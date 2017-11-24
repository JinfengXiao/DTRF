from functions import *

ntree = 15

n_attr, attr_n_level, n_label, train_labels, train_features = read_input(train_file, True)
test_labels, test_features = read_input(test_file, False)
used_attr = [False for x in range(n_attr)]

forest = []
for itree in range(ntree):
    bootstrap_labels = []
    bootstrap_features = []
    for i in range(len(train_labels)):
        bootstrap_index = randint(0, len(train_labels) - 1)
        bootstrap_labels.append(train_labels[bootstrap_index])
        bootstrap_features.append(train_features[bootstrap_index])
    forest.append(build_tree(n_attr, deepcopy(attr_n_level), n_label, deepcopy(bootstrap_labels), deepcopy(bootstrap_features), deepcopy(used_attr), True))

pred = predict_labels_forest(deepcopy(forest), deepcopy(test_features), n_label)
CM = get_CM(deepcopy(pred), deepcopy(test_labels), n_label)
print_CM(deepcopy(CM))

evaluate(deepcopy(CM))
