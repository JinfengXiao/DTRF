from functions import *


n_attr, attr_n_level, n_label, train_labels, train_features = read_input(train_file, True)
test_labels, test_features = read_input(test_file, False)


used_attr = [False for x in range(n_attr)]
tree = build_tree(n_attr, deepcopy(attr_n_level), n_label, deepcopy(train_labels), deepcopy(train_features), deepcopy(used_attr), False)

pred = predict_labels_tree(deepcopy(tree), deepcopy(test_features), n_label)
CM = get_CM(deepcopy(pred), deepcopy(test_labels), n_label)
print_CM(deepcopy(CM))

evaluate(deepcopy(CM))
