# DTRF
This repo contains a **Python 2.7** implementation of **decision trees** and **random forests** for binary or multi-class classfication. Currently the code works with **categorical attributes** only.

This repo was part of the author's solution to a programming assignment in 2016. You are welcome to use it in accordance with the license agreement and academic integrity.

## Usage

To train a **decision tree** on a `training_file` and evaluate its performance on a `test_file`, run the following:

```
python code/DecisionTree.py training_file test_file
```

For example, to run on the sample data provided in this repo:

```
python code/DecisionTree.py sample_data/poker.train sample_data/poker.test
```

To train a **random forest** on a `training_file` and evaluate its performance on a `test_file`, run the following:

```
python code/RandomForest.py training_file test_file
```

## Input
The input files `training_file` and `test_file` is expected to be in the [LIBSVM format](http://www.csie.ntu.edu.tw/~cjlin/liblinear/). Each line contains an instance and is eneded by a `\n` character. Each line is in the following format:

```
<label> <index1>:<value1> <index2>:<value2> ...
```

`<label>` is an integer indicating the class label. The pair `<index>:<value>` gives a feature (attribute) value: `<index>` is an integer starting from 1 and `<value>` is a number (we only consider categorical attributes in this assignment). Note that one attribute may have more than 2 possible values, meaning it is a multi-value categorical attribute.

An example pair of training and testing files are provided in `sample_data`. Those are subsets of the [Poker Hand Data Set](http://archive.ics.uci.edu/ml/datasets/Poker+Hand) processed by Fangbo Tao and/or Huan Gui.

## Output

A decision tree or random forest will be trained on `training_file` and evaluated on `test_file`. The following evaluation information will be printed to the screen:

- Confusion matrix
  * In i-th line, the j-th number is the number of data points in test where the actual label is i and predicted label is j.
- Overall class prediction accuracy
- For each class:
  * Sensitivity
  * Specificity
  * Precision
  * Recall
  * F<sub>1</sub> score
  * F<sub>0.5</sub> score
  * F<sub>2</sub> score

