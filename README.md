
# Non-Parameteric

Code for K-Nearest Neighbor algorithm (a non-parametric modeling technique)

---

## Theory

Please note that the theory corresponding to the codes are hosted on [[link]]

## Python Environment Setup

Ensure the following is installed in your pyhton environment:

- Numpy - required for vectorized operations
- Scikit-learn - For loading the IRIS dataset
- Matploitlib - required for plotting

## KNN

Execute the following to get a classification model using K-Nearest Neighbor algorithm:

```bash
python main.py <k>
```

After training: you should get plots like the following:
Train             |  &nbsp;Test&nbsp;
:-------------------------:|:-------------------------:
![A plot showing how the model fits the data on the train set](static/decision_boundary.png "KNN Train Fit")  |  ![A plot showing how the model fits the data on the test set](static/decision_boundary_test.png "KNN Test Fit")

The accuracy on the test split should be >95%:

```bash
$ python main.py 5
Test Accuracy: 0.967
```

Experiment with various values of `k`
