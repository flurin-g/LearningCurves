import argparse
from collections import namedtuple
from typing import Tuple, List

import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

# from sklearn.model_selection import cross_validation
# from yellowbrick.model_selection import LearningCurve

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

from sklearn.datasets import load_digits
from sklearn.datasets import load_iris


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv-folds', type=int, default=10)
    parser.add_argument('--n-points', type=int, default=20)
    parser.add_argument('--classifier', type=int, default=3)
    parser.add_argument('--dataset', type=int, default=0)
    args = parser.parse_args()
    return args


HyperParameters = namedtuple('HyperParameters',
                             ['cv_folds',
                              'points_on_learning_curve',
                              'which_classifier',
                              'which_dataset'])


def setup_experiments() -> Tuple[list, list]:
    params_naive_bayes = {}  # naive Bayes is parameter-free
    tune_naive_bayes = []

    params_knn = {'algorithm': 'ball_tree', 'n_neighbors': 2, 'weights': 'distance'}
    tune_knn = {'n_neighbors': [1, 4, 8, 16, 32, 64, 96, 128]}
    # tune_knn =   {'n_neighbors': [1,2,3,4,5,6,7,8]}

    params_svm_linear = {'kernel': 'linear', 'C': 1}
    tune_svm_linear = {'C': [1E-5, 1E-4, 1E-3, 0.01, 0.1, 1, 10, 100]}

    params_svm_rbf = {'kernel': 'rbf', 'gamma': 0.01, 'C': 1}
    tune_svm_rbf = {'C': [1E-5, 1E-4, 1E-3, 0.01, 0.1, 1, 10, 100],
                    'gamma': [1E-5, 1E-4, 1E-3, 0.01, 0.1, 1, 10, 100]}

    params_random_forest = {'n_estimators': 80}
    tune_random_forest = {'n_estimators': [1, 2, 4, 8, 10, 20, 40, 80, 100, 200, 400, 800]}

    estimators = [{'name': 'Naive Bayes', 'classifier': GaussianNB(), 'to_tune': tune_naive_bayes},
                  {'name': 'kNN', 'classifier': KNeighborsClassifier(**(params_knn)), 'to_tune': tune_knn},
                  {'name': 'Linear SVM', 'classifier': SVC(**(params_svm_linear)), 'to_tune': tune_svm_linear},
                  # ** is the keyword argument unpacking syntax
                  {'name': 'RBF SVM', 'classifier': SVC(**(params_svm_rbf)), 'to_tune': tune_svm_rbf},
                  {'name': 'Random Forest', 'classifier': RandomForestClassifier(**(params_random_forest)),
                   'to_tune': tune_random_forest}]

    datasets = [{'name': 'Digits', 'loader': load_digits},
                {'name': 'Iris', 'loader': load_iris}]

    return estimators, datasets


def plot_learning_curve(ax, estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    ax.grid(True)

    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                    train_scores_mean + train_scores_std, alpha=0.1,
                    color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                    test_scores_mean + test_scores_std, alpha=0.1, color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")
    ax.set_title(title)
    ax.legend(loc="best")

    return


def load_data(datasets: List[dict], which_dataset: int) -> Tuple[np.ndarray, np.ndarray, str]:
    print("Loading data")
    name = datasets[which_dataset]['name']
    data = datasets[which_dataset]['loader']()
    X, y = data.data, data.target
    return X, y, name


def print_dataset_info(name: str, X: np.ndarray) -> None:
    msg = (f"Name={name},\n"
           f"Number of samples: N={X.shape[0]},\n"
           f"Number of predictors: p={X.shape[1]}")
    print(msg)


def print_classifier_info(estimator: dict) -> None:
    print(f"Classifier: {estimator['name']}")
    plots = 1 + len(estimator['to_tune'])
    print(f"Hyper-parameter values to plot: {plots}")


def grid_learning_curves(estimator: dict, X: np.ndarray, y: np.ndarray, h_params: HyperParameters, cv) -> None:
    """
    Plots a learning curve for each value in estimator['to_tune']
    """

    for tunable_param, values in estimator['to_tune'].items():
        # loop through all parameters

        # set standard parameters if the other is modified
        if h_params.which_classifier == 3:
            setattr(estimator['classifier'], 'gamma', 0.01)
            setattr(estimator['classifier'], 'C', 1)

        fig, axs = plt.subplots(2, np.int32(np.ceil(len(values) / 2)), figsize=(16, 12))
        axs = axs.flatten()

        for ax, tunable_value in zip(axs, values):
            # loop through all values

            setattr(estimator['classifier'], tunable_param, tunable_value)

            title = "LC %s %s=%.1e" % (estimator['name'], tunable_param, tunable_value)

            plot_learning_curve(ax, estimator['classifier'], title, X, y, ylim=None, cv=cv,
                                n_jobs=-1, train_sizes=np.linspace(.1, 1.0, h_params.points_on_learning_curve))

        fn = "Learning_Curve_%s_%s.png" % (estimator['name'], tunable_param)
        plt.savefig(fn, dpi=600)


def single_learning_curve(estimator: dict, X: np.ndarray, y: np.ndarray, h_params: HyperParameters, cv) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    title = "LC %s" % (estimator['name'])
    plot_learning_curve(ax, estimator['classifier'], title, X, y, ylim=None, cv=cv,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, h_params.points_on_learning_curve))
    fn = "Learning_Curve_%s.png" % (estimator['name'])
    plt.savefig(fn, dpi=600)


def train_classifier(estimators: List[dict], X: np.ndarray, y: np.ndarray, h_params: HyperParameters) -> None:
    print("Training classifier")
    my_estimator = estimators[h_params.which_classifier]

    print_classifier_info(my_estimator)

    cv = ShuffleSplit(n_splits=h_params.cv_folds,
                      test_size=0.2, random_state=0)

    if len(my_estimator['to_tune']) > 0:
        grid_learning_curves(my_estimator, X, y, h_params, cv)
    else:
        single_learning_curve(my_estimator, X, y, h_params, cv)


def run_experiment(h_params: HyperParameters):
    estimators, datasets = setup_experiments()
    X, y, name = load_data(datasets, h_params.which_dataset)
    print_dataset_info(name, X)

    train_classifier(estimators, X, y, h_params)


if __name__ == "__main__":
    args = get_args()

    h_params = HyperParameters(cv_folds=args.cv_folds,
                               points_on_learning_curve=args.n_points,
                               which_classifier=args.classifier,
                               which_dataset=args.dataset)
    run_experiment(h_params)
