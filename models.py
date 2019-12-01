from sklearn.model_selection import train_test_split
from sklearn import svm
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import RidgeClassifier


os.makedirs('models', exist_ok=True)
os.makedirs('learningcurves', exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'


    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print(f"Normalized confusion matrix of {model}")
    # else:
    #     print(f'Confusion matrix, without normalization of {model}')

    # print(classes)
    # print(cm)
    st.markdown(f'## {title} ')
    calc = cm.trace()/cm.sum()
    cm = cm.tolist()
    df = pd.DataFrame({c:row for c,row in zip(classes, cm)},index=classes)
    st.table(df)

    st.markdown(f'**Accuracy:** {calc}')


    return cm

def train_svm(X,Y, label_encoder, model, model_name):
    le = label_encoder
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.10, random_state=42)

    classifier = model
    # print(y_train[0], type(y_train[0]), X_train[0], type(X_train[0]))
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)




    # Plot non-normalized confusion matrix
    # cm = plot_confusion_matrix(y_test, y_pred, classes=le.classes_,
    #                     title=model_name + ' without normalization')

    # Plot normalized confusion matrix
    cmn = plot_confusion_matrix(y_test, y_pred, classes=le.classes_, normalize=True,
                        title=model_name + ' Normalized confusion matrix')




    # return (cm, cmn)



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
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
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def learning_curves_preprocess(estimator, title, X, y, n_splits=10):


    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=0)

    plt = plot_learning_curve(estimator, title, X, y, cv=cv, n_jobs=4)
    plt.savefig(os.path.join('learningcurves', title + '.png'))

    # plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)

    return plt


def test_cases(vectorizer, X, Y, model, le):




    linearsvc = svm.LinearSVC()

    title = 'LinearSVC'

    name = '{vectorizer} + {title}'


    train_svm(X,Y, le, linearsvc, name.format(title=title, vectorizer=vectorizer))

    if st.checkbox(f'Plot Learning Curves for {title} + {vectorizer}'):
        plot = learning_curves_preprocess(svm.LinearSVC(), name.format(title=title, vectorizer=vectorizer), X, Y)

        st.pyplot(plot)


    st.markdown('****')

    title = 'GaussianNB'
    estimator = GaussianNB()


    train_svm(X,Y, le, estimator, name.format(title=title, vectorizer=vectorizer))

    if st.checkbox('Plot Learning Curves' + name.format(title=title, vectorizer=vectorizer)):
        plot = learning_curves_preprocess(estimator, name.format(title=title, vectorizer=vectorizer), X, Y)

        st.pyplot(plot)


    st.markdown('****')

    title = 'SVC'
    if st.checkbox('Show ' +  name.format(title=title, vectorizer=vectorizer)):
        estimator = svm.SVC()


        train_svm(X,Y, le, estimator, name.format(title=title, vectorizer=vectorizer))

        if st.checkbox('Plot Learning Curves ' + name.format(title=title, vectorizer=vectorizer)):
            plot = learning_curves_preprocess(estimator, name.format(title=title, vectorizer=vectorizer), X, Y)

            st.pyplot(plot)


    st.markdown('****')


    title = 'mlpclassifier'
    estimator = MLPClassifier()


    train_svm(X,Y, le, estimator, name.format(title=title, vectorizer=vectorizer))

    if st.checkbox('Plot Learning Curves ' + name.format(title=title, vectorizer=vectorizer)):
        plot = learning_curves_preprocess(estimator, name.format(title=title, vectorizer=vectorizer), X, Y)

        st.pyplot(plot)


    st.markdown('****')



    title = 'randomForest'
    estimator = RandomForestClassifier()


    train_svm(X,Y, le, estimator, name.format(title=title, vectorizer=vectorizer))

    if st.checkbox('Plot Learning Curves ' + name.format(title=title, vectorizer=vectorizer)):
        plot = learning_curves_preprocess(estimator, name.format(title=title, vectorizer=vectorizer), X, Y)

        st.pyplot(plot)

    st.markdown('****')



    title = 'RidgeClassifier'
    estimator = RidgeClassifier()


    train_svm(X,Y, le, estimator, name.format(title=title, vectorizer=vectorizer))

    if st.checkbox('Plot Learning Curves ' + name.format(title=title, vectorizer=vectorizer)):
        plot = learning_curves_preprocess(estimator, name.format(title=title, vectorizer=vectorizer), X, Y)

        st.pyplot(plot)

    st.markdown('****')


# plt.show()