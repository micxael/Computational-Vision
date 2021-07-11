import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer


def grid_search(X, Y):
    # grid search over a range of C values and return the best params

    params = []
    parameters = {'C': np.linspace(0.1, 10, 100, endpoint=True)}
    # numClasses = 5
    for k in range(5):
        Y_copy = np.copy(Y)
        label = k + 1
        Y_copy[Y_copy != label] = -1
        Y_copy[Y_copy == label] = 1
        svc = LinearSVC()
        clf = GridSearchCV(svc, parameters, cv=5)
        clf.fit(X, Y_copy)
        params.append(clf)
    return params


def generate_classifier(params, numClasses, x_train, y_train):
    # generate 5 classifiers with best param settings from grid_search

    C_best = [params[i].best_params_['C'] for i in range(numClasses)]
    clfs = []
    for k in range(numClasses):
        Y_copy = np.copy(y_train)
        label = k + 1
        Y_copy[Y_copy != label] = -1
        Y_copy[Y_copy == label] = 1
        clf = LinearSVC(C=C_best[k])
        clf.fit(x_train, Y_copy)
        clfs.append(clf)
    return clfs, C_best


def preprocessing(file_name):
    # data preprocessor
    # - pseudo random indices
    # - normalize the data
    with open(file_name, 'rb') as f:
        Data = pickle.load(f)

    x = Data[..., :-1]
    y = Data[..., -1]
    np.random.seed(1)
    np.random.shuffle(np.arange(x.shape[0]))

    x_random = x[np.arange(x.shape[0])]
    y_random = y[np.arange(x.shape[0])]

    x_norm = Normalizer().fit(x_random).transform(x_random)

    return x_norm, y_random


def calculate_performance(data, data_predicted, label, error=False):
    # calculates performance of the classifier

    # labal = original data
    idx = np.argwhere(data == label)

    # predicted data = original data
    idx_ = np.argwhere(data_predicted[idx] == label)

    if error:
        return (idx.shape[0] - idx_.shape[0]) / idx.shape[0]
    else:
        return idx_.shape[0] / idx.shape[0]


def train():
    x_train, trainY_shuffled = preprocessing('train.pkl')
    numClasses = np.unique(trainY_shuffled).size

    # grid search for the best parameters
    params = grid_search(x_train, trainY_shuffled)

    # generate 5 classifiers with best parameter settings for each K classes
    clfs, C_best = generate_classifier(params, numClasses, x_train, trainY_shuffled)
    # print("Best classifiers by image class:", C_best)

    # hyperparameters
    W = np.array([clfs[i].coef_ for i in range(numClasses)])[:, -1, :]
    B = np.array([clfs[i].intercept_ for i in range(numClasses)])
    w_norm = np.linalg.norm(W, axis=1)[..., np.newaxis]

    # calculate training error
    trainX_shuffled_normalized_copy = x_train.T
    D_train = (W @ trainX_shuffled_normalized_copy + B) / w_norm

    # predict training data labels using hyperparameters
    trainY_predict = np.argmax(D_train, axis=0) + 1

    # calculate training error
    clf_train_error = [calculate_performance(trainY_shuffled, trainY_predict, i + 1, error=True)
                       for i in range(numClasses)]
    print("Training Error:", clf_train_error)

    return W, B, w_norm, numClasses


def test(W, B, W_normalized, numClasses):
    # evaluation on test data
    x_test, y_test = preprocessing('test.pkl')

    x_test = x_test.T
    y_test = y_test.astype(np.int64)

    # decision matrix for test data
    decision_test = (W @ x_test + B) / W_normalized

    # final prediction
    y_pred = np.argmax(decision_test, axis=0) + 1

    # test accuracy
    clf_test_accuracy = [calculate_performance(y_test, y_pred, i + 1)
                          for i in range(numClasses)]
    print("Test accuracy:", clf_test_accuracy)

    # Print scores
    print("\nOverall precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Overall accuracy:", accuracy_score(y_test, y_pred))

    # print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n", cm)

    # plot confusion matrix
    # depricated ?
    # disp = ConfusionMatrixDisplay(cm)
    # disp.plot()
    # plot_confusion_matrix(cm, ['grass', 'ocean', 'redcarpet', 'road', 'wheatfield'])
    # plt.show()


np.set_printoptions(precision=3)
W, B, W_norm, numClasses = train()  # returns hyperparameters
test(W, B, W_norm, numClasses)
sys.exit()
