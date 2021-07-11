import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer


def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred) - 1]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)


    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


"""
This function computes classwise performance of the classifier.
If dealing with training data, it will print the classwise training error.
If dealing with test data, it will print the classwise training accuracy.
"""


def calculate_classwise_performance(Y, Yp, label,
                                    error=False):
    # indices where label matches with original data
    idxs = np.argwhere(Y == label)

    # indices where predicted data and original data has same values for 'idxs'
    idxs_t = np.argwhere(Yp[idxs] == label)

    if error:
        return (idxs.shape[0] - idxs_t.shape[0]) / idxs.shape[0]
    else:
        return idxs_t.shape[0] / idxs.shape[0]


def grid_search(X, Y):
    # grid search over a range of C values and return the best params
    GSCVclfs = []
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
        GSCVclfs.append(clf)
    return GSCVclfs


"""
This function will generate 5 classfiers, each for one class with best 
parameter settings received from grid_search.
classes:[1,2,3,4,5] = C:[1,1,5,0.5,1]
"""


def classifier_generate(GSCVclfs, numClasses, x_train, y_train):
    C_best = [GSCVclfs[i].best_params_['C'] for i in range(numClasses)]
    K_clfs = []
    for k in range(numClasses):
        Y_copy = np.copy(y_train)
        label = k + 1
        Y_copy[Y_copy != label] = -1
        Y_copy[Y_copy == label] = 1
        clf = LinearSVC(C=C_best[k])
        clf.fit(x_train, Y_copy)
        K_clfs.append(clf)
    return K_clfs, C_best


"""
This function will 
- import the train/test data descriptor file
- seperate data and label vector
- randomize the indices
- transform (normalize) the data (X); label (Y) will be intact 
"""


def preprocessing(file_name):
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



trainFILE = 'trainDesc.pkl'
trainX_shuffled_normalized, trainY_shuffled = preprocessing(trainFILE)

# number of classes
numClasses = np.unique(trainY_shuffled).size

# -----------------Classifier with best parameter generate------------------

# grid search for best parameters (C)
GSCVclfs = grid_search(trainX_shuffled_normalized, trainY_shuffled)

# generate 5 classifiers with best parameter (C) settings for each K classes
K_clfs, C_best = classifier_generate(GSCVclfs, numClasses,
                                     trainX_shuffled_normalized, trainY_shuffled)

# --------------------Model Hyperparameters Calculate-----------------------

# model hyperparameters
W = np.array([K_clfs[i].coef_ for i in range(numClasses)])[:, -1, :]
B = np.array([K_clfs[i].intercept_ for i in range(numClasses)])
W_normalized = np.linalg.norm(W, axis=1)[..., np.newaxis]

# ---------------------Calculate Training Error-----------------------------

# calculate training error
trainX_shuffled_normalized_copy = trainX_shuffled_normalized.T
D_train = (W @ trainX_shuffled_normalized_copy + B) / W_normalized

# predict training data labels using model hyperparameters
trainY_predict = np.argmax(D_train, axis=0) + 1

# calculate training error
clf_train_error = [calculate_classwise_performance(trainY_shuffled,
                                                   trainY_predict, i + 1,
                                                   error=True)
                   for i in range(numClasses)]

# --------------------Performance Evaluation on Test Data-------------------

# import test data
testFILE = 'testDesc.pkl'
testX_shuffled_normalized, testY_shuffled = preprocessing(testFILE)

# transpose test data
testX_shuffled_normalized = testX_shuffled_normalized.T
testY_shuffled = testY_shuffled.astype(np.int64)

# decision matrix for test data
D_test = (W @ testX_shuffled_normalized + B) / W_normalized

# final prediction
testY_predict = np.argmax(D_test, axis=0) + 1

# test accuracy
clf_model_accuracy = [calculate_classwise_performance(testY_shuffled,
                                                      testY_predict, i + 1)
                      for i in range(numClasses)]

# ---------------------------Model Evaluation-------------------------------

# Print scores
np.set_printoptions(precision=3)
print("F1 Score:", f1_score(testY_shuffled, testY_predict,
                            average='weighted'))
print("Precision:", precision_score(testY_shuffled, testY_predict,
                                    average='weighted'))
print("Recall:", recall_score(testY_shuffled, testY_predict,
                              average='weighted'))
print("Overall Accuracy:", accuracy_score(testY_shuffled, testY_predict))
print("Classwise best C:", C_best)
print("Classwise Training Error:", clf_train_error)
print("Classwise Accuracy:", clf_model_accuracy)


class_names = np.array(['grass', 'ocean', 'redcarpet', 'road',
                        'wheatfield'])
plot_confusion_matrix(testY_shuffled, testY_predict, classes=class_names)

# --------------------------------------------------------------------------
