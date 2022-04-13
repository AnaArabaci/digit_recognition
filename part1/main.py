import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.append("..")
from utils import *
from part1.linear_regression import *
from part1.svm import *
from part1.softmax import *
from part1.features import *
from part1.kernel import *


# trees and ensemble methods
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# svm and kernels
from sklearn import svm


#######################################################################
# 1. Load data
#######################################################################

# Load MNIST data:
train_x, train_y, test_x, test_y = get_MNIST_data()

# Plot the first 20 images of the training set.
plot_images(train_x[0:20, :])




#######################################################################
# 9. Kernels on SVM
#######################################################################

# # Linear
#
# Cs = [0.75, 0.5, 0.25, 0.1, 0.01]
#
# train_accs = [0,0,0,0,0]
# val_accs = [0,0,0,0,0]
#
# for i, val in enumerate(Cs):
#
#     clf = svm.SVC(C=val, kernel='linear')
#     clf.fit(train_x, train_y)
#
#     train_accs[i]=clf.score(train_x, train_y)
#     val_accs[i]=clf.score(test_x, test_y)
#
# print("Training accuracies:", train_accs)
# print("Validation accuracies:", val_accs)
#
# # Polynomial
# clf = svm.SVC(C=1.0, kernel='poly', degree=3, gamma=2)
# clf.fit(train_x, train_y)
#
# PolyK_train_accuracy=clf.score(train_x, train_y)
# PolyK_test_accuracy=clf.score(test_x, test_y)
#
# print("{:50} {:.4f}".format("Training accuracy for SVM w/ Polynomial Kernel:", PolyK_train_accuracy))
# print("{:50} {:.4f}".format("Validation accuracy for SVM w/ Polynomial Kernel:", PolyK_test_accuracy))


# RBF

Cs = [0.5, 0.1]
gammas = [1.0, 0.1]

train_accs = [[0,0],[0,0]]
val_accs = [[0,0],[0,0]]

for i, val in enumerate(Cs):
    for j, v in enumerate(gammas):

        clf = svm.SVC(C=val, kernel='rbf', gamma= v)
        clf.fit(train_x, train_y)

        train_accs[i][j]=clf.score(train_x, train_y)
        val_accs[i][j]=clf.score(test_x, test_y)

print("Training accuracies:", train_accs)
print("Validation accuracies:", val_accs)

# Sigmoid
clf = svm.SVC(C=1.0, kernel='sigmoid')
clf.fit(train_x, train_y)

SigmK_train_accuracy=clf.score(train_x, train_y)
SigmK_test_accuracy=clf.score(test_x, test_y)

print("{:50} {:.4f}".format("Training accuracy for SVM w/ Sigmoid Kernel:", SigmK_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for SVM w/ Sigmoid Kernel:", SigmK_test_accuracy))



#######################################################################
# 8. Ensemble methods
#######################################################################


## AdaBoost ##

dt=DecisionTreeClassifier(max_depth=2, random_state=1)
clf = AdaBoostClassifier(base_estimator=dt, n_estimators=100, random_state=0)
clf.fit(train_x, train_y)


AB_train_accuracy=clf.score(train_x, train_y)
AB_test_accuracy=clf.score(test_x, test_y)

print("{:50} {:.4f}".format("Training accuracy for AdaBoost:", AB_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for AdaBoost:", AB_test_accuracy))


# For AdaBoost:
#
#   n_estimators - the number of weak learners
#   learning_rate - controls the contribution of the weak learners in the final combination
#
#   By default, weak learners are decision stumps XXX ??
#   Different weak learners can be specified through the **base_estimator** parameter.

# For decision tree:
#
#   max_depth - If None, then nodes are expanded until all leaves are pure or
#   until all leaves contain less than min_samples_split samples
#
#   min_samples (default=2)
#   the minimum number of samples required to split an internal node
#
#   min_samples_leaf (default=1)
#   the minimum number of samples required to be at a leaf node
#
#   random_state - controls the random number generator used
#   (features are always randomly permuted at each split)


## Gradient boosting ##

clf2 = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf2.fit(train_x, train_y)

GB_train_accuracy=clf2.score(train_x, train_y)
GB_test_accuracy=clf2.score(test_x, test_y)

print("{:50} {:.4f}".format("Training accuracy for Gradient Boosting:", GB_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Gradient Boosting:", GB_test_accuracy))



## Random Forest ##

clf1 = RandomForestClassifier(n_estimators=100, random_state=0)
clf1.fit(train_x, train_y)

RF_train_accuracy=clf1.score(train_x, train_y)
RF_test_accuracy=clf1.score(test_x, test_y)

print("{:50} {:.4f}".format("Training accuracy for Random Forest:", RF_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Random Forest:", RF_test_accuracy))


#######################################################################
# 2. Linear Regression with Closed Form Solution
#######################################################################


def run_linear_regression_on_MNIST(lambda_factor=0.01):
    """
    Trains linear regression, classifies test data, computes test error on test set

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_x_bias = np.hstack([np.ones([train_x.shape[0], 1]), train_x])
    test_x_bias = np.hstack([np.ones([test_x.shape[0], 1]), test_x])
    theta = closed_form(train_x_bias, train_y, lambda_factor)
    test_error = compute_test_error_linear(test_x_bias, test_y, theta)
    return test_error


print('Linear Regression test_error =', run_linear_regression_on_MNIST(lambda_factor=0.01))


#######################################################################
# 3. Support Vector Machine
#######################################################################

def run_svm_one_vs_rest_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y[train_y != 0] = 1
    test_y[test_y != 0] = 1
    pred_test_y = one_vs_rest_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


print('SVM one vs. rest test_error:', run_svm_one_vs_rest_on_MNIST())


def run_multiclass_svm_on_MNIST():
    """
    Trains svm, classifies test data, computes test error on test set

    Returns:
        Test error for the binary svm
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    pred_test_y = multi_class_svm(train_x, train_y, test_x)
    test_error = compute_test_error_svm(test_y, pred_test_y)
    return test_error


print('Multiclass SVM test_error:', run_multiclass_svm_on_MNIST())

#######################################################################
# 4. Multinomial (Softmax) Regression and Gradient Descent
#######################################################################


def run_softmax_on_MNIST(temp_parameter=1):
    """
    Trains softmax, classifies test data, computes test error, and plots cost function

    Runs softmax_regression on the MNIST training set and computes the test error using
    the test set. It uses the following values for parameters:
    alpha = 0.3
    lambda = 1e-4
    num_iterations = 150

    Saves the final theta to ./theta.pkl.gz

    Returns:
        Final test error
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    theta, cost_function_history = softmax_regression(train_x, train_y, temp_parameter, alpha=0.3, lambda_factor=1.0e-4, k=10, num_iterations=150)
    title="Softmax on MNIST"
    plot_cost_function_over_time(cost_function_history, title)

    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)
    test_error = compute_test_error_mod3(test_x, test_y_mod3, theta, temp_parameter)

    return test_error


print('softmax test_error=', run_softmax_on_MNIST(temp_parameter=1))

#######################################################################
# 6. Changing Labels
#######################################################################



def run_softmax_on_MNIST_mod3(temp_parameter=1):
    """
    Trains Softmax regression on digit (mod 3) classifications.

    See run_softmax_on_MNIST for more info.
    """
    train_x, train_y, test_x, test_y = get_MNIST_data()
    train_y_mod3, test_y_mod3 = update_y(train_y, test_y)
    theta, cost_function_history = softmax_regression(train_x, train_y_mod3, temp_parameter, alpha=0.3, lambda_factor=1.0e-4,
                                                      k=10, num_iterations=150)
    title = "Softmax on MNIST - mod3"
    plot_cost_function_over_time(cost_function_history, title)
    test_error = compute_test_error(test_x, test_y_mod3, theta, temp_parameter)
    return test_error


print('softmax test_error mod3=', run_softmax_on_MNIST_mod3(temp_parameter=1))


#######################################################################
# 7. Classification Using Manually Crafted Features
#######################################################################

## Dimensionality reduction via PCA ##

# TODO: First fill out the PCA functions in features.py as the below code depends on them.


n_components = 18

###Correction note:  the following 4 lines have been modified since release.
train_x_centered, feature_means = center_data(train_x)
pcs = principal_components(train_x_centered)
train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)

# train_pca (and test_pca) is a representation of our training (and test) data
# after projecting each example onto the first 18 principal components.


# TODO: Train your softmax regression model using (train_pca, train_y)
#       and evaluate its accuracy on (test_pca, test_y).

def run_softmax_on_MNIST_pca(temp_parameter=1):

    n_components = 18
    pcs = principal_components(train_x)
    train_pca = project_onto_PC(train_x, pcs, n_components, feature_means)
    test_pca = project_onto_PC(test_x, pcs, n_components, feature_means)
    # train_pca (and test_pca) is a representation of our training (and test) data
    # after projecting each example onto the first 18 principal components.


    # TODO: Train your softmax regression model using (train_pca, train_y)
    #       and evaluate its accuracy on (test_pca, test_y).

    theta, cost_function_history = softmax_regression(train_pca, train_y, temp_parameter, alpha= 0.3, lambda_factor = 1.0e-4, k = 10, num_iterations = 150)
    title = "Softmax on MNIST - pca"
    plot_cost_function_over_time(cost_function_history, title)
    test_error = compute_test_error(test_pca, test_y, theta, temp_parameter)

    # TODO: Use the plot_PC function in features.py to produce scatterplot
    #       of the first 100 MNIST images, as represented in the space spanned by the
    #       first 2 principal components found above.
    plot_PC(train_x[range(100),], pcs, train_y[range(100)], feature_means)


    # TODO: Use the reconstruct_PC function in features.py to show
    #       the first and second MNIST images as reconstructed solely from
    #       their 18-dimensional principal component representation.
    #       Compare the reconstructed images with the originals.
    firstimage_reconstructed = reconstruct_PC(train_pca[0, ], pcs, n_components, train_x, feature_means)
    plot_images(firstimage_reconstructed)
    plot_images(train_x[0, ])

    secondimage_reconstructed = reconstruct_PC(train_pca[1, ], pcs, n_components, train_x, feature_means)
    plot_images(secondimage_reconstructed)
    plot_images(train_x[1, ])

    # plot the 1st 18th for comparison
    nimage_reconstructed = reconstruct_PC(train_pca[0:n_components, ], pcs, n_components, train_x, feature_means)
    plot_images(nimage_reconstructed[0:18, ])

    return test_error


print('softmax test_error_pca=', run_softmax_on_MNIST_pca(temp_parameter=1))


# TODO: Use the plot_PC function in features.py to produce scatterplot
#       of the first 100 MNIST images, as represented in the space spanned by the
#       first 2 principal components found above.
plot_PC(train_x[range(000, 100), ], pcs, train_y[range(000, 100)], feature_means)#feature_means added since release


# TODO: Use the reconstruct_PC function in features.py to show
#       the first and second MNIST images as reconstructed solely from
#       their 18-dimensional principal component representation.
#       Compare the reconstructed images with the originals.
firstimage_reconstructed = reconstruct_PC(train_pca[0, ], pcs, n_components, train_x, feature_means)#feature_means added since release
plot_images(firstimage_reconstructed)
plot_images(train_x[0, ])

secondimage_reconstructed = reconstruct_PC(train_pca[1, ], pcs, n_components, train_x, feature_means)#feature_means added since release
plot_images(secondimage_reconstructed)
plot_images(train_x[1, ])


## Cubic Kernel ##
# TODO: Find the 10-dimensional PCA representation of the training and test set


# TODO: First fill out cubicFeatures() function in features.py as the below code requires it.

train_cube = cubic_features(train_pca10)
test_cube = cubic_features(test_pca10)
# train_cube (and test_cube) is a representation of our training (and test) data
# after applying the cubic kernel feature mapping to the 10-dimensional PCA representations.


# TODO: Train your softmax regression model using (train_cube, train_y)
#       and evaluate its accuracy on (test_cube, test_y).

