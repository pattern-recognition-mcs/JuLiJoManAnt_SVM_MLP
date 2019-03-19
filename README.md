# Group's name: JuLiJoManAnt

# MNIST digits recognition using SVM and MLP

## Exercise 02_a - SVM
### Task
In this exercise you should aim to improve the recognition rate on the MNIST dataset
using SVM.
Discuss a good architecture for your framework so that you can reuse software components in
later exercises.
Reminder: As already discussed. From now on you are free to either implement algorithms
on your own or use any kinds of libraries. Use the provided training set to build your SVM. Apply the trained SVM to classify the test
set. Investigate at least two different kernels and optimize the SVM parameters by means of
cross-validation.

### Expected Output
- Access to your github so that we can inspect your code.
- Average accuracy during cross-validation for all investigated kernels (e.g. linear and RBF) and all parameter values (e.g. C and γ).
- Accuracy on the test set with the optimized parameter values.

## Exercise 02_b - MLP
### Task
Use the provided training set to train an MLP with one hidden layer. Apply the trained MLP
to classify the test set. Perform cross-validation with the following parameters:
- Optimize number of neurons in the hidden layer (typically in the range [10, 100]).
- Optimize learning rate c (typically in the range [0.001, 0.1]).
- Optimize number of training iterations. Plot a graph showing the error on the training set and the validation set, respectively, with respect to the training epochs.
- Perform the random initialization several times and choose the best network during cross-
validation.

### Expected Output
- Access to your github so that we can inspect your code.
- Plot showing the error-rate on the training and the validation set with respect to the training epochs.
- Test accuracy with the best parameters found during cross-validation.
