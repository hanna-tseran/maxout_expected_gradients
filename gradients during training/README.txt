The following command can be used to execute the script:

python gradients_during_training.py fc mnist 10

The arguments are:
1. network type: fc for fully-connected or cnn.
2. dataset: one of mnist, cifar10, cifar100, iris, fashion_mnist.
3. number of epochs.

Means, stds, and quartiles will be stored in the gradient_logs folder.

By default, the implementation will use a maxout network with the maxout rank K = 5 and c = 0.55555 (maxout initializaiton).