The following command can be used to execute the script:

python training.py fc mnist 10

The arguments are:
1. network type: fc for fully-connected or cnn.
2. dataset: one of mnist, cifar10, cifar100, svhn_cropped, iris, fashion_mnist.
4. number of epochs.

By default, the implementation will use a maxout network with the maxout rank K = 5 and c = 0.55555 (maxout initializaiton).