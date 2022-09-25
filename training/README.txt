The following command can be used to execute the script:

python training.py fc mnist 5 10

The arguments are:
1. network type: fc for fully-connected or cnn.
2. dataset: one of mnist, cifar10, cifar100, svhn_cropped, iris, fashion_mnist.
3. maxout rank. If maxout rank is 0 a ReLU network is used.
4. number of epochs.