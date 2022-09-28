from datetime import datetime
import math
import os
import sys
import time

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

np.random.seed(4325) # Experiment results in the paper are reported for different seeds

np.set_printoptions(threshold=sys.maxsize)

C_DICT = {'linear': 1, 'relu': 2}

S_DICT = {2: 2.7519, 3: 5.18678, 4: 8.28487, 5: 12.03698} # Reciprocal of the lower bound
L_DICT = {2: 0.61102, 3: 0.47559, 4: 0.40482, 5: 0.36052} # Reciprocal of the upper bound
M_DICT = {2: 1., 3: 0.78391, 4:0.64461, 5: 0.55555} # Initialization suggested in the paper

MAXOUT_RANK = 5

# To test max-pooling initialization set c = 33333
C = M_DICT[MAXOUT_RANK] if MAXOUT_RANK > 0 else None # None corresponds to ReLU

BATCH_SIZE = 32
AUGMENT_DATA = False # Has effect only on CNNs
FLIP = False # Randomly flip the image during preprocessing
LEARNING_RATE = 0.001
OPTIMIZER = 'sgd' # 'sgd' or 'adam'
EPOCHS_DROP = 100 # Learning rate if halved every epochs_drop epochs

FC_HIDDEN_LAYERS = [64, 64, 64, 64, 64, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8]

EXTRA = False # If extra set for SVHN is used

NO_RESCALE_DATASETS = ['iris']

weights_initializer = lambda c, fan_in: tf.keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(c / fan_in))
cnn_weights_initializer = lambda c, k, fan_in: tf.keras.initializers.RandomNormal(
    mean=0., stddev=np.sqrt(c / ((k**2) * fan_in)))

class TrainingExperiment:
    def __init__(self, network_type, dataset_name, epochs_num):
        self.network_type = network_type
        self.dataset_name = dataset_name
        self.maxout_rank = MAXOUT_RANK
        self.c = C
        self.epochs_num = epochs_num

    def get_maxout_fc_network(self, input_shape, output_shape, hidden_layer_widths, c):
        # Input layer
        layers = []
        if self.dataset_name not in NO_RESCALE_DATASETS:
            layers.append(tf.keras.layers.Rescaling(1./255))
        else:
            # Iris dataset only for now
            layers.append(self.normalization_layer)

        layers.append(tf.keras.layers.Flatten(input_shape=input_shape))

        fan_in = np.prod(np.asarray(input_shape))
        for layer_width in hidden_layer_widths:

            # This combination == one maxout layer
            layers.append(tf.keras.layers.Dense(units=layer_width * self.maxout_rank, activation='linear',
                kernel_initializer=weights_initializer(c, fan_in), bias_initializer=tf.zeros_initializer()))
            layers.append(tfa.layers.Maxout(layer_width))

            fan_in = layer_width

        # The last linear layer
        layers.append(tf.keras.layers.Dense(
            units=output_shape, kernel_initializer=weights_initializer(C_DICT['linear'], fan_in),
            bias_initializer=tf.zeros_initializer()))
        model = tf.keras.models.Sequential(layers)
        return model

    def get_relu_fc_network(self, input_shape, output_shape, hidden_layer_widths):
        layers = []
        if self.dataset_name not in NO_RESCALE_DATASETS:
            layers.append(tf.keras.layers.Rescaling(1./255))
        else:
            # Iris dataset only
            layers.append(self.normalization_layer)
        layers.append(tf.keras.layers.Flatten(input_shape=input_shape))

        fan_in = np.prod(np.asarray(input_shape))
        for layer_width in hidden_layer_widths:
            layers.append(tf.keras.layers.Dense(
                units=layer_width, activation='relu', kernel_initializer=weights_initializer(C_DICT['relu'], fan_in),
                bias_initializer=tf.zeros_initializer()))
            fan_in = layer_width

        # The last linear layer
        layers.append(tf.keras.layers.Dense(
            units=output_shape, kernel_initializer=weights_initializer(C_DICT['linear'], fan_in),
            bias_initializer=tf.zeros_initializer()))
        model = tf.keras.models.Sequential(layers)
        return model

    def get_maxout_cnn_vgg19(self, input_shape, output_shape, c):
        if AUGMENT_DATA:
            da_layers = []
            da_layers.append(tf.keras.layers.RandomTranslation(height_factor=.15, width_factor=.15))

            if FLIP:
                da_layers.append(tf.keras.layers.RandomFlip('horizontal', input_shape=input_shape))

            da_layers.extend([
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                ])
            data_augmentation = tf.keras.Sequential(da_layers)
            layers = [
                data_augmentation,
            ]
        else:
            layers = []

        if self.dataset_name not in NO_RESCALE_DATASETS:
            layers.append(tf.keras.layers.Rescaling(1./255))

        if input_shape[0] < 32:
            channel_numbers = [64, 128, 256, 512]
            block_sizes = [2, 2, 4, 4, 2, 1]
            kernel_sizes = [3, 3, 3, 3]
            fc_sizes = [4096, 1000]
        else:
            channel_numbers = [64, 128, 256, 512, 512]
            block_sizes = [2, 2, 4, 4, 4, 2, 1]
            kernel_sizes = [3, 3, 3, 3, 3]
            fc_sizes = [4096, 1000]

        pool_size = 2

        # Assume that the images are square
        filter_size = input_shape[0]
        fan_in = input_shape[-1]

        # This sets up only the conv layers
        for channels, block_size, kernel_size in zip(channel_numbers, block_sizes[:-2], kernel_sizes):
            for bi in range(block_size):
                # max-pooling initialization
                if c == 33333:
                    if bi < block_size - 1:
                        local_c = 0.55555
                    else:
                        local_c = 0.26573

                    layers.extend([
                        ######################################
                        # This combination == one maxout layer
                        tf.keras.layers.Conv2D(
                            filters=channels * self.maxout_rank,
                            kernel_size=kernel_size,
                            padding='same',
                            activation='linear',
                            kernel_initializer=cnn_weights_initializer(c=local_c, k=kernel_size, fan_in=fan_in),
                            bias_initializer=tf.zeros_initializer()
                            ),
                        tfa.layers.Maxout(channels)
                        ######################################
                        ])

                else:
                    layers.extend([
                        ######################################
                        # This combination == one maxout layer
                        tf.keras.layers.Conv2D(
                            filters=channels * self.maxout_rank,
                            kernel_size=kernel_size,
                            padding='same',
                            activation='linear',
                            kernel_initializer=cnn_weights_initializer(c=c, k=kernel_size, fan_in=fan_in),
                            bias_initializer=tf.zeros_initializer()
                            ),
                        tfa.layers.Maxout(channels)
                        ######################################
                        ])
                fan_in = channels

            layers.append(tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_size))

            filter_size = np.floor(filter_size / pool_size)

        # Fan-in after the flatten layer
        fan_in = fan_in * filter_size**2

        # Fully-connected part
        layers.append(tf.keras.layers.Flatten())
        for fc_size, block_size in zip(fc_sizes, block_sizes[-2:]):
            for _ in range(block_size):
                # max-pooling initialization
                if c == 33333:
                    local_c = 0.55555
                    layers.extend([
                        ######################################
                        # Maxout fully-connected layer
                        tf.keras.layers.Dense(
                            units=fc_size * self.maxout_rank,
                            activation='linear',
                            kernel_initializer=weights_initializer(local_c, fan_in),
                            bias_initializer=tf.zeros_initializer()
                            ),
                        tfa.layers.Maxout(fc_size),
                        ######################################
                    ])
                else:
                    layers.extend([
                        ######################################
                        # Maxout fully-connected layer
                        tf.keras.layers.Dense(
                            units=fc_size * self.maxout_rank,
                            activation='linear',
                            kernel_initializer=weights_initializer(c, fan_in),
                            bias_initializer=tf.zeros_initializer()
                            ),
                        tfa.layers.Maxout(fc_size),
                        ######################################
                    ])
                fan_in = fc_size

        # The last linear layer
        layers.append(tf.keras.layers.Dense(
            units=output_shape,
            kernel_initializer=weights_initializer(C_DICT['linear'], fan_in),
            bias_initializer=tf.zeros_initializer()
            )
        )

        model = tf.keras.models.Sequential(layers)
        return model

    def get_relu_cnn_vgg19(self, input_shape, output_shape):
        if AUGMENT_DATA:
            da_layers = []
            da_layers.append(tf.keras.layers.RandomTranslation(height_factor=.15, width_factor=.15))

            if FLIP:
                da_layers.append(tf.keras.layers.RandomFlip('horizontal', input_shape=input_shape))

            da_layers.extend([
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                ])
            data_augmentation = tf.keras.Sequential(da_layers)
            layers = [
                data_augmentation,
            ]
        else:
            layers = []

        if self.dataset_name not in NO_RESCALE_DATASETS:
            layers.append(tf.keras.layers.Rescaling(1./255))

        if input_shape[0] < 32:
            channel_numbers = [64, 128, 256, 512]
            block_sizes = [2, 2, 4, 4, 2, 1]
            kernel_sizes = [3, 3, 3, 3]
            fc_sizes = [4096, 1000]
        else:
            channel_numbers = [64, 128, 256, 512, 512]
            block_sizes = [2, 2, 4, 4, 4, 2, 1]
            kernel_sizes = [3, 3, 3, 3, 3]
            fc_sizes = [4096, 1000]

        pool_size = 2

        # Assume that the images are square
        filter_size = input_shape[0]
        fan_in = input_shape[-1]

        # This sets up only the conv layers
        for channels, block_size, kernel_size in zip(channel_numbers, block_sizes[:-2], kernel_sizes):
            for bi in range(block_size):
                layers.extend([
                    tf.keras.layers.Conv2D(
                        filters=channels,
                        kernel_size=kernel_size,
                        padding='same',
                        activation='relu',
                        kernel_initializer=cnn_weights_initializer(c=2, k=kernel_size, fan_in=fan_in),
                        bias_initializer=tf.zeros_initializer()
                        )
                    ######################################
                    ])
                fan_in = channels

            layers.append(tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_size))

            filter_size = np.floor(filter_size / pool_size)

        # Fan-in after the flatten layer
        fan_in = fan_in * filter_size**2

        # Fully-connected part
        layers.append(tf.keras.layers.Flatten())
        for fc_size, block_size in zip(fc_sizes, block_sizes[-2:]):
            for _ in range(block_size):
                layers.extend([
                    tf.keras.layers.Dense(
                        units=fc_size,
                        activation='relu',
                        kernel_initializer=weights_initializer(2, fc_size),
                        bias_initializer=tf.zeros_initializer()
                        ),
                    ######################################
                ])
                fan_in = fc_size

        # The last linear layer
        layers.append(tf.keras.layers.Dense(
            units=output_shape,
            kernel_initializer=weights_initializer(C_DICT['linear'], fan_in),
            bias_initializer=tf.zeros_initializer()
            )
        )

        model = tf.keras.models.Sequential(layers)
        return model

    def run_experiment(self, activation, c, train_dataset, val_dataset, test_dataset, input_shape, output_shape):
        epoch_train_loss_list = []
        epoch_train_acc_list = []
        epoch_val_acc_list  = []
        epoch_test_acc_list  = []
        epoch_test_epoch_list = []

        ############################################################################################################
        # Construct the network
        if self.network_type == 'fc':
            if activation == 'maxout':
                model = self.get_maxout_fc_network(input_shape=input_shape, output_shape=output_shape,
                    hidden_layer_widths=FC_HIDDEN_LAYERS, c=c)
            elif activation == 'relu':
                model = self.get_relu_fc_network(input_shape=input_shape, output_shape=output_shape,
                    hidden_layer_widths=FC_HIDDEN_LAYERS)
            else:
                raise Exception('Wrong init')

        elif self.network_type == 'cnn':
            if activation == 'maxout':
                model = self.get_maxout_cnn_vgg19(input_shape=input_shape, output_shape=output_shape, c=c)
            elif activation == 'relu':
                model = self.get_relu_cnn_vgg19(input_shape=input_shape, output_shape=output_shape)
            else:
                raise Exception('Wrong activation function')
        else:
            raise Exception('Wrong network type')

        # Optimizer
        if OPTIMIZER == 'adam':
            optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        elif OPTIMIZER == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=0.9, nesterov=True)
        else:
            raise Exception('Wrong optimizer')

        # Loss and metrics
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=[acc_metric])
        ############################################################################################################

        # Callbacks
        ################################################################################################################
        callbacks = []

        # Step decay
        ####################################################################################################
        def step_decay(epoch):
            initial_lrate = LEARNING_RATE
            # drop = self.drop
            epochs_drop = EPOCHS_DROP
            lrate = initial_lrate * math.pow(0.5, math.floor((1 + epoch) / epochs_drop))
            return lrate

        reduce_lr = tf.keras.callbacks.LearningRateScheduler(step_decay)
        callbacks = [reduce_lr]
        ####################################################################################################

        # Track performance on the test set
        ####################################################################################################
        class TestCallback(tf.keras.callbacks.Callback):
            def __init__(self, test_dataset, epochs_num):
                super().__init__()
                self.test_dataset = test_dataset
                self.test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

            def on_epoch_end(self, epoch, logs=None):
                for x_batch_test, y_batch_test in self.test_dataset:
                    test_logits = self.model(x_batch_test, training=False)
                    self.test_acc_metric.update_state(y_batch_test, test_logits)
                test_acc = self.test_acc_metric.result().numpy()
                self.test_acc_metric.reset_states()
                logs['test_sparse_categorical_accuracy'] = test_acc

        callbacks.append(TestCallback(test_dataset, self.epochs_num))
        ####################################################################################################
        ################################################################################################################

        history = model.fit(
            train_dataset,
            epochs=self.epochs_num,
            verbose=2, # one line per epoch
            validation_data=val_dataset,
            callbacks=callbacks
            )

        epoch_train_loss_list = history.history['loss']
        epoch_train_acc_list = history.history['sparse_categorical_accuracy']
        epoch_val_acc_list = history.history['val_sparse_categorical_accuracy']
        epoch_test_acc_list = history.history['test_sparse_categorical_accuracy']

        # Store model summary
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        model_summary = '\n'.join(stringlist)
        self.model_summary = model_summary

        tf.keras.backend.clear_session()
        return (epoch_train_loss_list, epoch_train_acc_list, epoch_val_acc_list, epoch_test_acc_list,
            epoch_test_epoch_list)

    def run(self):
        experiment_start_time = datetime.now()

        if tf.config.experimental.list_physical_devices('GPU'):
            tf.config.set_visible_devices(gpus[0], 'GPU')

        train_dataset, val_dataset, test_dataset, input_shape, output_shape = self.load_dataset()

        if self.c:
            (epoch_train_loss_list, epoch_train_acc_list,
            epoch_val_acc_list, epoch_test_acc_list,
            epoch_test_epoch_list) = self.run_experiment(
                activation='maxout', c=self.c, train_dataset=train_dataset, val_dataset=val_dataset,
                test_dataset=test_dataset, input_shape=input_shape, output_shape=output_shape)
        # c = None corresponds to the ReLU network
        else:
            (epoch_train_loss_list, epoch_train_acc_list,
            epoch_val_acc_list, epoch_test_acc_list,
            epoch_test_epoch_list) = self.run_experiment(
                activation='relu', c=self.c, train_dataset=train_dataset, test_dataset=test_dataset,
                val_dataset=val_dataset, input_shape=input_shape, output_shape=output_shape)

        total_time = datetime.now() - experiment_start_time
        hours = int(total_time.seconds / 3600)
        minutes = int(total_time.seconds / 60 - hours * 60)
        seconds = int(total_time.seconds - hours * 3600 - minutes * 60)

        # Store experiment results
        filename = f'result.txt'
        with open(filename, 'w') as f:
            f.write(
                f'network_type: {self.network_type}\n'
                + f'dataset_name: {self.dataset_name}\n'
                + f'maxout_rank: {self.maxout_rank}\n'
                + f'c: {self.c}\n'
                + f'epochs_num: {self.epochs_num}\n'
                + f'run time: {hours}h {minutes}min {int(seconds)}s\n'
                +'\n++++++++++++++++++++++++++++++++++++++++++++++++++\n'
                +'++++++++++++++++++++++++++++++++++++++++++++++++++\n\n'
                + f'Training loss list:\n{epoch_train_loss_list}\n'
                + f'Training accuracy list:\n{epoch_train_acc_list}\n'
                + f'Validation accuracy list:\n{epoch_val_acc_list }\n'
                + f'Test accuracy list:\n{epoch_test_acc_list }\n'
                +'\n++++++++++++++++++++++++++++++++++++++++++++++++++\n'
                +'++++++++++++++++++++++++++++++++++++++++++++++++++\n\n'
                + f'Model summary:\n{self.model_summary}\n'
                )

    def load_dataset(self):
        if self.dataset_name == 'mnist':
            (train_dataset, val_dataset, test_dataset), ds_info = tfds.load(
                'mnist',
                split=['train[0:50000]', 'train[50000:]', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True
            )

            input_shape = ds_info.features['image'].shape
            output_shape = ds_info.features['label'].num_classes

            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(ds_info.splits['train'].num_examples)
            train_dataset = train_dataset.batch(BATCH_SIZE)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

            val_dataset = val_dataset.batch(BATCH_SIZE)
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

            test_dataset = test_dataset.batch(BATCH_SIZE)
            test_dataset = test_dataset.cache()
            test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

            return train_dataset, val_dataset, test_dataset, input_shape, output_shape

        elif self.dataset_name == 'iris':
            (train_dataset, val_dataset, test_dataset), ds_info = tfds.load(
                'iris',
                split=['train[:70%]', 'train[70%:80%]', 'train[80%:]'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True
            )

            input_shape = ds_info.features['features'].shape
            output_shape = ds_info.features['label'].num_classes

            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(ds_info.splits['train'].num_examples)
            train_dataset = train_dataset.batch(BATCH_SIZE)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

            val_dataset = val_dataset.batch(BATCH_SIZE)
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

            test_dataset = test_dataset.batch(BATCH_SIZE)
            test_dataset = test_dataset.cache()
            test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

            values = np.concatenate([x for x, y in train_dataset], axis=0)
            normalization_layer = tf.keras.layers.Normalization(axis=-1)
            normalization_layer.adapt(values)
            self.normalization_layer = normalization_layer

            return train_dataset, val_dataset, test_dataset, input_shape, output_shape

        elif self.dataset_name == 'cifar10':
            (train_dataset, val_dataset, test_dataset), ds_info = tfds.load(
                'cifar10',
                split=['train[0%:90%]', 'train[90%:]', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True
            )

            input_shape = ds_info.features['image'].shape
            output_shape = ds_info.features['label'].num_classes

            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(ds_info.splits['train'].num_examples)
            train_dataset = train_dataset.batch(BATCH_SIZE)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

            val_dataset = val_dataset.batch(BATCH_SIZE)
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

            test_dataset = test_dataset.batch(BATCH_SIZE)
            test_dataset = test_dataset.cache()
            test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

            return train_dataset, val_dataset, test_dataset, input_shape, output_shape

        elif self.dataset_name == 'cifar100':
            (train_dataset, val_dataset, test_dataset), ds_info = tfds.load(
                'cifar100',
                split=['train[0%:90%]', 'train[90%:]', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True
            )

            input_shape = ds_info.features['image'].shape
            output_shape = ds_info.features['label'].num_classes

            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(ds_info.splits['train'].num_examples)
            train_dataset = train_dataset.batch(BATCH_SIZE)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

            val_dataset = val_dataset.batch(BATCH_SIZE)
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

            test_dataset = test_dataset.batch(BATCH_SIZE)
            test_dataset = test_dataset.cache()
            test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

            return train_dataset, val_dataset, test_dataset, input_shape, output_shape

        elif self.dataset_name == 'fashion_mnist':
            (train_dataset, val_dataset, test_dataset), ds_info = tfds.load(
                'fashion_mnist',
                split=['train[0:50000]', 'train[50000:]', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True
            )

            input_shape = ds_info.features['image'].shape
            output_shape = ds_info.features['label'].num_classes

            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(ds_info.splits['train'].num_examples)
            train_dataset = train_dataset.batch(BATCH_SIZE)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

            val_dataset = val_dataset.batch(BATCH_SIZE)
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

            test_dataset = test_dataset.batch(BATCH_SIZE)
            test_dataset = test_dataset.cache()
            test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

            return train_dataset, val_dataset, test_dataset, input_shape, output_shape

        # 400 samples per class selected from the training set and 200 samples per class from the extra set are used for
        # validation. The remainder of the training set and the extra set are used for training.
        elif self.dataset_name == 'svhn_cropped':
            (original_train_dataset, extra_dataset, test_dataset), ds_info = tfds.load(
            'svhn_cropped',
                split=['train', 'extra', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True
            )
            num_classes = ds_info.features["label"].num_classes

            # Number of samples that will go into the validation dataset
            samples_from_train = 400
            samples_from_extra = 200

            new_train_dataset = []
            new_train_dataset_labels = []
            new_val_dataset = []
            new_val_dataset_labels = []
            for li in range(num_classes):
                # Divide original train dataset
                label_train_dataset = list(original_train_dataset.filter(lambda img, label: label == li))
                new_val_dataset.extend([e[0].numpy() for e in label_train_dataset[:samples_from_train]])
                new_val_dataset_labels.extend([e[1].numpy() for e in label_train_dataset[:samples_from_train]])
                new_train_dataset.extend([e[0].numpy() for e in label_train_dataset[samples_from_train:]])
                new_train_dataset_labels.extend([e[1].numpy() for e in label_train_dataset[samples_from_train:]])

                # Divide extra dataset
                if EXTRA:
                    label_extra_dataset = list(extra_dataset.filter(lambda img, label: label == li))
                    new_val_dataset.extend([e[0].numpy() for e in label_extra_dataset[:samples_from_extra]])
                    new_val_dataset_labels.extend([e[1].numpy() for e in label_extra_dataset[:samples_from_extra]])
                    new_train_dataset.extend([e[0].numpy() for e in label_extra_dataset[samples_from_extra:]])
                    new_train_dataset_labels.extend([e[1].numpy() for e in label_extra_dataset[samples_from_extra:]])

            train_dataset = tf.data.Dataset.from_tensor_slices((new_train_dataset, new_train_dataset_labels))
            val_dataset = tf.data.Dataset.from_tensor_slices((new_val_dataset, new_val_dataset_labels))

            input_shape = ds_info.features['image'].shape
            output_shape = ds_info.features['label'].num_classes

            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(ds_info.splits['train'].num_examples)
            train_dataset = train_dataset.batch(BATCH_SIZE)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

            val_dataset = val_dataset.batch(BATCH_SIZE)
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

            test_dataset = test_dataset.batch(BATCH_SIZE)
            test_dataset = test_dataset.cache()
            test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

            return train_dataset, val_dataset, test_dataset, input_shape, output_shape

        else:
            raise Exception('Wrong dataset name')

def main(args):
    # Read command line arguments and setup the experiment
    network_type = str(args[0])
    dataset_name = str(args[1])
    epochs_num = int(args[2])

    print(f'network_type: {network_type}')
    print(f'dataset_name: {dataset_name}')
    print(f'epochs_num: {epochs_num}')

    experiment = TrainingExperiment(
        network_type=network_type,
        dataset_name=dataset_name,
        epochs_num=epochs_num
        )
    experiment.run()

if __name__ == "__main__":
    main(sys.argv[1:])