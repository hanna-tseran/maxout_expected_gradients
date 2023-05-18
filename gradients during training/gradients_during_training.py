from datetime import datetime
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

np.set_printoptions(threshold=sys.maxsize)

C_DICT = {'linear': 1}
NO_RESCALE_DATASETS = ['iris']

weights_initializer = lambda c, fan_in: tf.keras.initializers.RandomNormal(mean=0., stddev=np.sqrt(c / fan_in))
cnn_weights_initializer = lambda c, k, fan_in: tf.keras.initializers.RandomNormal(
    mean=0., stddev=np.sqrt(c / ((k**2) * fan_in)))


class TrainingExperiment:
    def __init__(self, dataset_name, epochs_num, flip, hidden_layer_width, learning_rate, network_type, report_epoch):
        self.dataset_name = dataset_name
        self.epochs_num = epochs_num
        self.flip = flip
        self.hidden_layer_widths = hidden_layer_width
        self.learning_rate = learning_rate
        self.network_type = network_type
        self.report_epoch = report_epoch

        self.augment_data = True
        self.batch_size = 32
        self.drop = 0.5
        self.epochs_drop = 100
        self.maxout_rank = 5
        self.step_decay = True
        self.stop_drop = 10000
        self.translation = True

        (self.train_dataset, self.val_dataset, self.test_dataset, self.input_shape,
             self.output_shape) = self.load_dataset()

        # Init the input that will be used for all runs
        if self.network_type == 'fc':
            self.x = next(iter(self.train_dataset))[0][:1]
        elif self.network_type == 'cnn':
            self.x = next(iter(self.train_dataset))[0][:1]
        else:
            raise Exception('Wrong network type')


    def get_maxout_fc_network(self, input_shape, output_shape, hidden_layer_widths, c):
        # Input layer
        layers = []
        if self.dataset_name not in NO_RESCALE_DATASETS:
            layers.append(tf.keras.layers.Rescaling(1./255))
        else:
            # Iris dataset only
            layers.append(self.normalization_layer)
        layers.append(tf.keras.layers.Flatten(input_shape=input_shape))

        fan_in = np.prod(np.asarray(input_shape))
        for layer_width in hidden_layer_widths:

            # This combination == one maxout layer
            layers.append(tf.keras.layers.Dense(units=layer_width * self.maxout_rank,
                                                activation='linear',
                                                kernel_initializer=weights_initializer(c, fan_in),
                                                bias_initializer=tf.zeros_initializer()))
            layers.append(tfa.layers.Maxout(layer_width))

            fan_in = layer_width

        # The last linear layer
        layers.append(tf.keras.layers.Dense(
            units=output_shape, kernel_initializer=weights_initializer(C_DICT['linear'], fan_in),
            bias_initializer=tf.zeros_initializer()))
        model = tf.keras.models.Sequential(layers)
        return model

    # VGG-19
    def get_maxout_cnn_vgg19(self, input_shape, output_shape, c):
        if self.augment_data:
            da_layers = []
            if self.translation:
                da_layers.append(tf.keras.layers.RandomTranslation(height_factor=.15, width_factor=.15))

            if self.flip:
                da_layers.append(tf.keras.layers.RandomFlip('horizontal', input_shape=input_shape))

            da_layers.extend([
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.1),
                ])
            data_augmentation = tf.keras.Sequential(da_layers)
            layers = [
                data_augmentation,
            ]
            print(f'added data augmentation')
        else:
            layers = []

        if self.dataset_name not in NO_RESCALE_DATASETS:
            layers.append(tf.keras.layers.Rescaling(1./255))

        if input_shape[0] < 32:
            print(f'Creating a smaller network!')
            channel_numbers = [64, 128, 256, 512]
            block_sizes = [2, 2, 4, 4, 2, 1]
            kernel_sizes = [3, 3, 3, 3]
            fc_sizes = [4096, 1000]
        else:
            print(f'Creating a full-sized VGG-19 network!')
            channel_numbers = [64, 128, 256, 512, 512]
            block_sizes = [2, 2, 4, 4, 4, 2, 1]
            kernel_sizes = [3, 3, 3, 3, 3]
            fc_sizes = [4096, 1000]

        pool_size = 2

        # Assume that the images are square
        filter_size = input_shape[0]
        fan_in = input_shape[-1]
        print(f'The first filter_size: {filter_size}')
        print(f'The first fan_in: {fan_in}')

        # This sets up only the conv layers
        for channels, block_size, kernel_size in zip(channel_numbers, block_sizes[:-2], kernel_sizes):
            for bi in range(block_size):
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
            # layers.append(tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_size))

            filter_size = np.floor(filter_size / pool_size)
            print(f'filter_size: {filter_size}')
            print(f'fan_in: {fan_in}')

        # Fan-in after the flatten layer
        fan_in = fan_in * filter_size**2
        print(f'FC fan in: {fan_in}')

        # Fully-connected part
        layers.append(tf.keras.layers.Flatten())
        for fc_size, block_size in zip(fc_sizes, block_sizes[-2:]):
            for _ in range(block_size):
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

        print(f'Linear fan in: {fan_in}')

        # The last linear layer
        layers.append(tf.keras.layers.Dense(
            units=output_shape,
            kernel_initializer=weights_initializer(C_DICT['linear'], fan_in),
            bias_initializer=tf.zeros_initializer()
            )
        )

        model = tf.keras.models.Sequential(layers)
        return model

    def load_dataset(self):
        if self.dataset_name == 'mnist':
            (train_dataset, val_dataset, test_dataset), ds_info = tfds.load(
                'mnist',
                split=['train[0:50000]', 'train[50000:]', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True
            )

            print(f'Number of train samples: {len(train_dataset)}')
            print(f'Number of validation samples: {len(val_dataset)}')
            print(f'Number of test samples: {len(test_dataset)}')

            input_shape = ds_info.features['image'].shape
            output_shape = ds_info.features['label'].num_classes
            print(f'input shape: {input_shape}, output shape: {output_shape}')

            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(ds_info.splits['train'].num_examples)
            train_dataset = train_dataset.batch(self.batch_size)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

            val_dataset = val_dataset.batch(self.batch_size)
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

            test_dataset = test_dataset.batch(self.batch_size)
            test_dataset = test_dataset.cache()
            test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

            print(f'Total number of batches: {len(train_dataset)}')
            return train_dataset, val_dataset, test_dataset, input_shape, output_shape

        elif self.dataset_name == 'iris':
            (train_dataset, val_dataset, test_dataset), ds_info = tfds.load(
                'iris',
                split=['train[:70%]', 'train[70%:80%]', 'train[80%:]'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True
            )

            print(f'Number of train samples: {len(train_dataset)}')
            print(f'Number of validation samples: {len(val_dataset)}')
            print(f'Number of test samples: {len(test_dataset)}')

            print(f'Features: {ds_info.features["features"]}')
            input_shape = ds_info.features['features'].shape
            output_shape = ds_info.features['label'].num_classes
            print(f'input shape: {input_shape}, output shape: {output_shape}')

            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(ds_info.splits['train'].num_examples)
            train_dataset = train_dataset.batch(self.batch_size)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

            val_dataset = val_dataset.batch(self.batch_size)
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

            test_dataset = test_dataset.batch(self.batch_size)
            test_dataset = test_dataset.cache()
            test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

            values = np.concatenate([x for x, y in train_dataset], axis=0)
            print(f'Total number of batches: {len(train_dataset)}')

            ############################################################################################################
            normalization_layer = tf.keras.layers.Normalization(axis=-1)
            normalization_layer.adapt(values)
            print(f'normalization layer: {normalization_layer.mean}')
            print(f'normalization layer: {normalization_layer.variance}')
            self.normalization_layer = normalization_layer
            ############################################################################################################

            print(f'Total number of batches: {len(train_dataset)}')
            return train_dataset, val_dataset, test_dataset, input_shape, output_shape

        elif self.dataset_name == 'cifar10':
            # Comment from https://www.tensorflow.org/datasets/performances?hl=en:
            # In addition to using ds.shuffle to shuffle records, you should also set shuffle_files=True
            # to get good shuffling behavior for larger datasets that are sharded into multiple files.
            # Otherwise, epochs will read the shards in the same order, and so data won't be truly randomized.
            (train_dataset, val_dataset, test_dataset), ds_info = tfds.load(
                'cifar10',
                split=['train[0%:90%]', 'train[90%:]', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True
            )

            print(f'Number of train samples: {len(train_dataset)}')
            print(f'Number of validation samples: {len(val_dataset)}')
            print(f'Number of test samples: {len(test_dataset)}')

            input_shape = ds_info.features['image'].shape
            output_shape = ds_info.features['label'].num_classes
            print(f'input shape: {input_shape}, output shape: {output_shape}')

            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(ds_info.splits['train'].num_examples)
            train_dataset = train_dataset.batch(self.batch_size)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

            val_dataset = val_dataset.batch(self.batch_size)
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

            test_dataset = test_dataset.batch(self.batch_size)
            test_dataset = test_dataset.cache()
            test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

            print(f'Total number of batches: {len(train_dataset)}')
            return train_dataset, val_dataset, test_dataset, input_shape, output_shape

        elif self.dataset_name == 'cifar100':
            (train_dataset, val_dataset, test_dataset), ds_info = tfds.load(
                'cifar100',
                split=['train[0%:90%]', 'train[90%:]', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True
            )

            print(f'Number of train samples: {len(train_dataset)}')
            print(f'Number of validation samples: {len(val_dataset)}')
            print(f'Number of test samples: {len(test_dataset)}')

            input_shape = ds_info.features['image'].shape
            output_shape = ds_info.features['label'].num_classes
            print(f'input shape: {input_shape}, output shape: {output_shape}')

            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(ds_info.splits['train'].num_examples)
            train_dataset = train_dataset.batch(self.batch_size)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

            val_dataset = val_dataset.batch(self.batch_size)
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

            test_dataset = test_dataset.batch(self.batch_size)
            test_dataset = test_dataset.cache()
            test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

            print(f'Total number of batches: {len(train_dataset)}')
            return train_dataset, val_dataset, test_dataset, input_shape, output_shape

        if self.dataset_name == 'fashion_mnist':
            (train_dataset, val_dataset, test_dataset), ds_info = tfds.load(
                'fashion_mnist',
                split=['train[0:50000]', 'train[50000:]', 'test'],
                shuffle_files=True,
                as_supervised=True,
                with_info=True
            )

            print(f'Number of train samples: {len(train_dataset)}')
            print(f'Number of validation samples: {len(val_dataset)}')
            print(f'Number of test samples: {len(test_dataset)}')

            input_shape = ds_info.features['image'].shape
            output_shape = ds_info.features['label'].num_classes
            print(f'input shape: {input_shape}, output shape: {output_shape}')

            train_dataset = train_dataset.cache()
            train_dataset = train_dataset.shuffle(ds_info.splits['train'].num_examples)
            train_dataset = train_dataset.batch(self.batch_size)
            train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

            val_dataset = val_dataset.batch(self.batch_size)
            val_dataset = val_dataset.cache()
            val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

            test_dataset = test_dataset.batch(self.batch_size)
            test_dataset = test_dataset.cache()
            test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

            print(f'Total number of batches: {len(train_dataset)}')
            return train_dataset, val_dataset, test_dataset, input_shape, output_shape

    def run_experiment(self, activation, c, train_dataset, val_dataset, test_dataset, input_shape,
                       output_shape):
        squared_grad_arr = []

        if self.network_type == 'fc':
            model = self.get_maxout_fc_network(input_shape=input_shape, output_shape=output_shape,
                hidden_layer_widths=self.hidden_layer_widths, c=c)
        elif self.network_type == 'cnn':
            model = self.get_maxout_cnn_vgg19(input_shape=input_shape, output_shape=output_shape, c=c)
        else:
            raise Exception('Wrong network type')

        # Optimizer
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.9,
                                            nesterov=True)

        # Loss and metrics
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=[acc_metric])

        # Callbacks
        callbacks = []

        # Step decay
        #######################################################################
        if self.step_decay:
            def step_decay(epoch):
                initial_lrate = self.learning_rate
                drop = self.drop
                epochs_drop = self.epochs_drop
                local_epoch = epoch if epoch + 1 < self.stop_drop else self.stop_drop - 2
                lrate = initial_lrate * math.pow(drop, math.floor((1 + local_epoch) / epochs_drop))
                return lrate

            reduce_lr = tf.keras.callbacks.LearningRateScheduler(step_decay)
            callbacks = [reduce_lr]
        #######################################################################

        # Tracking learning rate
        #######################################################################
        class LearningRateTracker(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs={}):
                optimizer = self.model.optimizer
                lr = tf.keras.backend.eval(optimizer.lr)
                print(f'Epoch {epoch}. lr: {lr}')
        callbacks.append(LearningRateTracker())
        #######################################################################

        # Track performance on the test set
        #######################################################################
        class TestCallback(tf.keras.callbacks.Callback):
            def __init__(self, test_dataset, epochs_num):
                super().__init__()
                self.test_dataset = test_dataset
                self.test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
                self.report_epoch = int(epochs_num*0.05) if int(epochs_num*0.05) > 0 else 1
                print(f'report_epoch: {self.report_epoch}')

            def on_epoch_end(self, epoch, logs=None):
                # Compute at 5%-s of the epochs
                if (epoch + 1) % self.report_epoch == 0:
                    losses = []
                    for x_batch_test, y_batch_test in self.test_dataset:
                        test_logits = self.model(x_batch_test, training=False)
                        self.test_acc_metric.update_state(y_batch_test, test_logits)
                    test_acc = self.test_acc_metric.result().numpy()
                    self.test_acc_metric.reset_states()
                    logs['test_sparse_categorical_accuracy'] = test_acc
                    logs['test_report_epoch'] = epoch

        callbacks.append(TestCallback(test_dataset, self.epochs_num))
        #######################################################################

        # Track the gradients during training
        #######################################################################
        def get_grad_stats(model, trainable_weights, x):
            all_grads = []
            with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
                # pred = model(np.asarray([x]))[0][0]
                pred = model(np.asarray(x))[0][0]
                grads = tape.gradient(pred, trainable_weights)

            for layer_grad in grads:
                all_grads.extend([g for g in layer_grad.numpy().ravel() if g!= 0])
            squared_grads = np.asarray(all_grads) ** 2
            return squared_grads

        class GradientCallback(tf.keras.callbacks.Callback):
            def __init__(self, report_epoch, trainable_weights, x):
                super(GradientCallback, self).__init__()
                self.report_epoch = report_epoch
                self.trainable_weights = trainable_weights
                self.x = x

            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.report_epoch == 0:
                    print(f'Recording for epoch {epoch}')
                    squared_grads = get_grad_stats(model=self.model, trainable_weights=self.trainable_weights, x=self.x)
                    squared_grad_arr.append(squared_grads)

        # Build the model
        model(self.x)
        trainable_weights = [w for w in model.trainable_weights if 'kernel' in w.name]
        squared_grad_arr.append(get_grad_stats(model=model, trainable_weights=trainable_weights, x=self.x))

        callbacks.append(GradientCallback(
            report_epoch=self.report_epoch,
            trainable_weights=trainable_weights,
            x=self.x
            ))
        #######################################################################

        history = model.fit(
            train_dataset,
            epochs=self.epochs_num,
            verbose=2, # one line per epoch
            validation_data=val_dataset,
            callbacks=callbacks
            )

        print(f'loss: {history.history["loss"]}')
        print(f'train_sparse_categorical_accuracy: {history.history["sparse_categorical_accuracy"]}')
        print(f'val_sparse_categorical_accuracy: {history.history["val_sparse_categorical_accuracy"]}')
        print(f'test_sparse_categorical_accuracy: '
              +f'{history.history["test_sparse_categorical_accuracy"]}')

        tf.keras.backend.clear_session()

        print (f'run_experiment squared_grad_arr len: {len(squared_grad_arr)}')
        return squared_grad_arr

    def run(self, c):
        experiment_start_time = datetime.now()

        (train_dataset, val_dataset, test_dataset, input_shape, output_shape) = (self.train_dataset,
                    self.val_dataset, self.test_dataset, self.input_shape, self.output_shape)

        squared_grad_arr = self.run_experiment(
            activation='maxout',
            c=c,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            input_shape=input_shape,
            output_shape=output_shape
        )

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        total_time = datetime.now() - experiment_start_time
        hours = int(total_time.seconds / 3600)
        minutes = int(total_time.seconds / 60 - hours * 60)
        seconds = int(total_time.seconds - hours * 3600 - minutes * 60)
        print(f'Full experiment took: {hours}h {minutes}min {int(seconds)}s')

        print (f'run squared_grad_arr len: {len(squared_grad_arr)}')
        return squared_grad_arr

def main(args):
    network_type = str(args[0])
    dataset_name = str(args[1])
    epochs_num = int(args[2])

    c = 0.55555

    # Used only for FC networks. Results in the paper use bigger networks, specifically
    # [64, 64, 64, 64, 64, 32, 32, 32, 32, 32, 16, 16, 16, 16, 16, 8, 8, 8, 8, 8]
    hidden_layer_width = [16, 8, 4]

    learning_rate = .002
    report_epoch = 1
    runs_num = 1
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if dataset_name == 'mnist':
        flip = False
    else:
        flip = True

    report_epochs_num = epochs_num // report_epoch + 1
    print(f'Reported results will have {report_epochs_num} entries. We report every {report_epoch}th epoch.')

    experiment = TrainingExperiment(
        dataset_name=dataset_name,
        epochs_num=epochs_num,
        flip=flip,
        hidden_layer_width=hidden_layer_width,
        learning_rate=learning_rate,
        network_type=network_type,
        report_epoch=report_epoch
    )

    squared_grad_arr = []
    for run_id in range(runs_num):
        squared_grad_arr.append(experiment.run(c=c))

    # Extend gradients for each epoch
    total_grad = [[] for _ in range(report_epochs_num)]
    for run_grad in squared_grad_arr:
        for (total_epoch_grad, run_epoch_grad) in zip(total_grad, run_grad):
            total_epoch_grad.extend(run_epoch_grad)

    grad_mean_arr = []
    grad_std_arr = []
    quartiles_arr = []

    for grad in total_grad:
        grad_mean_arr.append(np.mean(grad))
        grad_std_arr.append(np.std(grad))
        quartiles_arr.append(np.quantile(grad, [0.25, 0.5, 0.75]))

    np.savetxt('gradient_logs/'
        + f'{timestamp}_{network_type}_{dataset_name}_c{c}_runs{runs_num}_report{report_epoch}_mean.txt',
        grad_mean_arr, delimiter=',')
    np.savetxt('gradient_logs/'
        + f'{timestamp}_{network_type}_{dataset_name}_c{c}_runs{runs_num}_report{report_epoch}_std.txt',
        grad_std_arr, delimiter=',')
    np.savetxt('gradient_logs/'
        + f'{timestamp}_{network_type}_{dataset_name}_c{c}_runs{runs_num}_report{report_epoch}_quartiles.txt',
        quartiles_arr, delimiter=',')

if __name__ == "__main__":
    main(sys.argv[1:])
