import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping
import time
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from os.path import join

class CNN:

    def __init__(self, img_size=224, n_classes=6):
        self.img_size = img_size
        self.n_classes = n_classes
        self.model = None
        self.history = None
        self.metrics = {
            'loss': None,
            'accuracy': None,
            'kappa': None,
            'report': None
        }
        self.training_time = 0
        self.version = 1
        self.save_dir = './saved_models/cnn'

    def build(self, version):
        self.version = version

        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(256, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.n_classes, activation='softmax')
        ])

    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=metrics)

    def fit(self, train_data, val_data, batch_size=32, epochs=15, patience=4):
        """
        Fit the model on `training_data`.
        The training will autmatically stop if there is no improvement after `patience` epochs.

        :param train_data: the data used for training.
        :param val_data: the validation data.
        :param batch_size: By default, 32.
        :param epochs: number of epochs. By default, 15.
        :param patience: how many epochs to wait for improvements before early-stopping the training.
        """
        loss_stop = EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1)
        accuracy_stop = EarlyStopping(monitor='val_accuracy', patience=patience)

        checkpoint_filepath = join(self.save_dir, 'v' + self.version, 'weights')
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        start_time = time.time()
        self.history = self.model.fit(train_data,
                                      batch_size=batch_size,
                                      epochs=epochs,
                                      validation_data=val_data,
                                    #   steps_per_epoch=len(train_data),
                                    #   validation_steps=len(val_data),
                                      callbacks=[accuracy_stop, loss_stop])  # model_checkpoint
        end_time = time.time()
        self.training_time = (end_time - start_time)

    # def evaluate_model(self, test_data):
    #     print('Evaluating model on test data...')
    #
    #     test_loss, test_acc = self.model.evaluate(test_data, verbose=2)
    #     print(f'Test loss: {test_loss}')
    #     print('Test accuracy: {:.2f} %' .format(test_acc * 100))
    #
    #     return test_loss, test_acc

    def save(self):
        """
        Save the model in the `models` folder with version defined when calling `self.model.build()`.
        E.g: './saved_models/cnn/v1/models'.
        """
        self.model.save(join(self.save_dir, 'v' + self.version, 'model'))

    def load_weights(self, version=1):
        """
        Load the version `version` model's weights  from the `weights` folder.
        E.g: './saved_models/cnn/v1/weights'.

        :param version: a number that specifies the version of the model's weights load
        """
        self.model.load_weights(join(self.save_dir, 'v' + version, 'weights'))

    def compute_metrics(self, x, y):
        """
        Evaluate the model on the new data.
        Computes the Kappa score, creates a classifcation report and retrieve the loss and accuracy.
        :param x: an array of images. Originally of shape (length, 224, 224, 3)
        where (224, 224, 3) is the shape of the images
        :param y: an array of labels. Originally of shape (length, 6) where 6 is the number of classes.
        """
        y_pred_prob = self.model.predict(x)
        y_pred = np.argmax(y_pred_prob, axis=1)

        # Get true class labels
        y_true = np.argmax(y, axis=1)

        # classification report & Kappa score
        self.metrics["report"] = classification_report(y_true, y_pred)
        self.metrics["kappa"] = cohen_kappa_score(y_true, y_pred)

        # loss & accuracy
        loss, accuracy = self.model.evaluate(x, y, verbose=2)
        self.metrics["loss"] = loss
        self.metrics["accuracy"] = accuracy

        print(f"Metrics: {self.metrics}")

    def plot_history(self):
        """
        Plot accuracy and loss for training and validation data
        """
        plt.figure(figsize=(12, 5))

        # Plot training and validation accuracy
        plt.subplot(121)
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training and validation loss
        plt.subplot(122)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.suptitle('Custom CNN')
        plt.show()
