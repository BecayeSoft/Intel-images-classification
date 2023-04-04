import tensorflow as tf
from keras.applications import EfficientNetB0
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping
import time
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from os.path import join


class EfficientNet:

    def __init__(self, img_size=224, n_classes=6):
        self.base_model = None
        self.model = None
        self.img_size = img_size
        self.n_classes = n_classes
        self.model = None
        self.history = None
        self.metrics = {
            'kappa': None,
            'report': None
        }
        self.training_time = 0
        self.save_dir = './saved_models/efficient_net'
        self.version = 1

    def build(self, version):
        self.version = version

        # load EfficientNet
        self.base_model = EfficientNetB0(weights='imagenet',
                                         include_top=False,
                                         input_shape=(self.img_size, self.img_size, 3))

        # Customize the model
        self.model = Sequential()
        self.model.add(self.base_model)
        self.model.add(GlobalAveragePooling2D())
        # enet.add(Dense(128, activation=LeakyReLU(alpha=0.3)))
        self.model.add(Dense(6, activation='softmax'))

    def unfreeze(self):
        # Unfreeze base model layers
        for layer in self.base_model.layers:
            layer.trainable = True

    def compile(self):
        # optimizer = SGD(learning_rate=0.01, momentum=0.9, decay=0.001)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def fit(self, train_data, val_data, batch_size=32, epochs=5, patience=4):
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
                                      steps_per_epoch=len(train_data),
                                      validation_steps=len(val_data),
                                      callbacks=[accuracy_stop, loss_stop, model_checkpoint]
                                      )
        end_time = time.time()
        self.training_time = (end_time - start_time)

    def compute_metrics(self, x, y):
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

    def save(self):
        """
        Save the model in the `models` folder with version defined when calling `self.model.build()`.
        E.g: './saved_models/cnn/v1/models'.
        """
        self.model.save(join(self.save_dir, 'v' + self.version, 'model'))

    def load_weights(self, version):
        """
        Load the version `version` model's weights  from the `weights` folder.
        E.g: './saved_models/cnn/v1/weights'.

        :param version: a number that specifies the version of the model's weights load
        """
        self.model.load_weights(join(self.save_dir, 'v' + version, 'weights'))

    def plot_history(self):
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

        plt.suptitle('Efficient Net')
        plt.show()
