import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as wb
import pandas as pd
import seaborn as sn
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(42)
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from kerastuner.tuners import RandomSearch
import os


class Cell_Classifier:

    static_shape_len = 0
    os_tensorboard_log_dirr= 'tensorboard --logdir '
    save_model_dir = '/home/aidin/Ubuntu_Files_LIFE/PycharmProjects/pyCode/TensorFlow_Certification/Tensorflow_Certification_Project_Cancer_Classifier/Best_Model_Save'

    def __init__(self, data_source, log_model_dir, log_tensorboard_model_dir):
        self.data_source = data_source
        self.log_model_dir = log_model_dir
        self.log_tensorboard_model_dir = log_tensorboard_model_dir

    def __repr__(self):
        return '{self.__class__.__}({self.data_source}, {self.log_model_dir}, {self.log_tensorboard_model_dir}'.format(self=self)

    def prepare_split_data(self):
        data_frame = pd.read_csv(self.data_source)

        # drop columns
        data_frame = data_frame.drop(['id', 'Unnamed: 32'], axis=1)

        lb_make = LabelEncoder()
        data_frame['diagnosis'] = lb_make.fit_transform(data_frame['diagnosis'])
        column_len = len(data_frame.columns)

        independent_features = list(data_frame.columns[1:column_len])
        self.static_shape_len = len(independent_features)

        Y = data_frame[['diagnosis']]
        X = data_frame[independent_features]

        # Scale data independent features
        X = preprocessing.scale(X)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

        return X_train, X_test, y_train, y_test

    def drop_irrelevant_features_for_plot(self):
        data_frame = pd.read_csv(self.data_source)
        data_frame = data_frame.drop(['id', 'Unnamed: 32'], axis=1)
        return data_frame
#----------------------------------------------------------------------------------------------------------------------
    def plot_corr_matrix(self):
        data_frame = self.drop_irrelevant_features_for_plot()
        corr_val = data_frame.corr()
        print(corr_val)
        plt.figure(figsize=(20, 10))
        sn.heatmap(corr_val, annot=True)
        sn.set_style("darkgrid")
        sn.set_context("paper")
        plt.show()

    def plot_bins(self):
        data_frame = self.drop_irrelevant_features_for_plot()
        data_frame.hist(bins=50, alpha=0.9, figsize=(20, 10))
        plt.tight_layout()
        plt.show()

    def plot_group_by_mean(self):
        data_frame = self.drop_irrelevant_features_for_plot()
        data_frame.drop(data_frame.iloc[:, 1:11], axis=1, inplace=True)
        data_frame.groupby('diagnosis').mean().plot(kind="bar", subplots=True, sharex=True, sharey=False, figsize=(20, 10), layout=(2, 10), alpha=0.7, title="Average of features related to diagnosis")
        plt.tight_layout()
        plt.show()

    def plot_group_by_sum(self):
        data_frame = self.drop_irrelevant_features_for_plot()
        data_frame.drop(data_frame.iloc[:, 11:31], axis=1, inplace=True)
        data_frame.groupby('diagnosis').sum().plot(kind="bar", subplots=True, sharex=True, sharey=False, figsize=(20, 10), layout=(2, 5), alpha=0.7, title="Average of features related to diagnosis")
        plt.tight_layout()
        plt.show()

    def plot_by_mean(self):
        data_frame = self.drop_irrelevant_features_for_plot()
        sn.pairplot(data_frame, hue='diagnosis', vars=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean'])
        plt.show()

    def plot_invalid_option(self):
        return None
    # ----------------------------------------------------------------------------------------------------------------------
    def init_random_search_keras_tuner(self, max_trials_int, executions_per_trial_int, verbose):

        tuner = RandomSearch(
            self.build_model,
            objective='val_accuracy',
            max_trials=max_trials_int, #1-3
            executions_per_trial=executions_per_trial_int, #1-5
            directory=self.log_model_dir,
            project_name='Model_Logs_Keras_Tuner'
        )

        if verbose:
            tuner_summary = tuner.search_space_summary()
            print(tuner_summary)
            return tuner
        else:
            return tuner

    def run_keras_tuner(self, tuner,  X_train, X_test, y_train, y_test, epoch_int, batch_size_int, tensorboard_callbacks):

        if tensorboard_callbacks:
            tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_int, batch_size=batch_size_int, callbacks=[tf.keras.callbacks.TensorBoard(log_dir=self.log_tensorboard_model_dir), tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])
            command = self.os_tensorboard_log_dirr + self.log_tensorboard_model_dir
            print('check this', command)
            os.system("gnome-terminal -e 'bash -c \"" + command + ";bash\"'")
        else:
            tuner.search(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch_int, batch_size=batch_size_int, callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=5)])

        best_hps = tuner.get_best_hyperparameters()[0]
        models = tuner.get_best_models(num_models=1)
        best_model = models[0]

        return best_model, best_hps, tuner

    def tuner_information(self, tuner, X_test, y_test):

        print('Full Tuner Summary: ')
        tuner.results_summary()
        print('-' * 100)

        print_best_hps = tuner.get_best_hyperparameters()[0].values
        print('Best Model Hyper Params:', print_best_hps)
        print('-' * 100)

        best_models = tuner.get_best_models(num_models=1)[0]
        best_model_eval = best_models.evaluate(X_test, y_test)
        print('Best Model Eval:', best_model_eval)
        print('-' * 100)

        print('Best Model Summary: ')
        tuner.results_summary(1)
        print('-' * 100)

    def build_model(self, hp):

        model = Sequential()
        model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation=hp.Choice('activations', ['relu', 'softmax']), input_shape=(self.static_shape_len,)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(rate=hp.Choice('dropout_rate', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])))

        for i in range(hp.Int("Layers", min_value=0, max_value=3)):
            model.add(tf.keras.layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation=hp.Choice('activations', ['relu', 'softmax'])))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Dropout(rate=hp.Choice('dropout_rate', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])), metrics=['accuracy'])

        return model

    def save_best_mode(self, best_hps, tuner):
        model = tuner.hypermodel.build(best_hps)
        model.save(self.save_model_dir)
        print('Model saved at directory:', self.save_model_dir)

    def evaluate_best_model_confution_matrix(self, best_model, X_test, y_test):
        y_hat = best_model.predict(X_test)
        matrix = confusion_matrix(y_test, y_hat.round())
        print('Confusion matrix')
        print(matrix)

    def evaluate_best_model(self, tuner, best_hps, X_train, X_test, y_train, y_test, batch_size, epochs):
        model = tuner.hypermodel.build(best_hps)
        model_hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, y_test), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=1)])

        acc = model_hist.history['accuracy']
        val_acc = model_hist.history['val_accuracy']
        loss = model_hist.history['loss']
        val_loss = model_hist.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, 'r', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)

        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation accuracy')
        plt.legend(loc=0)

        plt.show()

    def evaluation_results_ROC(self, best_model, X_train, X_test, y_train, y_test): #operator curve (ROC) and area under curve(AUC)

        y_test_pred = best_model.predict(X_test)

        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_test_pred)
        auc_keras = auc(fpr_keras, tpr_keras)
        print('Testing data AUC: ', auc_keras)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Keras (Testing Data) (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')

        y_train_pred = best_model.predict(X_train)
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_train, y_train_pred)
        auc_keras = auc(fpr_keras, tpr_keras)
        print('Training data AUC: ', auc_keras)

        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr_keras, tpr_keras, label='Keras (Training Data) (area = {:.3f})'.format(auc_keras))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()


