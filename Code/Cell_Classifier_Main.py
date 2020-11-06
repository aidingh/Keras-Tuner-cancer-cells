import time
from datetime import datetime
from Cell_Classifier import Cell_Classifier


def main():

    data_path = '/home/aidin/Ubuntu_Files_LIFE/PycharmProjects/pyCode/TensorFlow_Certification/DATA_SETS/Cancer_Data/data.csv'
    log_model_dir = '/home/aidin/Ubuntu_Files_LIFE/PycharmProjects/pyCode/TensorFlow_Certification/Tensorflow_Certification_Project_Cancer_Classifier/Model_Logs'
    log_name = 'logs/Tensorboard_Logs'+ datetime.now().strftime("%Y%m%d-%H%M%S")
    log_model_tensorboard_dir = '/home/aidin/Ubuntu_Files_LIFE/PycharmProjects/pyCode/TensorFlow_Certification/Tensorflow_Certification_Project_Cancer_Classifier/'+log_name

    cell_obj = Cell_Classifier(data_path, log_model_dir, log_model_tensorboard_dir)
    X_train, X_test, y_train, y_test, = cell_obj.prepare_split_data()

    plot_switcher = {
        1: cell_obj.plot_corr_matrix,
        2: cell_obj.plot_bins,
        3: cell_obj.plot_group_by_mean,
        4: cell_obj.plot_group_by_sum,
        5: cell_obj.plot_by_mean
    }

    type_of_plot_option = 5
    plot_switcher.get(type_of_plot_option, cell_obj.plot_invalid_option)()

    max_trails = 5
    executions_per_trial = 1
    tuner = cell_obj.init_random_search_keras_tuner(max_trails, executions_per_trial, False)

    batch_size = 32
    epochs = 50
    tensorboard = True
    best_model, best_hps, tuner = cell_obj.run_keras_tuner(tuner, X_train, X_test, y_train, y_test, epochs, batch_size, tensorboard)
    cell_obj.tuner_information(tuner, X_test, y_test)

    cell_obj.save_best_mode(best_hps, tuner)

    cell_obj.evaluate_best_model_confution_matrix(best_model, X_test, y_test)
    cell_obj.evaluate_best_model(tuner, best_hps, X_train, X_test, y_train, y_test, batch_size,epochs)
    cell_obj.evaluation_results_ROC(best_model, X_train, X_test, y_train, y_test)

def debugg_function():
    print('Debug-Function')

if __name__ == '__main__':
        main()