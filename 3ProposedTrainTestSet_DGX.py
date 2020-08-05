import time
import numpy as np
import csv
import math
import matplotlib.pyplot as plt

from keras.initializers import glorot_normal
from keras.optimizers import Adam
from keras.callbacks import Callback

import multiprocessing as mp


def start_process():
    print('starting ', mp.current_process().name)


def export_mcs_folds(vers, mcs_tot, str_item, item):
    with open(vers + "_TrainTestOutput_MCS" + str(mcs_tot) + "_" + str_item + ".csv", "a") as f:
        writer = csv.writer(f)
        # print(np.shape(item))
        for i in item:
            # print('np.shape(i)', np.shape(i))
            for j in i:
                # print('j', j)
                writer.writerow(j)
    f.close()


# +
def export_mcs_nrmodels_folds(vers, mcs_tot, str_item, item, x_val_shape):
    with open(vers + "_TrainTestOutput_MCS" + str(mcs_tot) + "_" + str_item + ".csv", "a") as f:
        writer = csv.writer(f)
        # print(np.shape(item))
        for i in item:
            # print('np.shape(i)', np.shape(i))
            for q in i:
                # print('np.shape(q)', np.shape(q))
                writer.writerow(np.reshape(q, (x_val_shape,)))
                
def export_nrmodels_folds(vers, mcs_tot, str_item, item, x_val_shape):
    with open(vers + "_TrainTestOutput_MCS" + str(mcs_tot) + "_" + str_item + "Intermediate.csv", "a") as f:
        writer = csv.writer(f)
        # print(np.shape(item))
        for i in item:
            # print('np.shape(i)', np.shape(i))
            # print('np.shape(q)', np.shape(q))
            writer.writerow(np.reshape(i, (x_val_shape,)))


# -

class ShowLr(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        from keras import backend as k

        decay = self.model.optimizer.decay
        lr = self.model.optimizer.lr
        iters = self.model.optimizer.iterations  # only this should not be const
        beta_1 = self.model.optimizer.beta_1
        beta_2 = self.model.optimizer.beta_2
        # calculate
        lr = lr * (1. / (1. + decay * k.cast(iters, k.dtype(decay))))
        t = k.cast(iters, k.floatx()) + 1
        lr_t = lr * (k.sqrt(1. - k.pow(beta_2, t)) / (1. - k.pow(beta_1, t)))

        if epoch % 1000 == 0:
            print("epoch={:02d}, lr={:.10f}".format(epoch, np.float32(k.eval(lr_t))))
        return


def sse(y_true, y_pred):
    from keras import backend as k
    return k.sum(k.square(y_true - y_pred))


class DensityNeuralNetwork:
    def __init__(self, mcs_i, version, hidden_layers, hidden_neurons,
                 epoch, batch, obs, val_size, n_nodes):
        # input
        self.x = None
        self.y = None
        self.x_val = None
        self.y_val = None

        self.index = None
        # self.train_index = None
        # self.test_index = None
        self.model = None

        # output
        self.predictions = None
        self.test_index_saved = None
        self.train_metrics = None
        self.test_metrics = None

        # parameters
        self.version = version
        self.layers = hidden_layers
        self.neurons = hidden_neurons
        self.n_nodes = n_nodes
        self.combinations = np.array(np.meshgrid(hidden_layers, hidden_neurons)).T.reshape(-1, 2)
        self.nr_models = np.shape(self.combinations)[0]

        self.mcs_i = mcs_i
        self.m = None
        self.epochs = epoch
        self.batch_size = batch

        self.obs = obs
        self.val_size = val_size

    def import_sets(self, str_x, str_index, obs, val_size):
        # import sets
        x_original_set = []
        with open(str_x, newline='\n') as f:
            reader = csv.reader(f, delimiter=',')
            count_row = 0
            for row in reader:
                temp = []
                count_row += 1
                for j in row:
                    temp.append(eval(j))
                x_original_set.append(np.array(temp))
        count_row = int(count_row / obs)
        x_original_set = np.reshape(np.array(x_original_set), (int(count_row), observations, self.n_nodes))

        index_set = []
        with open(str_index, newline='\n') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                index_set.append(np.array([eval(row[i]) for i in range(val_size)]))
        return x_original_set, index_set
        # return True

    def create_target_set(self, step_size=0.001):
        if self.n_nodes == 1:
            axis = np.arange(np.min(self.x), np.max(self.x) + step_size, step_size)
            axis[0] = -math.inf
            grid = np.zeros([np.shape(axis)[0]])
            for m in range(len(self.x)):
                grid[np.where(self.x[m][0] >= axis)[0][-1]] += 1
            cdf = np.cumsum(grid, axis=0)
            cdf /= cdf[-1]
            self.y = np.array([cdf[np.where(self.x[m][0] >= axis)[0][-1]] for m in range(len(self.x))])
        if self.n_nodes == 2:
            axis = np.arange(np.min(self.x), np.max(self.x) + step_size, step_size)
            axis[0] = -math.inf
            grid = np.zeros([np.shape(axis)[0], np.shape(axis)[0]])
            for m in range(len(self.x)):
                grid[np.where(self.x[m][0] >= axis)[0][-1], np.where(self.x[m][1] >= axis)[0][-1]] += 1
            cdf = np.cumsum(np.cumsum(grid, axis=0), axis=1)
            cdf /= cdf[-1][-1]
            self.y = np.array([cdf[np.where(self.x[m][0] >= axis)[0][-1], np.where(self.x[m][1] >= axis)[0][-1]]
                               for m in range(len(self.x))])
        if self.n_nodes == 3:
            axis = np.arange(np.min(self.x), np.max(self.x) + step_size, step_size)
            axis[0] = -math.inf
            grid = np.zeros([np.shape(axis)[0], np.shape(axis)[0], np.shape(axis)[0]])
            for m in range(len(self.x)):
                grid[np.where(self.x[m][0] >= axis)[0][-1], np.where(self.x[m][1] >= axis)[0][-1],
                     np.where(self.x[m][2] >= axis)[0][-1]] += 1
            cdf = np.cumsum(np.cumsum(np.cumsum(grid, axis=0), axis=1), axis=2)
            cdf /= cdf[-1][-1][-1]
            self.y = np.array([cdf[np.where(self.x[m][0] >= axis)[0][-1], np.where(self.x[m][1] >= axis)[0][-1],
                                   np.where(self.x[m][2] >= axis)[0][-1]] for m in range(len(self.x))])

    def create_import_set(self, obs, val_size, str_x, str_index):
        # import sets
        x_original_set, self.index = self.import_sets(str_x, str_index, obs, val_size)

        # prepare sets
        x_original = x_original_set[self.mcs_i]
        index_vers = self.index[self.mcs_i].astype(int)

        self.x_val = x_original[index_vers]

        self.x = x_original

        # create target set without validation set ProposedVersion
        self.create_target_set()
        self.y_val = self.y[index_vers]

        self.x = np.zeros((self.obs - self.val_size, self.n_nodes))
        for i in range(self.n_nodes):
            self.x[:, i] = np.delete(x_original[:, i], index_vers, 0)

        self.create_target_set()

    def layer_names(self):
        # input layer, hidden layers and output layer
        model_inputs = dict()
        model_names = dict()

        # set-up
        model_inputs[str(0)] = '00'
        for nr in range(self.nr_models):
            model_names[str(nr + 1)] = dict()
            if self.combinations[nr][0] >= 1:
                for i in range(self.combinations[nr][0]+1):
                    model_names[str(nr + 1)][str(i+1)] = str(nr+1)+str(i+1)
        return model_inputs, model_names

    def create_model(self, init=glorot_normal(seed=123), act='tanh', b='True', output_act=0, opt=Adam(lr=0.001)):
        from keras.layers import Input
        from keras.layers import Dense
        from keras.models import Model

        # input layer
        models_input = Input(shape=(self.n_nodes,), name='00')

        models_layers = dict()

        for nr in range(self.nr_models):
            models_layers[str(nr + 1)] = dict()
            # first hidden layer
            models_layers[str(nr + 1)][str(1)] = Dense(self.combinations[nr][1], kernel_initializer=init,
                                                       activation=act, use_bias=b,
                                                       name=self.layer_names()
                                                       [1][str(nr + 1)][str(1)])(models_input)
            # multiple hidden layers
            for h in range(1, self.combinations[nr][0]):
                models_layers[str(nr + 1)][str(h + 1)] = Dense(self.combinations[nr][1], kernel_initializer=init,
                                                               activation=act, use_bias=b,
                                                               name=self.layer_names()[1][str(nr + 1)][str(h + 1)])(
                    models_layers[str(nr + 1)][str(h)])
            # output layer
            if output_act != 1:
                models_layers[str(nr + 1)][str(self.combinations[nr][0] + 1)] = Dense(1, kernel_initializer=init,
                                                                                      use_bias=b,
                                                                                      name=self.
                                                                                      layer_names()[1][str(nr + 1)]
                                                                                      [str(self.combinations[nr][
                                                                                               0] + 1)])(
                    models_layers[str(nr + 1)][str(self.combinations[nr][0])])
            else:
                # self.output_act == 1:
                models_layers[str(nr + 1)][str(self.combinations[nr][0] + 1)] = Dense(1, kernel_initializer=init,
                                                                                      activation='sigmoid', use_bias=b,
                                                                                      name=self.
                                                                                      layer_names()[1][str(nr + 1)]
                                                                                      [str(self.combinations[nr][
                                                                                               0] + 1)])(
                    models_layers[str(nr + 1)][str(self.combinations[nr][0])])

            # setting up small models
            models_seperate = dict()
            if self.nr_models > 1:
                models_seperate[str(nr + 1)] = Model(inputs=models_input,
                                                     outputs=models_layers[str(nr + 1)][
                                                         str(self.combinations[nr][0] + 1)])
                models_seperate[str(nr + 1)].compile(optimizer=opt,
                                                     loss=sse)

        # setting up big model
        self.model = Model(inputs=models_input,
                           outputs=[models_layers[str(j + 1)]
                                    [str(self.combinations[j][0] + 1)] for j in range(self.nr_models)])
        self.model.compile(optimizer=opt,
                           loss=sse)
        # self.model.summary()
        # return main_model (which is self.model)

    def train_cdf(self, it):
        # print('iterators', it[0], it[1])
        self.mcs_i = it

        print(self.mcs_i, ':', 'TRAIN: ', len(self.x), 'TEST: ', len(self.x_val), '\n')
        x_train, x_test = self.x, self.x_val
        y_train, y_test = self.y, self.y_val

        # compile model
        self.create_model()

        # fit compiled model
        hist = ShowLr()
        hist_model = self.model.fit(x_train, [y_train] * self.nr_models, callbacks=[hist],
                                    validation_data=(x_test, [y_test] * self.nr_models),
                                    epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        # evaluate model
        # evaluate model
        self.test_metrics = np.zeros([self.nr_models + 1, self.epochs])
        self.train_metrics = np.zeros([self.nr_models + 1, self.epochs])
        # print('names errors ', hist_model.history.keys())
        # print('values errors ', hist_model.history.values())
        # print('errors ', hist_model.history)
        # self.metrics = copy.deepcopy(hist_model.history)
        self.test_metrics[0] = hist_model.history['val_loss']
        self.test_metrics[1] = hist_model.history['val_12_loss']
        self.test_metrics[2] = hist_model.history['val_22_loss']
        self.test_metrics[3] = hist_model.history['val_32_loss']
        self.test_metrics[4] = hist_model.history['val_43_loss']
        self.test_metrics[5] = hist_model.history['val_53_loss']
        self.test_metrics[6] = hist_model.history['val_63_loss']
        self.test_metrics[7] = hist_model.history['val_74_loss']
        self.test_metrics[8] = hist_model.history['val_84_loss']
        self.test_metrics[9] = hist_model.history['val_94_loss']

        self.train_metrics[0] = hist_model.history['loss']
        self.train_metrics[1] = hist_model.history['12_loss']
        self.train_metrics[2] = hist_model.history['22_loss']
        self.train_metrics[3] = hist_model.history['32_loss']
        self.train_metrics[4] = hist_model.history['43_loss']
        self.train_metrics[5] = hist_model.history['53_loss']
        self.train_metrics[6] = hist_model.history['63_loss']
        self.train_metrics[7] = hist_model.history['74_loss']
        self.train_metrics[8] = hist_model.history['84_loss']
        self.train_metrics[9] = hist_model.history['94_loss']

        # plot training history
        plt.figure()
        # plt.plot(hist_model.history['loss'], label='train')
        # plt.plot(hist_model.history['val_loss'], label='test')
        plt.plot(self.train_metrics[0]/len(x_test), label='train')
        plt.plot(self.test_metrics[0]/len(x_test), label='test')
        plt.legend()
        plt.yscale('log')
        plt.savefig('3' + self.version + '_MCS' + str(self.mcs_i + 1) + '_TestAllFoldsLoss.png')
        plt.close()

        # predict model
        self.predictions = self.model.predict(x_test)
        print('self.predictions', np.shape(self.predictions))

        # save weights of model
        self.model.save_weights('3' + self.version + '_MCS' + str(self.mcs_i + 1) + 'TestAllFolds.h5')
        print('weights saved for MCS: ', self.mcs_i+1)

        return self.predictions, self.test_metrics, self.train_metrics
        # return True


def import_export(str_item):
    items = []
    with open(str_item + '.csv', newline='\n') as ff:
        ff_reader = csv.reader(ff, delimiter=',')
        for r in ff_reader:
            items.append([float(itt) for itt in r])
    return items


if __name__ == '__main__':
    start_time = time.time()

    start_mcs = 0
    mcs = 100

    observations = 6000
    x_val_size = 1000

    epochs = 10000
    batch_size = x_val_size
    
    nr_models = 9

    # objects to save for all mcs
    predictions = np.zeros([mcs, nr_models, x_val_size, 1])
    test_scores = np.zeros([mcs, nr_models + 1, epochs])
    train_scores = np.zeros([mcs, nr_models + 1, epochs])
    
    for mcs_ij in range(start_mcs, mcs):
        print('\nmcs_ij', mcs_ij + 1)

        # create class object
        nn = DensityNeuralNetwork(mcs_i=mcs_ij, version='ProposedVersion', hidden_layers=[1, 2, 3],
                                  hidden_neurons=[10, 25, 50], epoch=epochs, batch=x_val_size,
                                  obs=observations, val_size=x_val_size, n_nodes=2)

        # prepare sets
        nn.create_import_set(obs=6000, val_size=x_val_size,
                             str_x='bivariate_poisson_6000_mcs_100_x_original_set.csv',
                             str_index='bivariate_poisson_6000_mcs_100_x_index_set.csv')

        predictions[mcs_ij], test_scores[mcs_ij], train_scores[mcs_ij] = nn.train_cdf(mcs_ij)


        del nn
        
        print(np.shape(predictions[mcs_ij]))
        print(np.shape(test_scores[mcs_ij]))
        print(np.shape(train_scores[mcs_ij]))
        
        export_nrmodels_folds('univariatemixedn_', mcs, 'predictions_TV', predictions[mcs_ij], x_val_size)
        export_nrmodels_folds('univariatemixedn_', mcs, 'test_scores_TV', test_scores[mcs_ij], epochs)
        export_nrmodels_folds('univariatemixedn_', mcs, 'train_scores_TV', train_scores[mcs_ij], epochs)

    predictions = np.reshape(predictions, (mcs, nr_models, x_val_size))
    print('\nnp.shape(predictions)', np.shape(predictions))
    print('np.shape(test_scores)', np.shape(test_scores))
    print('np.shape(train_scores)', np.shape(train_scores))

    export_mcs_nrmodels_folds('3testtrivariatemixedpoisson_', mcs, 'predictions_MV', predictions, x_val_size)
    export_mcs_nrmodels_folds('3testtrivariatemixedpoisson_', mcs, 'test_scores_MV', test_scores, epochs)
    export_mcs_nrmodels_folds('3testtrivariatemixedpoisson_', mcs, 'train_scores_MV', train_scores, epochs)

    end_time = time.time()
    print('\nScript was running for ', (end_time-start_time)/60, ' minutes')
