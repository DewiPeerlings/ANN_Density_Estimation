import numpy as np
import time
import csv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors
from matplotlib import cm

from scipy.integrate import simps
import copy
import math

from functools import reduce
import operator


class Data:
    def __init__(self, xx, n_nodes):
        self.n_nodes = n_nodes

        # import sets
        self.x_vers = xx
        self.y_vers = None

        self.l_1 = 4
        self.l_2 = 7
        self.l_12 = 10

    def create_target_set(self, step_size=0.001):
        if self.n_nodes == 1:
            axis = np.arange(np.min(self.x_vers), np.max(self.x_vers) + step_size, step_size)
            axis[0] = -math.inf
            grid = np.zeros([np.shape(axis)[0]])
            for m in range(len(self.x_vers)):
                grid[np.where(self.x_vers[m][0] >= axis)[0][-1]] += 1
            cdf = np.cumsum(grid, axis=0)
            cdf /= cdf[-1]
            self.y_vers = np.array([cdf[np.where(self.x_vers[m][0] >= axis)[0][-1]] for m in range(len(self.x_vers))])
        if self.n_nodes == 2:
            axis = np.arange(np.min(self.x_vers), np.max(self.x_vers) + step_size, step_size)
            axis[0] = -math.inf
            grid = np.zeros([np.shape(axis)[0], np.shape(axis)[0]])
            for m in range(len(self.x_vers)):
                grid[np.where(self.x_vers[m][0] >= axis)[0][-1], np.where(self.x_vers[m][1] >= axis)[0][-1]] += 1
            cdf = np.cumsum(np.cumsum(grid, axis=0), axis=1)
            cdf /= cdf[-1][-1]
            self.y_vers = np.array([cdf[np.where(self.x_vers[m][0] >= axis)[0][-1],
                                        np.where(self.x_vers[m][1] >= axis)[0][-1]] for m in range(len(self.x_vers))])
        if self.n_nodes == 3:
            axis = np.arange(np.min(self.x_vers), np.max(self.x_vers) + step_size, step_size)
            axis[0] = -math.inf
            grid = np.zeros([np.shape(axis)[0], np.shape(axis)[0], np.shape(axis)[0]])
            for m in range(len(self.x_vers)):
                grid[np.where(self.x_vers[m][0] >= axis)[0][-1], np.where(self.x_vers[m][1] >= axis)[0][-1],
                     np.where(self.x_vers[m][2] >= axis)[0][-1]] += 1
            cdf = np.cumsum(np.cumsum(np.cumsum(grid, axis=0), axis=1), axis=2)
            cdf /= cdf[-1][-1][-1]
            self.y_vers = np.array([cdf[np.where(self.x_vers[m][0] >= axis)[0][-1],
                                        np.where(self.x_vers[m][1] >= axis)[0][-1],
                                        np.where(self.x_vers[m][2] >= axis)[0][-1]] for m in range(len(self.x_vers))])


class ModelTrained:
    def __init__(self):
        from keras.optimizers import Adam
        from keras.initializers import glorot_normal
        self.mcs = 100

        self.observations = 6000
        self.x_val_size = 1000

        self.l_1 = 4
        self.l_2 = 7
        self.l_12 = 10

        self.epochs = 10000
        self.batch_size = self.x_val_size
        self.n_split = 5

        self.n_nodes = 2
        self.layers = [1, 2, 3]
        self.neurons = [10, 25, 50]
        self.bias = 'True'

        self.output_act = 0
        self.act = 'tanh'
        self.optimal_model = 10
        self.learn_rate = 0.001
        self.opt = Adam(lr=self.learn_rate)
        self.initial = glorot_normal(seed=123)

        # models
        self.combinations = np.array(np.meshgrid(self.layers, self.neurons)).T.reshape(-1, 2)
        self.nr_models = np.shape(self.combinations)[0]
        main_model_names = dict()
        for i in range(self.nr_models):
            main_model_names[str('Model_' + str(i))] = self.combinations[i]


class ModelPredict:
    def __init__(self, mcs, nr_models, bias, output_act, act, initial, opt, combinations, n_nod, x_val_size,
                 x_vers, y_vers):
        # import from other class
        self.tot_nr_m = nr_models
        self.b = bias
        self.output_a = output_act
        self.a = act
        self.initial = initial
        self.opt = opt
        self.tot_combi = combinations
        self.n_nodes = n_nod
        self.x_val_size = x_val_size

        # create target set
        self.x_vers = x_vers
        self.y_vers = y_vers

        # layer_names
        self.model_inputs = dict()
        self.model_names = dict()

        # create model
        self.small_model = None

        # run manually
        self.nr_m = 1
        self.pred = np.zeros([mcs, self.tot_nr_m, self.x_val_size])

        # pdf total
        self.list_of_set, self.cardinality_of_pi, self.cardinality_of_b = self.tools_fdb(self.n_nodes + 1)
        flat_list = [item for sublist in self.list_of_set for item in sublist]
        flat_unique_list = [list(xx) for xx in set(tuple(xx) for xx in flat_list)]
        self.sorted_flat_unique_list = sorted(flat_unique_list, key=lambda l: (len(l), l))

        self.fin_derivative = np.zeros([mcs, self.tot_nr_m, self.x_val_size])

    @staticmethod
    def l2(y_true, y_pred):
        from keras import backend as kk
        return kk.sum(kk.square(y_true - y_pred))

    def layer_names(self):
        # input layer, hidden layers and output layer
        # set-up
        self.model_inputs[str(0)] = '00'
        for nrr in range(self.tot_nr_m):
            self.model_names[str(nrr + 1)] = dict()
            if self.tot_combi[nrr][0] >= 1:
                for ii in range(self.tot_combi[nrr][0] + 1):
                    self.model_names[str(nrr + 1)][str(ii + 1)] = str(nrr + 1) + str(ii + 1)

    def create_small_model(self, opt_mod):
        from keras.layers import Input
        from keras.layers import Dense
        from keras.models import Model

        # call functions used here
        self.layer_names()

        # input layer
        models_input = Input(shape=(self.n_nodes,), name='00')

        models_layers = dict()
        models_layers[str(opt_mod)] = dict()
        # first hidden layer
        models_layers[str(opt_mod)][str(1)] = Dense(self.tot_combi[opt_mod - 1][1],
                                                    kernel_initializer=self.initial, activation=self.a,
                                                    use_bias=self.b,
                                                    name=self.model_names[str(opt_mod)][str(1)])(models_input)
        # multiple hidden layers
        for h in range(1, self.tot_combi[opt_mod - 1][0]):
            models_layers[str(opt_mod)][str(h + 1)] = \
                Dense(self.tot_combi[opt_mod - 1][1], kernel_initializer=self.initial, activation=self.a,
                      use_bias=self.b,
                      name=self.model_names[str(opt_mod)][str(h + 1)])(models_layers[str(opt_mod)][str(h)])
        # output layer
        if self.output_a != 1:
            models_layers[str(opt_mod)][str(self.tot_combi[opt_mod - 1][0] + 1)] = \
                Dense(1, kernel_initializer=self.initial, use_bias=self.b, name=self.model_names
                [str(opt_mod)][str(self.tot_combi[opt_mod - 1][0] + 1)])(
                    models_layers[str(opt_mod)][str(self.tot_combi[opt_mod - 1][0])])
        else:
            # self.output_act == 1:
            models_layers[str(opt_mod)][str(self.tot_combi[opt_mod - 1][0] + 1)] = \
                Dense(1, kernel_initializer=self.initial, activation='sigmoid', use_bias=self.b,
                      name=self.model_names
                      [str(opt_mod)][str(self.tot_combi[opt_mod - 1][0] + 1)])(
                    models_layers[str(opt_mod)][str(self.tot_combi[opt_mod - 1][0])])

        # setting up small models
        models_seperate = dict()
        if self.nr_m > 1:
            models_seperate[str(opt_mod)] = \
                Model(inputs=models_input,
                      outputs=models_layers[str(opt_mod)][str(self.tot_combi[opt_mod - 1][0] + 1)])
            models_seperate[str(opt_mod)].compile(optimizer=self.opt, loss=self.l2)

        # setting up big model
        self.small_model = Model(inputs=models_input,
                                 outputs=models_layers[str(opt_mod - 1 + 1)][
                                     str(self.tot_combi[opt_mod - 1][0] + 1)])
        self.small_model.compile(optimizer=self.opt, loss=self.l2)
        # self.small_model.summary()

    @staticmethod
    def activation_function(x_vers, a):
        if a == 'sigmoid':
            return 1 / (1 + math.exp(-x_vers))
        elif a == 'tanh':
            return np.tanh(x_vers)

    def run_manually(self, weights, mcs_r, nr_m_r, lay, neu):
        tempp = np.zeros([len(self.x_vers), lay + 1, neu])
        self.pred[mcs_r][nr_m_r - 1] = np.zeros([len(self.x_vers)])

        bi = 0
        kit = 0
        while (lay - kit) >= 0:
            if kit == 0:
                for itt in range(len(self.x_vers)):
                    for jt in range(neu):
                        for lt in range(self.n_nodes):
                            tempp[itt][kit][jt] += self.x_vers[itt][lt] * weights[bi][lt][jt]
                        # bias == True
                        if self.b:
                            tempp[itt][kit][jt] += 1 * weights[bi + 1][jt]
                        tempp[itt][kit][jt] = self.activation_function(tempp[itt][kit][jt], self.a)
                bi += 1 + int(self.b == 'True')
                kit += 1
            elif (lay - kit) > 0 and kit != 0:
                for itt in range(len(self.x_vers)):
                    for jt in range(neu):
                        for lt in range(neu):
                            tempp[itt][kit][jt] += tempp[itt][kit - 1][lt] * weights[bi][lt][jt]
                        # bias == True
                        if self.b:
                            tempp[itt][kit][jt] += 1 * weights[bi + 1][jt]
                        tempp[itt][kit][jt] = self.activation_function(tempp[itt][kit][jt], self.a)
                bi += 1 + int(self.b == 'True')
                kit += 1
            elif (lay - kit) == 0:
                for itt in range(len(self.x_vers)):
                    for lt in range(neu):
                        tempp[itt][kit][0] += tempp[itt][kit - 1][lt] * weights[bi][lt][0]
                        self.pred[mcs_r][nr_m_r - 1][itt] += tempp[itt][kit - 1][lt] * weights[bi][lt][
                            0]
                    # bias == True
                    if self.b:
                        tempp[itt][kit][0] += 1 * weights[bi + 1][0]
                        self.pred[mcs_r][nr_m_r - 1][itt] += 1 * weights[bi + 1][0]
                    if self.output_a == 1:
                        tempp[itt][kit][0] = 1 / (1 + math.exp(-tempp[itt][kit][0]))
                        self.pred[mcs_r][nr_m_r - 1][itt] = 1 / (
                                    1 + math.exp(-self.pred[mcs_r][nr_m_r - 1][itt]))
                kit += 1  # stop while-loop
        return tempp

    # partition of integer set
    def partition(self, collection):
        if len(collection) == 1:
            yield [collection]
            return

        first = collection[0]
        for smaller in self.partition(collection[1:]):
            # insert `first` in each of the subpartition's subsets
            for n, subset in enumerate(smaller):
                yield smaller[:n] + [[first] + subset] + smaller[n + 1:]
            # put `first` in its own subset
            yield [[first]] + smaller

    # calculate pi and b
    def tools_fdb(self, max_x):
        something = list(range(1, max_x))
        # print('set of integers: ', something, '\n')

        list_of_set = []
        cardinality_of_pi = []

        for n, par in enumerate(self.partition(something), 0):
            # print(n, sorted(p))
            list_of_set.append(sorted(par))
        # print('list_of_set: \n', list_of_set)

        for n, itt in enumerate(list_of_set, 1):
            # print(i)
            cardinality_of_pi.append(len(itt))
        # print('cardinality_of_pi: \t', cardinality_of_pi)

        cardinality_of_b = []
        for itt, jt in enumerate(cardinality_of_pi, 0):
            tempp = []
            for kk in range(jt):
                tempp.append(len(list_of_set[itt][kk]))
            cardinality_of_b.append(tempp)
        # print('cardinality_of_b: \t', cardinality_of_b, '\n')

        return list_of_set, cardinality_of_pi, cardinality_of_b

    @staticmethod
    def dg_names(cardinality, vector):
        numerator = 'delta' + str(cardinality) + 'y'
        denominator = ''
        for itt in vector:
            denominator += ' delta x' + str(itt)
        return numerator + ' /' + denominator

    @staticmethod
    def df_names(cardinality_of_pi):
        return str('f') + '^(' + str(cardinality_of_pi) + ')'

    def names_fdb(self, list_of_set, cardinality_of_pi, cardinality_of_b):
        derivative = ''
        for itt in range(len(list_of_set)):
            tempp = ''
            for jt in range(cardinality_of_pi[itt]):
                tempp += ' (' + str(self.dg_names(cardinality_of_b[itt][jt], list_of_set[itt][jt])) + ')'
            derivative += '\t' + str(self.df_names(cardinality_of_pi[itt])) + tempp
        # print('derivative', derivative, '\n')
        return derivative

    def eulerian(self, d, kk):
        eul = [[1] + [0 for _ in range(kk + 1)] for _ in range(d + 1)]
        for itt in range(d + 1):
            for jt in range(min(kk + 1, itt)):
                eul[itt][jt] = (jt + 1) * eul[itt - 1][jt] + (itt - jt) * eul[itt - 1][jt - 1]
        return eul[d][kk]

    def d_sigmoid(self, neu, d, h, z, observations_nr):
        derivative = np.zeros([neu, observations_nr])
        for obs in range(observations_nr):
            for node in range(neu):
                for kk in range(1, d + 1):
                    derivative[node][obs] += math.pow((-1), kk - 1) * self.eulerian(d, kk - 1) * \
                                             math.pow(z[obs][h][node], kk) * math.pow(
                        (1 - z[obs][h][node]), d + 1 - kk)
        return derivative

    @staticmethod
    def n_choose_k(n_nod, kk):
        return math.factorial(n_nod) / (math.factorial(kk) * math.factorial(n_nod - kk))

    def stirling(self, m, kk):
        tempp = 0
        for itt in range(kk):
            tempp += math.pow(-1, itt) * self.n_choose_k(kk, itt) * math.pow(kk - itt, m)
        return 1 / math.factorial(kk) * tempp

    def d_tanh(self, neu, d, h, z, observations_nr):
        derivative = np.zeros([neu, observations_nr])
        for obs in range(observations_nr):
            for node in range(neu):
                tempp = 0
                for kk in range(d + 1):
                    derivative[node][obs] += math.factorial(kk) / math.pow(2, kk) * \
                        self.stirling(d, kk) * math.pow(z[obs][h][node] - 1, kk)
                derivative[node][obs] = math.pow(-2, d) * (z[obs][h][node] + 1) * derivative[node][obs]
        return derivative

    @staticmethod
    def d_linear(neu, fixed, fixed_value, d, w_index, weights):
        if fixed == 1000:
            derivative = np.zeros([neu, neu])
        else:
            derivative = np.zeros([neu, ])

        if d == 1 and fixed == 1000:
            derivative = np.array([[weights[w_index][kk][lt] for lt in range(neu)] for kk in range(neu)])
        elif d == 1 and fixed == 0:
            derivative = np.array([weights[w_index][fixed_value][lt] for lt in range(neu)])
        elif d == 1 and fixed == 1:
            derivative = np.array([weights[w_index][kk][fixed_value] for kk in range(neu)])
        return derivative

    @staticmethod
    def bell_number(n):
        bell = [[0] * (n + 1)] * (n + 1)
        bell[0][0] = 1
        for itt in range(1, n + 1):
            # Explicitly fill for j = 0
            bell[itt][0] = bell[itt - 1][itt - 1]
            # Fill for remaining values of j
            for jt in range(1, itt + 1):
                bell[itt][jt] = bell[itt - 1][jt - 1] + bell[itt][jt - 1]
        return bell[n][0]

    def mini_tools_fdb(self, something):
        # print('set of integers: ', something, '\n')

        list_of_set = []
        cardinality_of_pi = []

        for n, par in enumerate(self.partition(something), 0):
            print(n, sorted(par))
            list_of_set.append(sorted(par))
        # print('list_of_set: \n', list_of_set)

        for n, itt in enumerate(list_of_set, 1):
            cardinality_of_pi.append(len(itt))
        # print('cardinality_of_pi: \t', cardinality_of_pi)

        cardinality_of_b = []
        for itt, jt in enumerate(cardinality_of_pi, 0):
            tempp = []
            for kk in range(jt):
                tempp.append(len(list_of_set[itt][kk]))
            cardinality_of_b.append(tempp)
        return list_of_set, cardinality_of_pi, cardinality_of_b

    def tools_fdb_partial(self, something):
        # something = list(range(r_1, max_x))
        # print('set of integers: ', something, '\n')

        list_of_set = []
        cardinality_of_pi = []

        for n, par in enumerate(self.partition(something), 0):
            # print(n, sorted(p))
            list_of_set.append(sorted(par))
        # print('list_of_set: \n', list_of_set)

        for n, it in enumerate(list_of_set, 1):
            # print(i)
            cardinality_of_pi.append(len(it))
        # print('cardinality_of_pi: \t', cardinality_of_pi)

        cardinality_of_b = []
        for it, jt in enumerate(cardinality_of_pi, 0):
            tem = []
            for kk in range(jt):
                tem.append(len(list_of_set[it][kk]))
            cardinality_of_b.append(tem)
        # print('cardinality_of_b: \t', cardinality_of_b, '\n')
        return list_of_set, cardinality_of_pi, cardinality_of_b

    def pdf_partial(self, wrt_partial, nr_fdb_i, nr_fdb, count_lay, w_index, combinations_ij, temp_saved,
                    observations_nr, weights_ijt, d_previous, sorted_flat_unique_list_big):
        # print('PARTIAL')
        # list_of_set, cardinality_of_pi, cardinality_of_b = tools_fdb_partial(np.min(wrt_partial), n_nod + 1)
        # print(np.min(wrt_partial), list_of_set, cardinality_of_pi, cardinality_of_b)
        # flat_list = [item for sublist in list_of_set for item in sublist]
        # print(flat_list)

        # test
        # print('\nwrt_partial', wrt_partial)
        list_of_set, cardinality_of_pi, cardinality_of_b = self.tools_fdb_partial(wrt_partial)
        flat_list = [item for sublist in list_of_set for item in sublist]

        if len(flat_list) == 1:
            sorted_flat_unique_list = sorted(flat_list)
        else:
            flat_unique_list = [list(xit) for xit in set(tuple(xit) for xit in flat_list)]
            sorted_flat_unique_list = sorted(flat_unique_list, key=lambda l: (len(l), l))

        # pdf step
        for lay, neu in combinations_ij:
            # print('\nlayers', lay)
            # print('neurons', neu)
            # combination_ijt = np.array(np.meshgrid(lay, neu)).T.reshape(-1, 2)
            # print('combination_ij ', combination_ij)

            # for ONLY ONE composite steps

            # layer with hidden neurons or when output layer (sum these)

            # d = np.zeros([nr_fdb, neu, len(sorted_flat_unique_list)])
            d = np.zeros([neu, observations_nr])
            # print('np.shape(d)', np.shape(d))

            # print('count_lay', count_lay)
            # print('w_index', w_index)

            # prepare d1, d2 vectors such that dot-product is possible
            if nr_fdb_i > 0 and (nr_fdb_i + 1) != nr_fdb and nr_fdb_i % 2 != 0:
                d1 = np.zeros([len(list_of_set), neu, neu, observations_nr])
                d2 = np.ones([len(list_of_set), neu, observations_nr])
            else:
                d1 = np.zeros([len(list_of_set), neu, observations_nr])
                d2 = np.ones([len(list_of_set), neu, observations_nr])

            # FIRST DERIVATIVE
            for it_count, it in enumerate(cardinality_of_pi):
                # iterate through |p|
                # print('\nit (|p|) ', it, cardinality_of_pi)
                # non linear
                if nr_fdb_i % 2 == 0:
                    d1[it_count] = self.d_tanh(neu, it, count_lay, temp_saved, observations_nr)
                # linear
                else:
                    if it == 1 and count_lay < lay:
                        for jt in range(neu):
                            d1[it_count][jt] = np.array(
                                [np.array([weights_ijt[w_index][jj][jt] for k_nr in range(observations_nr)])
                                 for jj in range(neu)])
                    elif it == 1 and count_lay == lay:
                        d1[it_count] = np.array([[weights_ijt[w_index][jt][0] for k_nr in range(observations_nr)]
                                                 for jt in range(neu)])
                    else:
                        if count_lay < lay:
                            d1[it_count] = np.zeros([neu, neu, observations_nr])
                        else:
                            d1[it_count] = np.zeros([neu, observations_nr])

            # SECOND DERIVATIVE
            for jt_count, jt in enumerate(cardinality_of_b):
                # print('\njt ', jt, cardinality_of_b)
                # iterate through |B|
                for jt_temp_count, jt_temp in enumerate(cardinality_of_b[jt_count]):
                    pos = [list_of_set[jt_count][jt_temp_count][pos_i] for pos_i in range(jt_temp)]
                    if nr_fdb_i > 0:
                        if len(jt) == 1:
                            d2[jt_count] = d_previous[nr_fdb_i - 1][sorted_flat_unique_list_big.index(pos)]
                        else:
                            d2[jt_count] *= d_previous[nr_fdb_i - 1][sorted_flat_unique_list[
                                                                         sorted_flat_unique_list.index(pos)][0] - 1]
                    else:
                        if len(jt) == 1:
                            # opposite of d1 (linear or non linear)
                            # linear
                            if nr_fdb_i % 2 == 0:
                                if len(pos) == 1 and nr_fdb_i == nr_fdb:
                                    d2[jt_count] = np.array([np.array([
                                        weights_ijt[w_index][jt][0] for jt in range(neu)])
                                        for k_nr in range(observations_nr)])
                                elif len(pos) == 1 and nr_fdb_i == 0:
                                    d2[jt_count] = np.array([np.array([
                                        weights_ijt[w_index][sorted_flat_unique_list[
                                                                 sorted_flat_unique_list.index(pos)][0] - 1][jt]
                                        for k_nr in range(observations_nr)]) for jt in range(neu)])
                                elif len(pos) == 1 and (nr_fdb_i != 0 or nr_fdb_i != nr_fdb):
                                    d2[jt_count] = np.array([np.array([
                                        weights_ijt[w_index][jj] for k_nr in range(observations_nr)])
                                        for jj in range(neu)])
                                else:
                                    d2[jt_count] = np.zeros([neu, observations_nr])
                            # non linear
                            else:
                                d2[jt_count] = self.d_tanh(neu, jt, count_lay, temp_saved, observations_nr)
                        else:
                            # opposite of d1 (linear or non linear)
                            # linear
                            if nr_fdb_i % 2 == 0:
                                if len(pos) == 1 and (nr_fdb_i == 0 or nr_fdb_i == nr_fdb):
                                    d2[jt_count] *= np.array([np.array([
                                        weights_ijt[w_index][sorted_flat_unique_list[
                                                                 sorted_flat_unique_list.index(pos)][0] - 1][jj]
                                        for k_nr in range(observations_nr)]) for jj in range(neu)])
                                elif len(pos) == 1 and (nr_fdb_i != 0 or nr_fdb_i != nr_fdb):
                                    d2[jt_count] *= np.array([np.array([
                                        weights_ijt[w_index][jj] for k_nr in range(observations_nr)])
                                        for jj in range(neu)])
                                else:
                                    d2[jt_count] *= np.zeros([neu, observations_nr])
                            # non linear
                            else:
                                d2[jt_count] *= self.d_tanh(neu, jt, count_lay, temp_saved, observations_nr)
            # SUM FAA DI BRUNO
            if nr_fdb_i == 0 or (nr_fdb_i + 1) == nr_fdb or nr_fdb_i % 2 == 0:
                for d_it in range(len(list_of_set)):
                    for obs in range(observations_nr):
                        d[:, obs] += d1[d_it, :, obs] * d2[d_it, :, obs]
            else:
                for d_it in range(len(list_of_set)):
                    for obs in range(observations_nr):
                        for neu_i in range(neu):
                            d[neu_i, obs] += np.dot(d1[d_it, neu_i, :, obs], d2[d_it, :, obs])
        # end partial derivative
        return d

    def pdf_total(self, mcs_it, count_it, combinations_ij, temp_saved, weights_ijt):
        # list_of_set, cardinality_of_pi, cardinality_of_b = self.tools_fdb(self.n_nodes + 1)
        # print('list_of_set', list_of_set)
        # print('cardinality_of_pi', cardinality_of_pi)
        # print('cardinality_of_b', cardinality_of_b)
        # flat_list = [item for sublist in list_of_set for item in sublist]
        # print('flat list', flat_list)
        # flat_unique_list = [list(xx) for xx in set(tuple(xx) for xx in flat_list)]
        # print('flat_unique_list', flat_unique_list)
        # sorted_flat_unique_list = sorted(flat_unique_list, key=lambda l: (len(l), l))
        # print('sorted_flat_unique_list', sorted_flat_unique_list)
        # # print(sorted_flat_unique_list.index([1,2,3,4]))

        # pdf step
        for lay, neu in combinations_ij:
            # CHECK HOW MANY COMPOSITE FUNCTIONS
            # if 1 layer then 3
            # if 2 layers then 5
            # if 3 layers then 7
            nr_composite = 3 + 2 * (lay - 1)
            # print('Number of composite functions: ', nr_composite)

            # CHECK HOW MANY EXECUTIONS OF FAA DI BRUNO
            # if 3 composite functions then 2
            # if 5 composite functions then 4
            # if 7 composite functions then 6
            nr_fdb = nr_composite - 1 + self.output_a
            # print('Number of Faa di Bruno executions: ', nr_fdb)

            count_lay = 0
            w_index = 0
            d = np.zeros([nr_fdb, len(self.sorted_flat_unique_list), neu, self.x_val_size])

            # for all composite steps
            for nr_fdb_i in range(nr_fdb):
                # print('nr_fdb_i', nr_fdb_i, nr_fdb)
                # layer with hidden neurons or when output layer (sum these)

                # prepare d1, d2 vectors such that dot-product is possible
                if nr_fdb_i > 0 and (nr_fdb_i + 1) != nr_fdb and nr_fdb_i % 2 != 0:
                    d1 = np.zeros([len(self.list_of_set), neu, neu, self.x_val_size])
                    d2 = np.ones([len(self.list_of_set), neu, self.x_val_size])
                else:
                    d1 = np.zeros([len(self.list_of_set), neu, self.x_val_size])
                    d2 = np.ones([len(self.list_of_set), neu, self.x_val_size])

                # FIRST DERIVATIVE
                for it_count, it in enumerate(self.cardinality_of_pi):
                    # iterate through |p|
                    # print('\nit (|p|) ', it, self.cardinality_of_pi)
                    # non linear
                    if nr_fdb_i % 2 == 0:
                        d1[it_count] = self.d_tanh(neu, it, count_lay, temp_saved, self.x_val_size)
                    # linear
                    else:
                        if it == 1 and count_lay < lay:
                            for jj in range(neu):
                                d1[it_count][jj] = np.array([np.array([
                                    weights_ijt[w_index][jjj][jj] for k_nr in range(self.x_val_size)])
                                    for jjj in range(neu)])
                        elif it == 1 and count_lay == lay:
                            d1[it_count] = np.array(
                                [[weights_ijt[w_index][jt][0] for k_nr in range(self.x_val_size)]
                                 for jt in range(neu)])
                        else:
                            if count_lay < lay:
                                d1[it_count] = np.zeros([neu, neu, self.x_val_size])
                            else:
                                d1[it_count] = np.zeros([neu, self.x_val_size])

                # SECOND DERIVATIVE
                for jt_count, jt in enumerate(self.cardinality_of_b):
                    # print('\njt ', jt, self.cardinality_of_b)
                    # iterate through |B|
                    for jt_temp_count, jt_temp in enumerate(self.cardinality_of_b[jt_count]):
                        pos = [self.list_of_set[jt_count][jt_temp_count][pos_i] for pos_i in range(jt_temp)]
                        if nr_fdb_i > 0:
                            if len(jt) == 1:
                                d2[jt_count] = d[nr_fdb_i - 1][self.sorted_flat_unique_list.index(pos)]
                            else:
                                d2[jt_count] *= d[nr_fdb_i - 1][self.sorted_flat_unique_list.index(pos)]
                        else:
                            if len(jt) == 1:
                                # opposite of d1 (linear or non linear)
                                # linear
                                if nr_fdb_i % 2 == 0:
                                    if len(pos) == 1 and nr_fdb_i == nr_fdb:
                                        d2[jt_count] = np.array([np.array([
                                            weights_ijt[w_index][jt][0] for k_nr in range(self.x_val_size)])
                                            for jt in range(neu)])

                                    elif len(pos) == 1 and nr_fdb_i == 0:
                                        d2[jt_count] = np.array([np.array([
                                            weights_ijt[w_index][0][jt] for k_nr in range(self.x_val_size)])
                                            for jt in range(neu)])

                                    elif len(pos) == 1 and (nr_fdb_i != 0 or nr_fdb_i != nr_fdb):
                                        d2[jt_count] = np.array([np.array([
                                            weights_ijt[w_index][jj] for k_nr in range(self.x_val_size)])
                                            for jj in range(neu)])

                                    else:
                                        d2[jt_count] = np.zeros([neu, self.x_val_size])
                                # non linear
                                else:
                                    d2[jt_count] = self.d_tanh(neu, jt, count_lay, temp_saved, self.x_val_size)
                            else:
                                # opposite of d1 (linear or non linear)
                                # linear
                                if nr_fdb_i % 2 == 0:
                                    if len(pos) == 1 and (nr_fdb_i == 0 or nr_fdb_i == nr_fdb):
                                        d2[jt_count] *= np.array([np.array([
                                            weights_ijt[w_index][
                                                self.sorted_flat_unique_list[
                                                    self.sorted_flat_unique_list.index(pos)][0] - 1][jt]
                                            for k_nr in range(self.x_val_size)]) for jt in range(neu)])
                                    elif len(pos) == 1 and (nr_fdb_i != 0 or nr_fdb_i != nr_fdb):
                                        d2[jt_count] *= np.array([np.array([
                                            weights_ijt[w_index][jj] for k_nr in range(self.x_val_size)])
                                            for jj in range(neu)])
                                    else:
                                        d2[jt_count] *= np.zeros([neu, self.x_val_size])
                                # non linear
                                else:
                                    d2[jt_count] *= self.d_tanh(neu, jt, count_lay, temp_saved, self.x_val_size)
                # SUM FAA DI BRUNO
                if nr_fdb_i == 0 or (nr_fdb_i + 1) == nr_fdb or nr_fdb_i % 2 == 0:
                    for d_it in range(len(self.list_of_set)):
                        for obs in range(self.x_val_size):
                            d[nr_fdb_i, -1, :, obs] += d1[d_it, :, obs] * d2[d_it, :, obs]
                else:
                    for d_it in range(len(self.list_of_set)):
                        for obs in range(self.x_val_size):
                            for neu_i in range(neu):
                                d[nr_fdb_i, -1, neu_i, obs] += np.dot(d1[d_it, neu_i, :, obs], d2[d_it, :, obs])

                # derivative done, now: lower derivatives (e.g. D'' w.r.t. x1 and x2, now: D' wrt. x1 and w.r.t. x2)
                for count_partial, partial_i in enumerate(self.sorted_flat_unique_list[0:-1]):
                    # print('PARTIAL: W.R.T. ???')
                    # print('partial_i', partial_i, '\tposition', count_partial)
                    d[nr_fdb_i][count_partial] = self.pdf_partial(partial_i, nr_fdb_i, nr_fdb, count_lay, w_index,
                                                                  combinations_ij, temp_saved, self.x_val_size,
                                                                  weights_ijt, d, self.sorted_flat_unique_list)

                if nr_fdb_i % 2 == 0 or (nr_fdb_i + 1) == nr_fdb:
                    count_lay += 1
                    w_index += 1 + int(self.b == 'True')

        # sum
        final_d = np.zeros([self.x_val_size, ])
        for neu_j in range(neu):
            final_d += d[-1, -1, neu_j, :]

        self.fin_derivative[mcs_it, count_it - 1] = copy.deepcopy(final_d)


class ModelMetrics:
    def __init__(self, mcs, nr_models):
        self.nr_models = nr_models

        self.data_info = ModelTrained()

        # CDF cross-validation training metrics
        self.optimal_models = np.zeros((mcs, ), dtype=int)
        self.cdf_ft = np.zeros((mcs, nr_models))
        self.cdf_fa = np.zeros((mcs, nr_models))
        self.cdf_at = np.zeros((mcs, ))

        # PDF cross-validation training metrics
        self.pdf_fa = np.zeros((mcs, nr_models))

    def optimal_model(self, mcs_it, loss_metrics):
        self.cdf_ft[mcs_it] = loss_metrics[mcs_it, 1:10, -1]
        self.optimal_models[mcs_it] = np.argmin(self.cdf_ft[mcs_it]) + 1

    def check_optimal_model(self, mcs_it, predict, test_target):
        check = np.array([np.sum(np.array([
            np.square(predict[mcs_it][model_count][itt] - test_target[mcs_it][itt])
            for itt in range(len(test_target[mcs_it]))]))
            for model_count in range(self.nr_models)])
        print('check cdf_ft', check)

    # def compute_cdf_fa(self, mcs_it, predict, val_input):
    #     self.cdf_fa[mcs_it] = np.array([np.sum(np.array([
    #         np.square(predict[mcs_it][model_count] - multivariate_normal.cdf(val_input[mcs_it],
    #                                                                           self.data_info.M_par_1,
    #                                                                           self.data_info.M_par_2))]))
    #         for model_count in range(self.nr_models)])

    def compute_pdf_fa(self, mcs_it, predict_derivative, val_input):
        self.pdf_fa[mcs_it] = np.array([np.sum(np.array([
            np.square(predict_derivative[mcs_it][model_count] - pmf_poisson(val_input[mcs_ij][:, 0], val_input[mcs_ij][:, 1], self.data_info.l_1, self.data_info.l_2, self.data_info.l_12, self.data_info.x_val_size))]))
            for model_count in range(self.nr_models)])

    # def compute_cdf_at(self, mcs_it, test_target, val_input):
    #     self.cdf_at[mcs_it] = np.sum(np.array([
    #         np.square(test_target[mcs_it] - multivariate_normal.cdf(val_input[mcs_it],
    #                                                                  self.data_info.M_par_1,
    #                                                                  self.data_info.M_par_2))]))


def import_export(str_item):
    items = []
    with open('3testtrivariatemixedpoisson__TrainTestOutput_MCS100_' + str_item + '.csv', newline='\n') as ff:
        ff_reader = csv.reader(ff, delimiter=',')
        for r in ff_reader:
            items.append([float(itt) for itt in r])
    return items


def plot_cdf_3d(mcs_it, nr_m, x_1, x_2, target, fitted):
    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(1, 3, 1, projection='3d')

    # ax.scatter(x_1, x_2, actual, color='black', linewidth=2)
    # ax.set_title('true CDF')
    # ax.set_xlim(-4, 4)
    # ax.set_ylim(-4, 4)
    # ax.set_zlim(0, 1)
    # # ax.set_xlabel(str_1)
    # # ax.set_ylabel(str_2)

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(x_1, x_2, target, color='grey', linewidth=2)
    ax.set_title('target CDF')
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 1)
    # ax.set_xlabel(str_1)
    # ax.set_ylabel(str_2)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(x_1, x_2, fitted, color='red', linewidth=2)
    ax.set_title('estimated CDF')
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 1)
    # ax.set_xlabel(str_1)
    # ax.set_ylabel(str_2)
    # plt.savefig('2cdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_fold' + str(f_it) + '_dim' +
    # str(str_1) + '_' + str(str_2) + '_3D.pdf')
    plt.savefig('4cdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_3D.pdf')
    plt.show(block=False)
    plt.close()

    # fig = plt.figure(figsize=(7.4, 5.8))
    # ax = Axes3D(fig)
    # ax.scatter(x_1, x_2, actual, color='black', linewidth=2)
    # ax.set_title('true CDF', pad=15.0)
    # ax.set_xlim(-4, 4)
    # ax.set_ylim(-4, 4)
    # ax.set_zlim(0, 1)
    # plt.savefig('4cdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_3DTrue.pdf')
    # plt.show(block=False)
    # plt.close()

    fig = plt.figure(figsize=(7.4, 5.8))
    ax = Axes3D(fig)
    ax.scatter(x_1, x_2, fitted, color='red', linewidth=2)
    ax.set_title('estimated CDF', pad=15.0)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 1)
    plt.savefig('4cdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_3DEstimated.pdf')
    plt.show(block=False)
    plt.close()

    fig = plt.figure(figsize=(7.4, 5.8))
    ax = Axes3D(fig)
    ax.scatter(x_1, x_2, target, color='grey', linewidth=2)
    ax.set_title('target CDF', pad=15.0)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 1)
    plt.savefig('4cdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_3DTarget.pdf')
    plt.show(block=False)
    plt.close()


def plot_cdf_2d(mcs_it, nr_m, x_1, x_2, target, fitted, min_, max_):
    normalize = matplotlib.colors.Normalize(vmin=min_, vmax=max_)
    colormap = cm.gray  # or any other colormap

    # fig, ax = plt.subplots()
    # img = ax.scatter(x_1, x_2, c=actual, norm=normalize, cmap=colormap, s=5, linewidth=5)
    # # ax.set_title('true CDF')
    # plt.axis([-4, 4, -4, 4])
    # fig.colorbar(img, ax=ax)
    # fig.tight_layout()
    # plt.savefig('4cdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_2DTrue.pdf')
    # plt.show(block=False)
    # plt.close()

    fig, ax = plt.subplots()
    img = ax.scatter(x_1, x_2, c=fitted, norm=normalize, cmap=colormap, s=5, linewidth=5)
    # ax.set_title('estimated CDF')
    plt.axis([0, 40, 0, 40])
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    plt.savefig('4cdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_2DEstimated.pdf')
    plt.show(block=False)
    plt.close()

    fig, ax = plt.subplots()
    img = ax.scatter(x_1, x_2, c=target, norm=normalize, cmap=colormap, s=5, linewidth=5)
    # ax.set_title('target CDF')
    plt.axis([0, 40, 0, 40])
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    plt.savefig('4cdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_2DTarget.pdf')
    plt.show(block=False)
    plt.close()

    diff = fitted-target
    min_ = np.min(diff)
    max_ = np.max(diff)

    fig, ax = plt.subplots()
    colormap = cm.seismic  # or any other colormap
    normalize = matplotlib.colors.TwoSlopeNorm(vmin=min_, vcenter=0, vmax=max_)
    img = ax.scatter(x_1, x_2, c=diff, s=5, norm=normalize, cmap=colormap, linewidth=5)
    plt.axis([0, 40, 0, 40])
    fig.colorbar(img, ax=ax)
    # img.set_clim(vmin=min, vmax=max)
    fig.tight_layout()
    plt.savefig('4cdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_2DDiffTargetEstimated.pdf')
    plt.show(block=False)
    plt.close()

    s2_diff = np.square(fitted-target)
    min_ = np.min(s2_diff)
    max_ = np.max(s2_diff)

    fig, ax = plt.subplots()
    colormap = cm.Reds  # or any other colormap
    normalize = matplotlib.colors.Normalize(vmin=min_, vmax=max_)
    img = ax.scatter(x_1, x_2, c=diff, s=5, norm=normalize, cmap=colormap, linewidth=5)
    plt.axis([0, 40, 0, 40])
    fig.colorbar(img, ax=ax)
    # img.set_clim(vmin=min, vmax=max)
    fig.tight_layout()
    plt.savefig('4cdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_2DS2TargetEstimated.pdf')
    plt.show(block=False)
    plt.close()


def plot_pdf_2d(mcs_it, nr_m, x_1, x_2, actual_pdf, fitted_pdf, min_, max_):
    normalize = matplotlib.colors.Normalize(vmin=min_, vmax=max_)
    colormap = cm.gray  # or any other colormap

    fig, ax = plt.subplots()
    img = ax.scatter(x_1, x_2, c=actual_pdf, norm=normalize, cmap=colormap, s=5, linewidth=5)
    # ax.set_title('true PDF')
    plt.axis([0, 40, 0, 40])
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    plt.savefig('4pdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_2DTrue.pdf')
    plt.show(block=False)
    plt.close()

    fig, ax = plt.subplots()
    img = ax.scatter(x_1, x_2, c=fitted_pdf, norm=normalize, cmap=colormap, s=5, linewidth=5)
    # ax.set_title('estimated PDF')
    plt.axis([0, 40, 0, 40])
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    plt.savefig('4pdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_2DEstimated.pdf')
    plt.show(block=False)
    plt.close()

    diff = fitted_pdf - actual_pdf
    min_ = np.min(diff)
    max_ = np.max(diff)

    fig, ax = plt.subplots()
    colormap = cm.seismic  # or any other colormap
    normalize = matplotlib.colors.TwoSlopeNorm(vmin=min_, vcenter=0, vmax=max_)
    img = ax.scatter(x_1, x_2, c=diff, s=5, norm=normalize, cmap=colormap, linewidth=5)
    plt.axis([0, 40, 0, 40])
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    plt.savefig('4pdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_2DDiffTrueEstimated.pdf')
    plt.show(block=False)
    plt.close()

    s2_diff = np.square(fitted_pdf - actual_pdf)
    min_ = np.min(s2_diff)
    max_ = np.max(s2_diff)

    fig, ax = plt.subplots()
    colormap = cm.OrRd  # or any other colormap
    normalize = matplotlib.colors.Normalize(vmin=min_, vmax=max_)
    img = ax.scatter(x_1, x_2, c=diff, s=5, norm=normalize, cmap=colormap, linewidth=5)
    plt.axis([0, 40, 0, 40])
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    plt.savefig('4pdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_2DS2TargetEstimated.pdf')
    plt.show(block=False)
    plt.close()


def plot_pdf_3d(mcs_it, nr_m, x_1, x_2, actual_pdf, fitted_pdf):
    # (x_1, str_1, x_2, str_2, actual_pdf, fitted_pdf, mcs_selected, model_selected)
    # ax = plt.axes(projection='3d')
    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 0.015)
    ax.scatter(x_1, x_2, actual_pdf, c='black', linewidth=2)
    ax.set_title('true PDF')

    # ax.set_xlabel(str_1)
    # ax.set_ylabel(str_2)
    # ax.set_zlabel('Z axis')
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 0.015)
    ax.scatter(x_1, x_2, fitted_pdf, c='red', linewidth=2)
    ax.set_title('estimated PDF')

    # ax.set_xlabel(str_1)
    # ax.set_ylabel(str_2)
    # # ax.set_zlabel('Z axis')
    # plt.savefig('2pdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_fold' + str(f_it) +
    #             '_dim' + str(str_1) + '_' + str(str_2) + '_3D.pdf')
    plt.savefig('4pdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_3D.pdf')
    plt.show(block=False)
    plt.close()

    fig = plt.figure(figsize=(7.4, 5.8))
    ax = Axes3D(fig)
    ax.scatter(x_1, x_2, actual_pdf, color='black', linewidth=2)
    ax.set_title('true CDF', pad=15.0)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 0.015)
    plt.savefig('4pdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_3DTrue.pdf')
    plt.show(block=False)
    plt.close()

    fig = plt.figure(figsize=(7.4, 5.8))
    ax = Axes3D(fig)
    ax.scatter(x_1, x_2, fitted_pdf, color='red', linewidth=2)
    ax.set_title('estimated CDF', pad=15.0)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_zlim(0, 0.015)
    plt.savefig('4pdf_mcs' + str(mcs_it + 1) + '_model' + str(nr_m) + '_3DEstimated.pdf')
    plt.show(block=False)
    plt.close()


def pmf_poisson(x_1, x_2, l_1t, l_2t, l_12t, x_val_size_t):
    pmf = []
    for obs_t in range(x_val_size_t):
        temp_sum = 0
        for it in range(0, np.min([int(x_1[obs_t]), int(x_2[obs_t])])):
            a = math.pow(l_1t, int(x_1[obs_t])-it)*math.pow(l_2t, int(x_2[obs_t])-it)*\
                math.pow(l_12t, it)
            b = np.math.factorial(int(x_1[obs_t])-it)*np.math.factorial(int(x_2[obs_t])-it)*\
                np.math.factorial(it)
            temp_sum += a/b
        pmf.append(np.exp(-(l_1t+l_2t+l_12t))*temp_sum)
    return pmf


start_time = time.time()
main_model = ModelTrained()
metrics_model = ModelMetrics(mcs=main_model.mcs, nr_models=main_model.nr_models)

predictions = np.reshape(np.array(import_export('predictions_MV')),
                         (main_model.mcs, main_model.nr_models, main_model.x_val_size))
print('np.shape(predictions)', np.shape(predictions))

test_scores = np.reshape(np.array(import_export('test_scores_MV')),
                         (main_model.mcs, main_model.nr_models + 1, main_model.epochs))
print('np.shape(test_scores)', np.shape(test_scores))
train_scores = np.reshape(np.array(import_export('train_scores_MV')),
                          (main_model.mcs, main_model.nr_models + 1, main_model.epochs))
print('np.shape(train_scores)', np.shape(train_scores))

# import sets
x_original_set = []
with open("bivariate_poisson_6000_mcs_100_x_original_set.csv", newline='\n') as f:
    reader = csv.reader(f, delimiter=',')
    count_row = 0
    for row in reader:
        temp = []
        count_row += 1
        for j in row:
            temp.append(eval(j))
        x_original_set.append(np.array(temp))
count_row = int(count_row / main_model.observations)
x_original_set = np.reshape(np.array(x_original_set), (int(count_row), main_model.observations, main_model.n_nodes))

index_set = []
with open("bivariate_poisson_6000_mcs_100_x_index_set.csv", newline='\n') as f:
    reader = csv.reader(f, delimiter=',')
    for row in reader:
        index_set.append(np.array([eval(row[i]) for i in range(main_model.x_val_size)]))


# save predictions and targets
x_val_saved = []
y_val_saved = []
cdf_predict = []
pdf_predict = []

min_actual = []
min_target = []
min_fitted = []
max_actual = []
max_target = []
max_fitted = []

min_actual_pdf = []
min_fitted_pdf = []
max_actual_pdf = []
max_fitted_pdf = []
for mcs_ij in range(main_model.mcs):
    print('\nmcs_ij', mcs_ij+1)

    # prepare sets
    x_original = x_original_set[mcs_ij]
    index = index_set[mcs_ij].astype(int)
    x_val = x_original[index]
    x_val_saved.append(x_val)
    x = np.delete(x_original, index, 0)

    # validation sets
    # create target set MyVersion
    data_original = Data(xx=x_original, n_nodes=main_model.n_nodes)
    data_original.create_target_set()
    y_val = data_original.y_vers[index]
    y_val_saved.append(y_val)
    # create target set without validation set MyVersion
    data_set = Data(xx=x, n_nodes=main_model.n_nodes)
    data_set.create_target_set()
    y = data_set.y_vers

    print('np.shape(x)', np.shape(x))
    print('np.shape(y)', np.shape(y))

    # metrics (fitted and target) (per model of val set)
    metrics_model.optimal_model(mcs_it=mcs_ij, loss_metrics=test_scores)

    predict_val = ModelPredict(mcs=main_model.mcs, nr_models=main_model.nr_models, bias=main_model.bias,
                               output_act=main_model.output_act, act=main_model.act, initial=main_model.initial,
                               opt=main_model.opt, combinations=main_model.combinations, n_nod=main_model.n_nodes,
                               x_val_size=main_model.x_val_size, x_vers=x_val, y_vers=y_val)

    count = 1
    for i, j in main_model.combinations:
        predict_val.create_small_model(opt_mod=count)
        combination_ij = np.array(np.meshgrid(i, j)).T.reshape(-1, 2)
        # fit and differentiate val CDF
        predict_val.small_model.load_weights('3ProposedVersion_MCS' + str(mcs_ij + 1) + 'TestAllFolds.h5', by_name=True)
        weights_test = predict_val.small_model.get_weights()
        test_temp_saved = predict_val.run_manually(weights=weights_test, mcs_r=mcs_ij, nr_m_r=count, lay=i, neu=j)
        predict_val.pdf_total(mcs_it=mcs_ij, count_it=count, combinations_ij=combination_ij,
                               temp_saved=test_temp_saved, weights_ijt=weights_test)

        # WARNING
        print('number of negative values: ', np.sum(predict_val.fin_derivative[mcs_ij][count - 1] < 0, axis=0))
        predict_val.fin_derivative[mcs_ij][count - 1][predict_val.fin_derivative[mcs_ij][count - 1] < 0] = 0

        # determine range of future colorbars
        # min_actual.append(np.min(multivariate_normal.cdf(x_val, data_set.M_par_1, data_set.M_par_2)))
        # max_actual.append(np.max(multivariate_normal.cdf(x_val, data_set.M_par_1, data_set.M_par_2)))
        min_target.append(np.min(y_val))
        max_target.append(np.max(y_val))
        min_fitted.append(np.min(predict_val.pred[mcs_ij][count - 1]))
        max_fitted.append(np.max(predict_val.pred[mcs_ij][count - 1]))

        min_actual_pdf.append(np.min(pmf_poisson(x_val[:, 0], x_val[:, 1], data_set.l_1, data_set.l_2, data_set.l_12, len(x_val[:, 0]))))
        max_actual_pdf.append(np.max(pmf_poisson(x_val[:, 0], x_val[:, 1], data_set.l_1, data_set.l_2, data_set.l_12, len(x_val[:, 0]))))
        min_fitted_pdf.append(np.min(predict_val.fin_derivative[mcs_ij][count - 1]))
        max_fitted_pdf.append(np.max(predict_val.fin_derivative[mcs_ij][count - 1]))

        count += 1
    print('done predicting\n')

    # CDF metrics (fitted and target) (per model of val set)
    # metrics_model.check_optimal_model(mcs_it=mcs_ij, predict=predict_val.pred, test_target=y_val_saved)

    print('calculating metrics...')

    # CDF metrics (fitted and actual) (per model of val set )
    # metrics_model.compute_cdf_fa(mcs_it=mcs_ij, predict=predict_val.pred, val_input=x_val_saved)
    # print('\tcdf_fa')

    # PDF metrics (fitted and actual) (per model of test set averaged over folds)
    metrics_model.compute_pdf_fa(mcs_it=mcs_ij, predict_derivative=predict_val.fin_derivative, val_input=x_val_saved)
    print('\tpdf_fa')

    # CDF metrics (target and actual) (per model of test set averaged over folds)
    # metrics_model.compute_cdf_at(mcs_it=mcs_ij, test_target=y_val_saved, val_input=x_val_saved)
    # print('\tcdf_at')

    # save predictions
    cdf_predict.append(predict_val.pred[mcs_ij])
    pdf_predict.append(predict_val.fin_derivative[mcs_ij])

    print('optimal model ', np.argmin(metrics_model.cdf_ft[mcs_ij]) + 1)
    print('cdf_ft', metrics_model.cdf_ft[mcs_ij][np.argmin(metrics_model.cdf_ft[mcs_ij])])
    # print('cdf_at', metrics_model.cdf_at[mcs_ij])
    # print('cdf_fa', metrics_model.cdf_fa[mcs_ij][np.argmin(metrics_model.cdf_ft[mcs_ij])])
    print('pdf_fa', metrics_model.pdf_fa[mcs_ij][np.argmin(metrics_model.cdf_ft[mcs_ij])], '\n')

    print('cdf_ft', metrics_model.cdf_ft[mcs_ij])
    # print('cdf_fa', metrics_model.cdf_fa[mcs_ij])
    print('pdf_fa', metrics_model.pdf_fa[mcs_ij])

print('\ncdf_predict', np.shape(cdf_predict))  # (100, 9, 1000)
print('pdf_predict', np.shape(pdf_predict), '\n')  # (100, 9, 1000)


with open("4ProposedVersionTestSet_cdf_predict.csv", "a") as f:
    writer = csv.writer(f)
    for item in cdf_predict:
        for it in item:
            writer.writerow(it)
f.close()

with open("4ProposedVersionTestSet_pdf_predict.csv", "a") as f:
    writer = csv.writer(f)
    for item in pdf_predict:
        for it in item:
            writer.writerow(it)
f.close()


# print('min_actual', np.shape(min_actual))
print('min_target', np.shape(min_target))
print('min_fitted', np.shape(min_fitted))
# print('max_actual', np.shape(max_actual))
print('max_target', np.shape(max_target))
print('max_fitted', np.shape(max_fitted))

print('min_actual_pdf', np.shape(min_actual_pdf))
print('min_fitted_pdf', np.shape(min_fitted_pdf))
print('max_actual_pdf', np.shape(max_actual_pdf))
print('max_fitted_pdf', np.shape(max_fitted_pdf))

minimum_values_cdf = [np.min(min_target), np.min(min_fitted)]
maximum_values_cdf = [np.max(max_target), np.max(max_fitted)]
minimum_values_pdf = [np.min(min_actual_pdf), np.min(min_fitted_pdf)]
maximum_values_pdf = [np.max(max_actual_pdf), np.max(max_fitted_pdf)]

vmin_cdf = np.min(minimum_values_cdf)
vmax_cdf = np.max(maximum_values_cdf)
vmin_pdf = np.min(minimum_values_pdf)
vmax_pdf = np.max(maximum_values_pdf)

print('\nvmin_cdf', vmin_cdf)
print('vmax_cdf', vmax_cdf)
print('vmin_pdf', vmin_pdf)
print('vmax_pdf', vmax_pdf, '\n')

for mcs_ij in range(main_model.mcs):
    count = 1
    for i, j in main_model.combinations:
        # plot CDF fit test set
        plot_cdf_2d(mcs_it=mcs_ij, nr_m=count, x_1=x_val_saved[mcs_ij][:, 0], x_2=x_val_saved[mcs_ij][:, 1],
                    target=y_val_saved[mcs_ij], fitted=cdf_predict[mcs_ij][count - 1], min_=vmin_cdf, max_=vmax_cdf)

        plot_cdf_3d(mcs_it=mcs_ij, nr_m=count, x_1=x_val[:, 0], x_2=x_val[:, 1],
                    target=y_val, fitted=predict_val.pred[mcs_ij][count - 1])

        # plot PDF fit test set
        plot_pdf_2d(mcs_it=mcs_ij, nr_m=count, x_1=x_val_saved[mcs_ij][:, 0], x_2=x_val_saved[mcs_ij][:, 1],
                    actual_pdf=pmf_poisson(x_val_saved[mcs_ij][:, 0], x_val_saved[mcs_ij][:, 1], main_model.l_1, main_model.l_2, main_model.l_12, main_model.x_val_size),
                    fitted_pdf=pdf_predict[mcs_ij][count - 1], min_=vmin_pdf, max_=vmax_pdf)

        plot_pdf_3d(mcs_it=mcs_ij, nr_m=count, x_1=x_val[:, 0], x_2=x_val[:, 1],
                    actual_pdf=pmf_poisson(x_val_saved[mcs_ij][:, 0], x_val_saved[mcs_ij][:, 1], main_model.l_1, main_model.l_2, main_model.l_12, main_model.x_val_size),
                    fitted_pdf=predict_val.fin_derivative[mcs_ij][count - 1])
        count += 1
    print('all models illustrated for mcs ', mcs_ij + 1)

# save metrics
# count models chosen best out of all mcs
unique, counts = np.unique(metrics_model.optimal_models, return_counts=True)
print('\n\nCount models chosen as best out of all mcs ', dict(zip(unique, counts)))

# metrics of all optimal models
print('\ncdf_ft')
print(np.mean(metrics_model.cdf_ft[np.arange(main_model.mcs), metrics_model.optimal_models - 1]))
print('(', np.std(metrics_model.cdf_ft[np.arange(main_model.mcs), metrics_model.optimal_models - 1]), ')')
# print('cdf_at')
# print(np.mean(metrics_model.cdf_at))
# print('(', np.std(metrics_model.cdf_at), ')')
# print('cdf_fa')
# print(np.mean(metrics_model.cdf_fa[np.arange(main_model.mcs), metrics_model.optimal_models - 1]))
# print('(', np.std(metrics_model.cdf_fa[np.arange(main_model.mcs), metrics_model.optimal_models - 1]), ')')
print('pdf_fa')
print(np.mean(metrics_model.pdf_fa[np.arange(main_model.mcs), metrics_model.optimal_models - 1]))
print('(', np.std(metrics_model.pdf_fa[np.arange(main_model.mcs), metrics_model.optimal_models - 1]), ')')

# plot corresponding L2 differences
# cdf_fa_l2diff = np.array([np.square(cdf_predict[mcs_ij][metrics_model.optimal_models[mcs_ij] - 1] -
#                           np.reshape(multivariate_normal.cdf(x_val_saved[mcs_ij], main_model.M_par_1,
#                                                                                    main_model.M_par_2),
# (main_model.x_val_size, ))) for mcs_ij in range(main_model.mcs)])
# print('f1')
# print(np.shape(cdf_predict[0][0]))
# print(np.shape(np.reshape(multivariate_normal.cdf(x_val_saved[0], main_model.M_par_1, main_model.M_par_2),
#                           (main_model.x_val_size, ))))
cdf_ft_l2diff = np.array([np.square(cdf_predict[mcs_ij][metrics_model.optimal_models[mcs_ij] - 1] - y_val_saved[mcs_ij])
                          for mcs_ij in range(main_model.mcs)])
print('f2')
print(np.shape(cdf_predict[0][0]))
print(np.shape(y_val_saved[0]))
# cdf_at_l2diff = np.array([np.square(y_val_saved[mcs_ij] -
#                                     np.reshape(multivariate_normal.cdf(x_val_saved[mcs_ij], main_model.M_par_1,
#                                                                    main_model.M_par_2),
#                                                (main_model.x_val_size, ))) for mcs_ij in range(main_model.mcs)])
# print('f3')
pdf_fa_l2diff = np.array([np.square(pdf_predict[mcs_ij][metrics_model.optimal_models[mcs_ij] - 1] -
                                    np.reshape(pmf_poisson(x_val_saved[mcs_ij][:, 0], x_val_saved[mcs_ij][:, 1], main_model.l_1, main_model.l_2, main_model.l_12, main_model.x_val_size),
                                               (main_model.x_val_size, ))) for mcs_ij in range(main_model.mcs)])
print('f4')
print(np.shape(pdf_predict[0][0]))
print(np.shape(np.reshape(pmf_poisson(x_val_saved[0][:, 0], x_val_saved[0][:, 1], main_model.l_1, main_model.l_2, main_model.l_12, main_model.x_val_size),
                          (main_model.x_val_size, ))))

x_1 = np.array(x_val_saved)[np.arange(main_model.mcs), :, 0]
x_2 = np.array(x_val_saved)[np.arange(main_model.mcs), :, 1]

# remove below 95th percentile
# cdf_fa_index = np.argwhere(cdf_fa_l2diff.flatten() > np.percentile(cdf_fa_l2diff.flatten(), 95))[:, 0]
cdf_ft_index = np.argwhere(cdf_ft_l2diff.flatten() > np.percentile(cdf_ft_l2diff.flatten(), 95))[:, 0]
# cdf_at_index = np.argwhere(cdf_at_l2diff.flatten() > np.percentile(cdf_at_l2diff.flatten(), 95))[:, 0]
pdf_fa_index = np.argwhere(pdf_fa_l2diff.flatten() > np.percentile(pdf_fa_l2diff.flatten(), 95))[:, 0]

# cdf_fa_l2diff_select = cdf_fa_l2diff.flatten()[cdf_fa_index]
cdf_ft_l2diff_select = cdf_ft_l2diff.flatten()[cdf_ft_index]
# cdf_at_l2diff_select = cdf_fa_l2diff.flatten()[cdf_at_index]
pdf_fa_l2diff_select = pdf_fa_l2diff.flatten()[pdf_fa_index]

# cdf_fa_order_index = np.argsort(cdf_fa_l2diff.flatten())
cdf_ft_order_index = np.argsort(cdf_ft_l2diff.flatten())
# cdf_at_order_index = np.argsort(cdf_at_l2diff.flatten())
pdf_fa_order_index = np.argsort(pdf_fa_l2diff.flatten())

# cdf_fa_order_index_select = np.argsort(cdf_fa_l2diff_select)
cdf_ft_order_index_select = np.argsort(cdf_ft_l2diff_select)
# cdf_at_order_index_select = np.argsort(cdf_at_l2diff_select)
pdf_fa_order_index_select = np.argsort(pdf_fa_l2diff_select)

# cdf_fa_x_1_select = x_1.flatten()[cdf_fa_index]
cdf_ft_x_1_select = x_1.flatten()[cdf_ft_index]
# cdf_at_x_1_select = x_1.flatten()[cdf_at_index]
pdf_fa_x_1_select = x_1.flatten()[pdf_fa_index]

# cdf_fa_x_2_select = x_2.flatten()[cdf_fa_index]
cdf_ft_x_2_select = x_2.flatten()[cdf_ft_index]
# cdf_at_x_2_select = x_2.flatten()[cdf_at_index]
pdf_fa_x_2_select = x_2.flatten()[pdf_fa_index]

# fig, ax = plt.subplots()
# img = ax.scatter(x_1.flatten()[cdf_fa_order_index], x_2.flatten()[cdf_fa_order_index],
#                  cdf_fa_l2diff.flatten()[cdf_fa_order_index], c=cdf_fa_l2diff.flatten()[cdf_fa_order_index],
#                  vmin=0, vmax=np.max(cdf_fa_l2diff.flatten()[cdf_fa_order_index]), linewidth=7.5)
# plt.axis([0, 40, 0, 40])
# fig.colorbar(img, ax=ax)
# fig.tight_layout()
# plt.savefig('4ProposedVersionTestSet_cdf_fa_L2diff.pdf')
# plt.show(block=False)
# plt.close()

# fig, ax = plt.subplots()
# img = ax.scatter(cdf_fa_x_1_select[cdf_fa_order_index_select], cdf_fa_x_2_select[cdf_fa_order_index_select],
#                  cdf_fa_l2diff_select[cdf_fa_order_index_select], c=cdf_fa_l2diff_select[cdf_fa_order_index_select],
#                  vmin=0, vmax=np.max(cdf_fa_l2diff_select[cdf_fa_order_index_select]), linewidth=7.5)
# plt.axis([-4, 4, -4, 4])
# fig.colorbar(img, ax=ax)
# fig.tight_layout()
# plt.savefig('4ProposedVersionTestSet_cdf_fa_l2diff_percentile.pdf')
# plt.show(block=False)
# plt.close()

fig, ax = plt.subplots()
img = ax.scatter(x_1.flatten()[cdf_ft_order_index], x_2.flatten()[cdf_ft_order_index],
                 cdf_ft_l2diff.flatten()[cdf_ft_order_index], c=cdf_ft_l2diff.flatten()[cdf_ft_order_index],
                 vmin=0, vmax=np.max(cdf_ft_l2diff.flatten()[cdf_ft_order_index]), linewidth=7.5)
plt.axis([0, 40, 0, 40])
fig.colorbar(img, ax=ax)
fig.tight_layout()
plt.savefig('4ProposedVersionTestSet_cdf_ft_L2diff.pdf')
plt.show(block=False)
plt.close()

fig, ax = plt.subplots()
img = ax.scatter(cdf_ft_x_1_select[cdf_ft_order_index_select], cdf_ft_x_2_select[cdf_ft_order_index_select],
                 cdf_ft_l2diff_select[cdf_ft_order_index_select], c=cdf_ft_l2diff_select[cdf_ft_order_index_select],
                 vmin=0, vmax=np.max(cdf_ft_l2diff_select[cdf_ft_order_index_select]), linewidth=7.5)
plt.axis([0, 40, 0, 40])
fig.colorbar(img, ax=ax)
fig.tight_layout()
plt.savefig('4ProposedVersionTestSet_cdf_ft_l2diff_percentile.pdf')
plt.show(block=False)
plt.close()

# fig, ax = plt.subplots()
# img = ax.scatter(x_1.flatten()[cdf_at_order_index], x_2.flatten()[cdf_at_order_index],
#                  cdf_at_l2diff.flatten()[cdf_at_order_index], c=cdf_at_l2diff.flatten()[cdf_at_order_index],
#                  vmin=0, vmax=np.max(cdf_at_l2diff.flatten()[cdf_at_order_index]), linewidth=7.5)
# plt.axis([-4, 4, -4, 4])
# fig.colorbar(img, ax=ax)
# fig.tight_layout()
# plt.savefig('4ProposedVersionTestSet_cdf_at_L2diff.pdf')
# plt.show(block=False)
# plt.close()

# fig, ax = plt.subplots()
# img = ax.scatter(cdf_at_x_1_select[cdf_at_order_index_select], cdf_at_x_2_select[cdf_at_order_index_select],
#                  cdf_at_l2diff_select[cdf_at_order_index_select], c=cdf_at_l2diff_select[cdf_at_order_index_select],
#                  vmin=0, vmax=np.max(cdf_at_l2diff_select[cdf_at_order_index_select]), linewidth=7.5)
# plt.axis([-4, 4, -4, 4])
# fig.colorbar(img, ax=ax)
# fig.tight_layout()
# plt.savefig('4ProposedVersionTestSet_cdf_at_l2diff_percentile.pdf')
# plt.show(block=False)
# plt.close()

fig, ax = plt.subplots()
img = ax.scatter(x_1.flatten()[pdf_fa_order_index], x_2.flatten()[pdf_fa_order_index],
                 pdf_fa_l2diff.flatten()[pdf_fa_order_index], c=pdf_fa_l2diff.flatten()[pdf_fa_order_index],
                 vmin=0, vmax=np.max(pdf_fa_l2diff.flatten()[pdf_fa_order_index]), linewidth=7.5)
plt.axis([0, 40, 0, 40])
fig.colorbar(img, ax=ax)
fig.tight_layout()
plt.savefig('4ProposedVersionTestSet_pdf_fa_L2diff.pdf')
plt.show(block=False)
plt.close()

fig, ax = plt.subplots()
img = ax.scatter(pdf_fa_x_1_select[pdf_fa_order_index_select], pdf_fa_x_2_select[pdf_fa_order_index_select],
                 pdf_fa_l2diff_select[pdf_fa_order_index_select], c=pdf_fa_l2diff_select[pdf_fa_order_index_select],
                 vmin=0, vmax=np.max(pdf_fa_l2diff_select[pdf_fa_order_index_select]), linewidth=7.5)
plt.axis([0, 40, 0, 40])
fig.colorbar(img, ax=ax)
fig.tight_layout()
plt.savefig('4ProposedVersionTestSet_pdf_fa_l2diff_percentile.pdf')
plt.show(block=False)
plt.close()

# with open("4ProposedVersionTestSet_cdf_fa_l2_diff_modeloptimal.csv", "a") as f:
#     writer = csv.writer(f)
#     for item in cdf_fa_l2diff:
#         writer.writerow(item)
# f.close()

with open("4ProposedVersionTestSet_cdf_ft_l2_diff_modeloptimal.csv", "a") as f:
    writer = csv.writer(f)
    for item in cdf_ft_l2diff:
        writer.writerow(item)
f.close()

# with open("4ProposedVersionTestSet_cdf_at_l2_diff_modeloptimal.csv", "a") as f:
#     writer = csv.writer(f)
#     for item in cdf_at_l2diff:
#         writer.writerow(item)
# f.close()

with open("4ProposedVersionTestSet_pdf_fa_l2_diff_modeloptimal.csv", "a") as f:
    writer = csv.writer(f)
    for item in pdf_fa_l2diff:
        writer.writerow(item)
f.close()

for nr in range(metrics_model.nr_models):
    print('\nmodel ', nr+1)
    print('cdf_ft')
    print(np.mean(metrics_model.cdf_ft[np.arange(main_model.mcs), nr]))
    print('(', np.std(metrics_model.cdf_ft[np.arange(main_model.mcs), nr]), ')')
    # print('cdf_at')
    # print(np.mean(metrics_model.cdf_at))
    # print('(', np.std(metrics_model.cdf_at), ')')
    # print('cdf_fa')
    # print(np.mean(metrics_model.cdf_fa[np.arange(main_model.mcs), nr]))
    # print('(', np.std(metrics_model.cdf_fa[np.arange(main_model.mcs), nr]), ')')
    print('pdf_fa')
    print(np.mean(metrics_model.pdf_fa[np.arange(main_model.mcs), nr]))
    print('(', np.std(metrics_model.pdf_fa[np.arange(main_model.mcs), nr]), ')\n')

    # plot corresponding L2 differences per model nr
    # cdf_fa_l2diff = np.array([np.square(cdf_predict[mcs_ij][nr] -
    #                                     np.reshape(multivariate_normal.cdf(x_val_saved[mcs_ij], main_model.M_par_1,
    #                                                                        main_model.M_par_2),
    #                                                (main_model.x_val_size,))) for mcs_ij in range(main_model.mcs)])
    # print('f1')
    # print(np.shape(cdf_predict[0][0]))
    # print(np.shape(np.reshape(multivariate_normal.cdf(x_val_saved[0], main_model.M_par_1, main_model.M_par_2),
    #                           (main_model.x_val_size,))))
    cdf_ft_l2diff = np.array([np.square(cdf_predict[mcs_ij][nr] - y_val_saved[mcs_ij])
                              for mcs_ij in range(main_model.mcs)])
    print('f2')
    print(np.shape(cdf_predict[0][0]))
    print(np.shape(y_val_saved[0]))
    # cdf_at_l2diff = np.array([np.square(y_val_saved[mcs_ij] -
    #                                     np.reshape(multivariate_normal.cdf(x_val_saved[mcs_ij], main_model.M_par_1,
    #                                                                        main_model.M_par_2),
    #                                                (main_model.x_val_size,))) for mcs_ij in range(main_model.mcs)])
    # print('f3')
    pdf_fa_l2diff = np.array([np.square(pdf_predict[mcs_ij][nr] -
                                        np.reshape(pmf_poisson(x_val_saved[mcs_ij][:, 0], x_val_saved[mcs_ij][:, 1], main_model.l_1, main_model.l_2, main_model.l_12, main_model.x_val_size),
                                                   (main_model.x_val_size,))) for mcs_ij in range(main_model.mcs)])
    print('f4')
    print(np.shape(pdf_predict[0][0]))
    print(np.shape(np.reshape(pmf_poisson(x_val_saved[0][:, 0], x_val_saved[0][:, 1], main_model.l_1, main_model.l_2, main_model.l_12, main_model.x_val_size),
                              (main_model.x_val_size,))))

    # remove below 95th percentile
    # cdf_fa_index = np.argwhere(cdf_fa_l2diff.flatten() > np.percentile(cdf_fa_l2diff.flatten(), 95))[:, 0]
    cdf_ft_index = np.argwhere(cdf_ft_l2diff.flatten() > np.percentile(cdf_ft_l2diff.flatten(), 95))[:, 0]
    # cdf_at_index = np.argwhere(cdf_at_l2diff.flatten() > np.percentile(cdf_at_l2diff.flatten(), 95))[:, 0]
    pdf_fa_index = np.argwhere(pdf_fa_l2diff.flatten() > np.percentile(pdf_fa_l2diff.flatten(), 95))[:, 0]

    # cdf_fa_l2diff_select = cdf_fa_l2diff.flatten()[cdf_fa_index]
    cdf_ft_l2diff_select = cdf_ft_l2diff.flatten()[cdf_ft_index]
    # cdf_at_l2diff_select = cdf_fa_l2diff.flatten()[cdf_at_index]
    pdf_fa_l2diff_select = pdf_fa_l2diff.flatten()[pdf_fa_index]

    # cdf_fa_order_index = np.argsort(cdf_fa_l2diff.flatten())
    cdf_ft_order_index = np.argsort(cdf_ft_l2diff.flatten())
    # cdf_at_order_index = np.argsort(cdf_at_l2diff.flatten())
    pdf_fa_order_index = np.argsort(pdf_fa_l2diff.flatten())

    # cdf_fa_order_index_select = np.argsort(cdf_fa_l2diff_select)
    cdf_ft_order_index_select = np.argsort(cdf_ft_l2diff_select)
    # cdf_at_order_index_select = np.argsort(cdf_at_l2diff_select)
    pdf_fa_order_index_select = np.argsort(pdf_fa_l2diff_select)

    # cdf_fa_x_1_select = x_1.flatten()[cdf_fa_index]
    cdf_ft_x_1_select = x_1.flatten()[cdf_ft_index]
    # cdf_at_x_1_select = x_1.flatten()[cdf_at_index]
    pdf_fa_x_1_select = x_1.flatten()[pdf_fa_index]

    # cdf_fa_x_2_select = x_2.flatten()[cdf_fa_index]
    cdf_ft_x_2_select = x_2.flatten()[cdf_ft_index]
    # cdf_at_x_2_select = x_2.flatten()[cdf_at_index]
    pdf_fa_x_2_select = x_2.flatten()[pdf_fa_index]

    # fig, ax = plt.subplots()
    # img = ax.scatter(x_1.flatten()[cdf_fa_order_index], x_2.flatten()[cdf_fa_order_index],
    #                  cdf_fa_l2diff.flatten()[cdf_fa_order_index], c=cdf_fa_l2diff.flatten()[cdf_fa_order_index],
    #                  vmin=0, vmax=np.max(cdf_fa_l2diff.flatten()[cdf_fa_order_index]), linewidth=7.5)
    # plt.axis([-4, 4, -4, 4])
    # fig.colorbar(img, ax=ax)
    # fig.tight_layout()
    # plt.savefig('4ProposedVersionTestSet_cdf_fa_L2diff_model' + str(nr+1) + '.pdf')
    # plt.show(block=False)
    # plt.close()

    # fig, ax = plt.subplots()
    # img = ax.scatter(cdf_fa_x_1_select[cdf_fa_order_index_select], cdf_fa_x_2_select[cdf_fa_order_index_select],
    #                  cdf_fa_l2diff_select[cdf_fa_order_index_select], c=cdf_fa_l2diff_select[cdf_fa_order_index_select],
    #                  vmin=0, vmax=np.max(cdf_fa_l2diff_select[cdf_fa_order_index_select]), linewidth=7.5)
    # plt.axis([-4, 4, -4, 4])
    # fig.colorbar(img, ax=ax)
    # fig.tight_layout()
    # plt.savefig('4ProposedVersionTestSet_cdf_fa_l2diff_percentile_model' + str(nr+1) + '.pdf')
    # plt.show(block=False)
    # plt.close()

    fig, ax = plt.subplots()
    img = ax.scatter(x_1.flatten()[cdf_ft_order_index], x_2.flatten()[cdf_ft_order_index],
                     cdf_ft_l2diff.flatten()[cdf_ft_order_index], c=cdf_ft_l2diff.flatten()[cdf_ft_order_index],
                     vmin=0, vmax=np.max(cdf_ft_l2diff.flatten()[cdf_ft_order_index]), linewidth=7.5)
    plt.axis([0, 40, 0, 40])
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    plt.savefig('4ProposedVersionTestSet_cdf_ft_L2diff_model' + str(nr+1) + '.pdf')
    plt.show(block=False)
    plt.close()

    fig, ax = plt.subplots()
    img = ax.scatter(cdf_ft_x_1_select[cdf_ft_order_index_select], cdf_ft_x_2_select[cdf_ft_order_index_select],
                     cdf_ft_l2diff_select[cdf_ft_order_index_select], c=cdf_ft_l2diff_select[cdf_ft_order_index_select],
                     vmin=0, vmax=np.max(cdf_ft_l2diff_select[cdf_ft_order_index_select]), linewidth=7.5)
    plt.axis([0, 40, 0, 40])
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    plt.savefig('4ProposedVersionTestSet_cdf_ft_l2diff_percentile_model' + str(nr+1) + '.pdf')
    plt.show(block=False)
    plt.close()

    # fig, ax = plt.subplots()
    # img = ax.scatter(x_1.flatten()[cdf_at_order_index], x_2.flatten()[cdf_at_order_index],
    #                  cdf_at_l2diff.flatten()[cdf_at_order_index], c=cdf_at_l2diff.flatten()[cdf_at_order_index],
    #                  vmin=0, vmax=np.max(cdf_at_l2diff.flatten()[cdf_at_order_index]), linewidth=7.5)
    # plt.axis([-4, 4, -4, 4])
    # fig.colorbar(img, ax=ax)
    # fig.tight_layout()
    # plt.savefig('4ProposedVersionTestSet_cdf_at_L2diff_model' + str(nr+1) + '.pdf')
    # plt.show(block=False)
    # plt.close()

    # fig, ax = plt.subplots()
    # img = ax.scatter(cdf_at_x_1_select[cdf_at_order_index_select], cdf_at_x_2_select[cdf_at_order_index_select],
    #                  cdf_at_l2diff_select[cdf_at_order_index_select], c=cdf_at_l2diff_select[cdf_at_order_index_select],
    #                  vmin=0, vmax=np.max(cdf_at_l2diff_select[cdf_at_order_index_select]), linewidth=7.5)
    # plt.axis([-4, 4, -4, 4])
    # fig.colorbar(img, ax=ax)
    # fig.tight_layout()
    # plt.savefig('4ProposedVersionTestSet_cdf_at_l2diff_percentile_model' + str(nr+1) + '.pdf')
    # plt.show(block=False)
    # plt.close()

    fig, ax = plt.subplots()
    img = ax.scatter(x_1.flatten()[pdf_fa_order_index], x_2.flatten()[pdf_fa_order_index],
                     pdf_fa_l2diff.flatten()[pdf_fa_order_index], c=pdf_fa_l2diff.flatten()[pdf_fa_order_index],
                     vmin=0, vmax=np.max(pdf_fa_l2diff.flatten()[pdf_fa_order_index]), linewidth=7.5)
    plt.axis([0, 40, 0, 40])
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    plt.savefig('4ProposedVersionTestSet_pdf_fa_L2diff_model' + str(nr+1) + '.pdf')
    plt.show(block=False)
    plt.close()

    fig, ax = plt.subplots()
    img = ax.scatter(pdf_fa_x_1_select[pdf_fa_order_index_select], pdf_fa_x_2_select[pdf_fa_order_index_select],
                     pdf_fa_l2diff_select[pdf_fa_order_index_select], c=pdf_fa_l2diff_select[pdf_fa_order_index_select],
                     vmin=0, vmax=np.max(pdf_fa_l2diff_select[pdf_fa_order_index_select]), linewidth=7.5)
    plt.axis([0, 40, 0, 40])
    fig.colorbar(img, ax=ax)
    fig.tight_layout()
    plt.savefig('4ProposedVersionTestSet_pdf_fa_l2diff_percentile_model' + str(nr+1) + '.pdf')
    plt.show(block=False)
    plt.close()

    # with open("4ProposedVersionTestSet_cdf_fa_l2_diff_model" + str(nr + 1) + ".csv", "a") as f:
    #     writer = csv.writer(f)
    #     for item in cdf_fa_l2diff:
    #         writer.writerow(item)
    # f.close()

    with open("4ProposedVersionTestSet_cdf_ft_l2_diff_model" + str(nr + 1) + ".csv", "a") as f:
        writer = csv.writer(f)
        for item in cdf_ft_l2diff:
            writer.writerow(item)
    f.close()

    # with open("4ProposedVersionTestSet_cdf_at_l2_diff_model" + str(nr + 1) + ".csv", "a") as f:
    #     writer = csv.writer(f)
    #     for item in cdf_at_l2diff:
    #         writer.writerow(item)
    # f.close()

    with open("4ProposedVersionTestSet_pdf_fa_l2_diff_model" + str(nr + 1) + ".csv", "a") as f:
        writer = csv.writer(f)
        for item in pdf_fa_l2diff:
            writer.writerow(item)
    f.close()

with open("4ProposedVersionTestSet_optimal_models.csv", "a") as f:
    writer = csv.writer(f)
    writer.writerow(metrics_model.optimal_models)
f.close()

with open("4ProposedVersionTestSet_cdf_ft.csv", "a") as f:
    writer = csv.writer(f)
    for item in metrics_model.cdf_ft:
        writer.writerow(item)
f.close()

# with open("4ProposedVersionTestSet_cdf_at.csv", "a") as f:
#     writer = csv.writer(f)
#     writer.writerow(metrics_model.cdf_at)
# f.close()

# with open("4ProposedVersionTestSet_cdf_fa.csv", "a") as f:
#     writer = csv.writer(f)
#     for item in metrics_model.cdf_fa:
#         writer.writerow(item)
# f.close()

with open("4ProposedVersionTestSet_pdf_fa.csv", "a") as f:
    writer = csv.writer(f)
    for item in metrics_model.pdf_fa:
        writer.writerow(item)
f.close()

end_time = time.time()
print('\nScript was running for ', (end_time - start_time)/60, ' minutes')
