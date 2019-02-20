# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 22:10:43 2019

@author: Jun
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.linalg import norm
"""
Quadratic loss function as linear regression
"""
def f(x, y, theta):
    return sum((y - np.dot(theta.T, x))**2)

"""
Derivative of the loss function respect to theta
"""
def gradf(x, y, theta):
    #sum start from 1
    return -2*sum(np.dot((y-np.dot(theta.T, x)), x.T), 1)

sets = {}

def get_features(sets, data):
    #skip the first line
    data_file = open(data)
    data_file.readline()
    for line in data_file:
        home_team = line.split(',')[2]
        away_team = line.split(',')[3]
        if (home_team not in sets.keys()):
            sets[home_team] = []
        if (away_team not in sets.keys()):
            sets[away_team] = []
        result = 0
        if (line.split(',')[6] == 'H'):
            result = 1
        elif (line.split(',')[6] == 'A'):
            result = -1
    
        sets[home_team].append([result, 0.5] + [float(i) for i in line.split(',')[4:5]] + [float(j) for j in line.split(',')[11:22]])
        sets[away_team].append([-result, -0.5] + [float(i) for i in line.split(',')[4:5]] + [float(j) for j in line.split(',')[11:22]])

def separation(sets, training, validation, test):
    '''Return the assigned training, validation, test data dictionary
    Arguments:
    act -- datasets of each team in Premier League
    training -- range of training set
    validation -- range of validation set
    test -- range of test set
    '''
    training_set = {}
    validation_set = {}
    test_set = {}
    for team in sets.keys():
        training_set[team] = []
        validation_set[team] = []
        test_set[team] = []
        i = 0
        #assign sets by dates
        while (len(sets[team]) > i and i < training + validation + test):
            if len(training_set[team]) < training:
                training_set[team].append(sets[team][i])
            elif len(validation_set[team]) < validation:
                validation_set[team].append(sets[team][i])
            elif len(test_set[team]) < test:
                test_set[team].append(sets[team][i])
            i += 1
            
    return training_set, validation_set, test_set

def create_matrix(data_set):
    '''
    return the dictionary containing feature and label(result) of each team
    '''
    team_data = {}
    for team in data_set.keys():
        team_data[team] = {}
        team_data[team]["feature"] = np.array([])
        team_data[team]["feature"] = team_data[team]["feature"].reshape(14, 0)
        team_data[team]["label"] = np.array([])
        
        for games in data_set[team]:            
            #result
            team_data[team]["label"] = np.hstack((team_data[team]["label"], np.array(games[0])))
            features = (np.array(games[1:])).reshape(13,1)
            features = np.vstack((np.array([1]), features))
            team_data[team]["feature"] = np.hstack((team_data[team]["feature"], features))
            
    return team_data


def output(x, theta):
    result = []
    for h in np.dot(theta.T, x):
        if h < -0.5:
            result.append(-1)
        elif h >= -0.5 and h <= 0.5:
            result.append(0)
        else:
            result.append(1)
    return result

def performance(feature, theta, label):
    num_correct = 0.0
    classification = output(feature, theta)
    for i in range(label.size):
        if classification[i] == label[i]:
            num_correct += 1
    
    return num_correct/label.size

def record_grad_descent(f, gradf, x, y, init_t, alpha, max_iter, recording_step, EPS = 1e-5):
    iter_num = []
    f_values = []
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter = 0
    init_f = f(x, y, init_t)
    while norm(t-prev_t) > EPS and iter < max_iter and f(x, y, t) <= init_f*2:
        prev_t = t.copy()
        t -= alpha*gradf(x, y, t)
        if iter % recording_step == 0:
            f_values.append(f(x, y, t))
            iter_num.append(iter)
        iter += 1
        
    return t, f_values, iter_num

def grad_descent(f, gradf, x, y, init_t, alpha, max_iter, EPS = 1e-5):
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter = 0
    while norm(t-prev_t) > EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*gradf(x, y, t)
        iter += 1
        
    return t


def predict(sets):
    try:
        if not os.path.exists("figures/"):
            os.makedirs("figures/")
    except OSError:
        print("Error: Creating directory. " + "figures/")
      
    #classifier = {'W': 1, 'D': 0, 'L': -1}
    training_set, validation_set, test_set = separation(sets, 20, 10, 5)
    team_data = create_matrix(training_set)
    '''
    print("Learning rate #1:")
    plt.figure(1)
    theta0 = np.random.normal(0, 0., 15)
    legend = []
    max_iter = 100
    recording_step = 20
    for x in range(16, 26, 2):
        alpha = 0.000001*x
        theta, f_value, iter_num = record_grad_descent(f, gradf, data, data_label, theta0, alpha, max_iter, recording_step)
        plt.plot(iter_num, f_value, '^-')
        legend.append("alpha = "+str(alpha))
    plt.legend(legend)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function value on Training Set")
    plt.savefig("figures/p3f1.jpg")
    plt.show()
  
    print("Learning rate #2:")
    plt.figure(2)
    theta0 = np.random.normal(0, 0., 1025)
    legend = []
    max_iter = 1000
    recording_step = 200
    for x in range(12, 24, 2):
        alpha = 0.000001*x
        theta, f_value, iter_num = record_grad_descent(f, gradf, data, data_label, theta0, alpha, max_iter, recording_step)
        plt.plot(iter_num, f_value, '^-')
        legend.append("alpha = "+str(alpha))
    plt.legend(legend)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cost function value on Training Set")
    plt.savefig("figures/p3f2.jpg")
    plt.show()
  '''
    print("Theta Initialization")
    np.random.seed(0)
    for deviation in [0., 0.0001, 0.001, 0.01, 0.1, 1]:
        f_value = []
        #Try 5 times
        for team in test_set:
            theta0 = np.random.normal(0,deviation,14)
            f_value.append(f(team_data[team]["feature"], team_data[team]["label"],theta0))
        print "Initial value of cost function on the training set with standard deviation of theta to be %.4f : Mean: f(x) = %.2f  Standard Error: %.2f" % (deviation, np.mean(f_value), np.std(f_value))
      
    
    print "\nReport the values of the cost function on the training and the validation sets"
    print "Report performance on the training and the validation sets\n"
    alpha = 0.000001
    max_iter = 100000
    cost_t = []
    cost_v = []
    perf_t = []
    perf_v = []
    val_data = create_matrix(validation_set)
    for team in test_set:
        theta0 = np.random.normal(0, 0., 14)
        data = team_data[team]["feature"]
        dlabel = team_data[team]["label"]
        validation = val_data[team]["feature"]
        vlabel = val_data[team]["label"]
        theta = grad_descent(f, gradf, data, dlabel, theta0, alpha, max_iter)
        cost_t.append(f(data, dlabel, theta)/(2*dlabel.size))
        cost_v.append(f(validation, vlabel, theta)/(2*vlabel.size))
        perf_t.append(performance(data, theta, dlabel)*100)
        perf_v.append(performance(validation, theta, vlabel)*100)
    print "Value of cost function on the training set: f(x) = %f +/- %f" % (np.mean(cost_t),np.std(cost_t))
    print "Value of cost function on the validation set: f(x) = %f +/- %f" % (np.mean(cost_v),np.std(cost_v))
    print "Performance of the classifier on the training set:  %.2f%% +/- %f%%" % (np.mean(perf_t),np.std(perf_t))
    print "Performance of the classifier on the validation set:  %.2f%% +/- %f%% \n" % (np.mean(perf_v),np.std(perf_v))


if __name__ == "__main__":
    sets = {}
    get_features(sets, "E0.csv")
    get_features(sets, "E1.csv")
    get_features(sets, "E2.csv")
    predict(sets)