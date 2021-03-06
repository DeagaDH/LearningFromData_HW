# Implementation of quests 7-10 for Homework #1 from "Learning from Data" / Professor Yaser Abu-Mostafa, Caltech
# http://work.caltech.edu/homework/hw1.pdf

# This notebook defines the functions used in the OOP implementation of the solution.
# Actual solution is available in the hw1_oop.ipynb notebook

# In this problem, you will create your own target function f and data set D to see
# how the Perceptron Learning Algorithm works. Take d = 2 so you can visualize the
# problem, and assume X = \[−1, 1\] × \[−1, 1\] with uniform probability of picking each
# x ∈ X .

# In each run, choose a random line in the plane as your target function f (do this by
# taking two random, uniformly distributed points in [−1, 1] × [−1, 1] and taking the
# line passing through them), where one side of the line maps to +1 and the other maps
# to −1. Choose the inputs xn of the data set as random points (uniformly in X), and
# evaluate the target function on each xn to get the corresponding output yn.

# Now, in each run, use the Perceptron Learning Algorithm to find g. Start the PLA
# with the weight vector w being all zeros (consider sign(0) = 0, so all points are initially
# misclassified), and at each iteration have the algorithm choose a point randomly
# from the set of misclassified points. We are interested in two quantities: the number
# of iterations that PLA takes to converge to g, and the disagreement between f and
# g which is P\[f(x) != g(x)\] (the probability that f and g will disagree on their classification
# of a random point). You can either calculate this probability exactly, or
# approximate it by generating a sufficiently large, separate set of points to estimate it.

# In order to get a reliable estimate for these two quantities, you should repeat the
# experiment for 1000 runs (each run as specified above) and take the average over
# these runs.


# Date: 11/02/2021
# Author: Deaga


import random
import pandas as pd
import numpy as np 

def random_point(xlim=[-1,1],ylim=[-1,1]):
    """
    Creates a random point with coordinates(x,y)
    Random values will be bounded by xlim and ylim
    """

    x = random.uniform(xlim[0],xlim[1])
    y = random.uniform(ylim[0],ylim[1])

    return (x,y)

class line:

    def __init__(self,p1=None,p2=None,angular=None,linear=None,random=False,xlim=[-1,1],ylim=[-1,1]):
        """
        Initialize a line from either a pair of points (x,y)
        or from angular and linear coefficients.

        If random is set to true, generate a line from two random points,
        bounded by xlim and ylim.
        """

        if random:
            #Get a random line
            self.random_line(xlim,ylim)
        
        else:
            #Use the redefine function with the given inputs
            self.redefine(p1=p1,p2=p2,angular=angular,linear=linear)
    
    def redefine(self,p1=None,p2=None,angular=None,linear=None):
        """
        Redefines current line from either a pair of points (x,y)
        or from angular and linear coefficients.
        """

        #First case: given two points p1 and p2
        if (p1 != None and p2 != None):
            try:
                self.a= (p1[1] - p2[1]) / (p1[0] - p2[0]) #a = (y1-y2)/(x1-x2)
                self.b= (p1[0]*p2[1] - p2[0]*p1[1]) / (p1[0] - p2[0]) #b = (x1y2 - x2y1)/(x1 - x2)
            except:
                raise ValueError('Invalid format for p1 and/or p2. Use tuples with just two entries, p1=(x1,y1) and p2=(x2,y2).')
        #With given angular and linear
        elif (angular != None and linear != None):
            try:
                self.a=angular
                self.b=linear
            except:
                raise ValueError('Invalid format for angular or linear. Use numbers as input!')
        else:
            raise ValueError('Invalid inputs! p1 and p2 must be tuples in the form p1=(x1,y1), p2=(x2,y2).\nOtherwise, use angular=number and linear=number!')
    
    def random_line(self,xlim=[-1,1],ylim=[-1,1]):
        """
        Returns a line that passes two random points.
        Both points will be in the domain [xlim] x [ylim]
        """

        p1 = random_point(xlim,ylim)
        p2 = random_point(xlim,ylim)

        self.redefine(p1=p1,p2=p2)

    def get_y(self,x=0):
        """
        Calculates y value for a given x, for the current line.
        """

        return self.a*x+self.b

    def get_x(self,y=0):
        """
        Calculates x value for a given y, for the current line
        """

        return (y-self.b)/self.a

    def map(self,p):
        """
        Maps a value of +1 or -1 to point defined by p=(xp,yp)
        If yp > y(xp), return +1
        Else, return -1
        """
        if p[1] > self.get_y(p[0]):
            return 1
        else:
            return -1

def create_dataset(x_func,y_func,N_entries=100):
    """
    Creates a dataset of 'N_entries' entries.
    x values (inputs) are generated by x_func, while the
    y values (outputs) are generated by y_func
    """

    #Start with empty lists
    x_set = []
    y_set = []

    #Append N_entries to list:
    for i in range(0,N_entries):
        x_set.append(x_func())
        y_set.append(y_func(x_set[i]))

    #Return as a tuple
    return (x_set,y_set)

class perceptron():
    """
    Perceptron class to apply to Perceptron Learning Algorithm (PLA).
    """

    def __init__(self,data_x, data_y):
        """
        Initializes the perceptron. Expected input:
        data_x -> dictionary with x values (data points).
                  Each key is one type of feature, with values being the data points
                  For this exercise, data_x={'x1':[list of x1 coordinates],'x2':[list of x2 coordinates]}
        
        data_y -> dictionary with values of the target function for each point in data_x
                  For this exercise, data_y={'y':[list of +1/-1 mappings]}
        """
        #Store data_x (data points) and data_y (expected results)
        #in dataframes
        
        self.x=data_x
        self.y=data_y

        #Get number of data points
        self.N = len(self.x)

        #Get the number of features
        self.d = len(self.x[0])

        #Initialize weights as 0s
        self.weights = [0 for i in range(0,self.d+1)] #add 1 to account for artificial coordinate

        #Initialize Perceptron function as 0
        self.h = [0 for i in range(0,self.N)] #As many as there are points


    def h_func(self,x_eval):
        """
        Function to evaluate  h = sign(inner(weights,x_entry))
        """

        #Get length of x_eval
        N_eval = len(x_eval)

        #Initialize a results array of zeros
        h_temp = [0 for i in range(0,N_eval)]

        #Iterate through h
        for i in range(0,N_eval):
            
            # Create an array of the x values for this entry, adding
            # the artificial value x0=1    
            x_entry = [1] # Start with 1

            #Append columns of x_eval to x_entry, at the i-th row
            for j in range(0,self.d):
                x_entry.append(x_eval[i][j]) 

            #Calculate h[i]
            h_temp[i] = int(np.sign(np.inner(self.weights,x_entry)))

        return h_temp

    def update_weights(self,i):
        """
        Updates the weights to the i-th data point is classified
        correctly
        """


        # Update w0, which corresponds to the artificial point x0
        self.weights[0] += self.y[i] #x0=1

        #Update the other weights
        for j in range(0,self.d):
            self.weights[j+1] += self.y[i]*self.x[i][j]

    def learn(self,max_iter=100):
        """
        Implementation of the Perceptron Learning Algorithm
        """
        
        #Count iterations
        iter_count=0
    
        #Loop until converged
        while True:

            #Store old weight values to check convergence
            weights_old = self.weights.copy()
            
            #Increment iteration counter
            iter_count += 1

            #Evaluate h_func
            self.h = self.h_func(self.x)

            #Check for wrong points
            for i in range(0,self.N):
                if (self.h[i]!=self.y[i]):
                    #Update weights
                    self.update_weights(i)
                    break

            #Check convergence
            if (np.array_equal(self.weights,weights_old) or iter_count > max_iter):
                return int(iter_count) #Return iteration count
    
    def test_learning(self,x_test,y_test):
        """
        Tests the learning algorithm on the test set defined by x_test
        and y_test

        Returns the numerical value of the error probability in the test set
        """

        #Calculate h_test for the test set
        h_test = self.h_func(x_test)

        #Number of test points:
        N_test = len(x_test)

        #Count number of errors
        error_count = 0

        for i in range(0,N_test):
            if (h_test[i] != y_test[i]):
                error_count += 1

        #Return error probability        
        error_probability = error_count/N_test
        return error_probability