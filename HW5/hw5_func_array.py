import random
import numpy as np 
from math import e, log

def random_point(xlim=[-1,1],ylim=[-1,1]):
    """
    Creates a random point with coordinates(x,y)
    Random values will be bounded by xlim and ylim
    """

    x = random.uniform(xlim[0],xlim[1])
    y = random.uniform(ylim[0],ylim[1])

    return (x,y)

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
        Else, return 0
        """
        if p[1] > self.get_y(p[0]):
            return 1
        else:
            return -1

class LearningAlgorithm():
    """
    General class for LearningAlgorithms. Each LA will have its own subclass
    """

    def __init__(self,data_x,data_y):
        """
        Initializes the learning algorithm. Expected input:
        data_x -> list of x (input) values.
                  if each x has more than one value (ie points on a 2D plane have two values)
                  then each item of data_x should also be a list with all relevant values
        
        data_y -> list with values of the target function for each point in data_x
        """

        #Store data_x (data points) and data_y (expected results)        
        self.x=np.array(data_x)
        self.y=np.array(data_y)

        #Get number of data points
        self.N = len(self.x)

        #Get the number of features
        self.d = len(self.x[0])

        #Initialize weights as 0s
        self.weights = np.array([0 for i in range(0,self.d+1)]) #add 1 to account for artificial coordinate
    
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

class Perceptron(LearningAlgorithm):
    """
    Perceptron class to apply to Perceptron Learning Algorithm (PLA).
    """

    def __init__(self,data_x,data_y,weights=[]):
        #Initialize parent class
        LearningAlgorithm.__init__(self,data_x,data_y)

        #If a valid list is provided for self.weights, use it
        if (len(weights)==(self.d+1)): #Check valid lenght
            self.weights=weights 

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

    def learn(self,max_iter=100):
        """
        Implementation of the Perceptron Learning Algorithm
        """
        
        #Count iterations
        iter_count=0

        #Initialize Perceptron function as 0
        h = [0 for i in range(0,self.N)] #As many as there are points

        #Loop until converged
        while True:

            #Store old weight values to check convergence
            weights_old = self.weights.copy()
            
            #Increment iteration counter
            iter_count += 1

            #Evaluate h_func
            h = self.h_func(self.x)

            #List of point indexes with wrong values
            wrong = [] #Start empty, fill

            #Check for wrong points
            for i in range(0,self.N):
                if (h[i]!=self.y[i]):
                    #Update wrong list
                    wrong.append(i)

            #Update weights based on a random point from wrong
            try:
                self.update_weights(random.choice(wrong))
            except IndexError: 
                pass
                
            #Check convergence
            if (np.array_equal(self.weights,weights_old) or iter_count > max_iter):
                return int(iter_count) #Return iteration count

class LinearRegression(LearningAlgorithm):
    """
    Class to perform linear regression on a dataset
    """

    def __init__(self,data_x, data_y):
        #Initialize parent class
        LearningAlgorithm.__init__(self,data_x,data_y)

        #No specific initialization required for this class

    def learn(self):
        """
        Obtain the weights w of the linear regression with the currently loaded data_x and data_y
        """

        #Obtain matrix x_art, which includes the artificial entry x0=0 for all x
        x_art = [] #empty list

        for i in range(0,self.N):
            #Append the x0=0 coordinate in all lines
            x_art.append([1]) #Append a list of a single element at first
            x_art[i].extend(self.x[i]) #x[i] is a list; use .extend to add each item from x[i] individually

        #Obtain the pseudo inverse of self.x
        pseudo_inv = np.linalg.pinv(x_art)

        #Update weights accordingly
        self.weights = np.inner(pseudo_inv,self.y)

class LogisticRegression(LearningAlgorithm):
    """
    Class to perform linear regression on a dataset
    """

    def __init__(self,data_x, data_y):
        #Initialize parent class
        LearningAlgorithm.__init__(self,data_x,data_y)

    def gradient(self,x,y):
        """
        Returns the gradient of Ein for a data point defined by x (inputs) and y(expected output)
        """

        #Add artificial coordinate 1 to x
        x=np.insert(x,0,1)

        #Evalute numerators (x_art is a list; num is a list!)
        num = -y*x

        #Evaluate denominator (this is a scalar; inner product returns a scalar!)
        den = 1+e**(y*np.inner(self.weights,x))

        #Final result
        res = num/den
       
        #Return fraction
        return res

    def update_weights(self,eta):
        """
        Updates the weights vector
        """

        #Get a random permutation of points
        perm = np.random.permutation(self.N)

        #Go through all points
        for i in range(self.N):
            #Get the random x and y values
            x_rand = self.x[perm[i]]
            y_rand = self.y[perm[i]]

            #Evaluate the gradient
            gradient = self.gradient(x_rand,y_rand)

            #Update weights
            self.weights = self.weights - eta*gradient

    def learn(self,eta=0.01,max_iter=1000,tol=0.01):
        """
        Implementation of the Logistic Regression Learning Algorithm
        """
        
        #Count iterations
        iter_count=0

        #Loop until converged
        while True:

            #Store old weight values to check convergence
            weights_old = self.weights.copy()
            
            #Increment iteration counter
            iter_count += 1

            #Update weights based on Stochastic Gradient Descent
            self.update_weights(eta=eta)

            #Check convergence
            norm = np.linalg.norm(self.weights - weights_old)

            if (norm < tol or iter_count > max_iter):
                return int(iter_count) #Return iteration count

    def test_learning(self,x_test,y_test):
        """
        Tests the learning algorithm on the test set defined by x_test
        and y_test. Returns the average cross-entropy error found in the test set.
        """

        #Number of test points:
        N_test = len(x_test)

        #Initialize error at 0
        error = 0

        for i in range(N_test):
            #Add artificial coordinate 1 to x
            x=np.insert(np.array(x_test[i]),0,1)
            
            #Evaluate error
            error += log(1 + e**(-y_test[i]*np.inner(self.weights,x) ))

        #Return average error    
        error = error/N_test
        return error