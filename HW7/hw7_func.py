import random
import numpy as np 
from math import e, log
import pandas as pd 

def read_hw6_datasets(file_path):
    """
    Reads the in.dta or out.dta files from HW6 and returns a list of lists with the values.
    List of lists is used for compatibility with the implemented functions
    """

    df = pd.read_csv(file_path,names=['x1','x2','y'],delim_whitespace=True)
    df['y'] = pd.to_numeric(df['y'],downcast='integer')
    x_list = df[['x1','x2']].values.tolist()
    y_list = df['y'].values.tolist()

    #Return as a tuple for unpacking
    return (x_list,y_list)
    
def hw7_transform(x_values,k):
    """
    Applies the hw7 transform to a list of data points x.
    x_values = [data1,data2,data3...] where data_i = [x1,x2].
    Output is xtilde = [xt1,xt2,xt3...] where 
    xt_i = [x1,x2,x1**2,x2**2,x1x2,abs(x1-x2),abs(x1+x2)]
    The artificial coordinate 1 is built into the methods and must not be added here!
    This method will only apply the 'k' first components of the xt transform, counting from 1
    (ie k=3 applies the transforms x1,x2,x1**2 ; transform 0 is the artificial 1 that is not included)
    """

    #Start with an empty list
    x_tilde = []

    #Transform dictionary:
    for x in x_values:

        #Temp variable to append the terms to
        temp=[]
        if k >= 1:
            temp.append(x[0])

        if k >= 2:
            temp.append(x[1])

        if k >= 3:
            temp.append(x[0]**2)

        if k >= 4:
            temp.append(x[1]**2)

        if k >= 5:
            temp.append(x[0]*x[1])

        if k >= 6:
            temp.append(abs(x[0]-x[1]))

        if k >= 7:
            temp.append(abs(x[0]+x[1]))

        x_tilde.append(temp)
    return x_tilde

def questions1to4(x_train,y_train,x_test,y_test,k_list,q_number=1):
    """
    Runs the required Linear Regression code for questions 1 though 4
    k_list is a  list of k_values to decide the transformation to use for the questions
    q_number is the question number for printing
    function will print the minimum value of E_out for the given k's in k_list with the
    corresponding k.
    """

    #Initialize e_out as an empty list
    e_out = []

    for k in k_list:
        #Apply the appropriate transform to x_train and x_test
        x_train_tilde = hw7_transform(x_train,k=k)
        x_test_tilde  = hw7_transform(x_test,k=k)

        #Run linear regression
        #Initialize a LinearRegression object with the x and y lists
        linreg = LinearRegression(x_train_tilde,y_train)

        #Calculate the linear regression
        linreg.learn(lamb=0)

        #Evaluate e_out based on the test set
        e_out.append(linreg.test_learning(x_test_tilde,y_test))

    #Find minimum error and corresponding k
    e_out_min = min(e_out)
    k_min = k_list[e_out.index(e_out_min)]

    # Print results
    print(f'QUESTION {q_number}')
    print(f'Minimum Eout  = {round(e_out_min,2)}')
    print(f'Corresponding k = {k_min}')
    print()
    

def split_dataset(x,y,i=25):
    """
    Splits a dataset of inputs (x) and outputs (y) into a train portion
    and a test portion. The first 'i' values will be set aside as the train
    portion of the dataset; the remaining points will be the test set.
    """

    x_train = x[:i]
    y_train = y[:i]
    x_test = x[i:]
    y_test = y[i:]

    return (x_train,y_train,x_test,y_test)

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

    def learn(self,lamb=0):
        """
        Obtain the weights w of the linear regression with the currently loaded data_x and data_y

        lamb is the weight decay lambda. By defult, lamb=0, that is, no decay is applied
        """

        #Obtain matrix x_art, which includes the artificial entry x0=0 for all x
        x_art = [] #empty list

        for i in range(0,self.N):
            #Append the x0=0 coordinate in all lines
            x_art.append([1]) #Append a list of a single element at first
            x_art[i].extend(self.x[i]) #x[i] is a list; use .extend to add each item from x[i] individually

        # Convert to array, get transpose and number of columns
        x_art = np.array(x_art) 
        x_art_t = np.transpose(x_art)
        ncol=x_art.shape[1]

        #Obtain the pseudo inverse of self.x
        pseudo_inv = np.linalg.pinv(x_art_t @ x_art + lamb * np.identity(ncol)) @ x_art_t # @ = matrix multiplication

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