import pandas as pd 
from sklearn.model_selection import train_test_split,cross_val_score,ShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss,make_scorer,accuracy_score
import numpy as np

def read_hw8_datasets(file_path,y1=1,y2=None):
    """
    Reads the features.test and features.train files.
    If both y1 and y2 are set, this will result in y1 vs y2 classification,
    discarding entries with other values (ie if y1=1 and y2=5, discard all
    points where digit is not 1 or 5)
    
    By default, this will assign value 1 to points where digit=y1 and
    value -1 to points where digit=y2. Other points are discarded
    
    If y2 is set to None, then do y1 vs all classification. This assigns
    value 1 to points where digit=y1 and value -1 to points where digit != y1
    
    Input data is expected in the form
    Digit | Intensity | Symmetry
    """

    df = pd.read_csv(file_path,names=['y','x1','x2'],delim_whitespace=True)
    df['y'] = pd.to_numeric(df['y'],downcast='integer')

    #Check for a given y2. If valid, do y1 vs y2 compare (eliminate other digits)
    if y2 and y2 >= 0 and y2 <= 9:
        df = df[(df['y']==y1) | (df['y']==y2)]

    #Now assign value +1 to digit=y1 and -1 to digit != y1 (if y2 was given, only y2 remains)
    df['y'] = df['y'].apply(lambda x: 1 if x==y1 else -1)

    #Return as a dataframe
    return (df.drop('y',axis=1),df['y'])

def final_transform(x_values):
    """
    Applies the final transform to a list of data points x.
    x_values = [data1,data2,data3...] where data_i = [x1,x2].
    Output is xtilde = [xt1,xt2,xt3...] where 
    xt_i = [x1,x2,x1**2,x2**2,x1x2,abs(x1-x2),abs(x1+x2)]
    The artificial coordinate 1 is built into the methods and must not be added here!
    """

    #Start with an empty list
    x_tilde = []

    for x in x_values:
        x_tilde.append([x[0],x[1],x[0]*x[1],x[0]**2,x[1]**2])

    return x_tilde

def questions7and8(y1_list,q=7,lamb=1):
    """
    Code to solve questions 7 and 8 of the final. Inputs are:
    y1_list - list of y1 values to use in classification of the datasets. For 1 vs all, y1 = 1, etc
    q - number of the question for results printing. If q=8, also apply the desired transform
    lamb - lambda value for weight decay; both questions use the default value of 1
    """

    # Initialize problem variables
    y_min = y1_list[0]
    error_min = 10000000000

    for y1 in y1_list:
        # Read from dataset
        x,y= read_hw8_datasets('features.test',y1=y1)

        # Convert to lists for compatibility
        x = x.values.tolist()
        y = y.values.tolist()

        # Transform x if question 8
        if q==8:
            x = final_transform(x)

        # Instantiate LinearRegression object
        linreg = LinearRegression(x,y)

        # Fit
        linreg.learn(lamb=lamb)

        # Evaluate error
        error = linreg.test_learning(x,y)

        # Store error and support vector count
        if error < error_min:
            y_min = y1
            error_min = error

    #Print results
    print(f'QUESTION {q}')
    print(f'Corresponding y1 = {y_min}')
    print(f'Ein = {round(error_min,3)}')
    print()

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

        #Get the values of x and y at point i
        y_loc = self.y[i]
        x_loc = np.concatenate([[1],self.x[i]])

        #Update weights
        self.weights = self.weights +  y_loc*x_loc

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

    def learn(self,lamb):
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