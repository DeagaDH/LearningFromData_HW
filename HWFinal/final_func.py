import pandas as pd 
from sklearn.svm import SVC
from sklearn.metrics import zero_one_loss
from sklearn.cluster import KMeans
import numpy as np
from math import sin, pi
import random

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
        x,y= read_hw8_datasets('features.train',y1=y1)

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

def final_target(x1,x2):
    """
    Target function for the RBF questions of the final.
    f(x1,x2) = sign(x2 - x1 + 0.25sin(pi*x1))
    """

    # Evalue innermost term
    res = x2 - x1 + 0.25*sin(pi*x1)

    # Return sign
    return np.sign(res)

def final_dataset(N=100,target_function=final_target,x1_lim=[-1,1], x2_lim=[-1,1]):
    """
    Generates a dataset of N data points for questions 13
    to 18. x1 and x2 values are bounded by x1_lim and x2_lim.
    """

    # Start with empty lists
    x=[]
    y=[]

    # Create N points
    for i in range(N):
        #x1 and x2 must respect given limits
        x1 = random.uniform(x1_lim[0],x1_lim[1])
        x2 = random.uniform(x2_lim[0],x2_lim[1])

        #Append to x and y
        x.append([x1,x2])
        y.append(target_function(x1,x2))

    return (x,y)

def final_phi(x_list,mu_list,gamma=1.5):
    """
    Applies the RBF kernel to x and mu. Returns a list of lists that represents a matrix
    phi = [[exp(-gamma*norm(x1-mu1)**2), exp(-gamma*norm(x1-mu2)**2), ... exp(-gamma*norm(x1-muk)**2)],
           [exp(-gamma*norm(x2-mu1)**2), exp(-gamma*norm(x2-mu2)**2), ... exp(-gamma*norm(x2-muk)**2)]
           .                                        .                                        .
           .                                        .                                        .
           .                                        .                                        .
           [exp(-gamma*norm(xN-mu1)**2), exp(-gamma*norm(xN-mu2)**2), ... exp(-gamma*norm(xN-muk)**2)]]
    """

    res =[]

    for x in x_list:
        temp=[]

        for mu in mu_list:
            #Convert to numpy arrays
            x_array = np.array(x)
            mu_array = np.array(mu)

            # Evalute conversion
            temp2=np.exp(-gamma*np.linalg.norm(x_array-mu_array)**2)
            
            # Back to list
            temp2 = temp2.tolist()

            #append to temp
            temp.append(temp2)

        #Append to final results
        res.append(temp)

    return res

def questionsRBF(N_train=100,N_test=100,N_exp=500):
    """
    Solves various questions regarding RBF. Questions 13, 14, 15, 16 and 18.
    """

    # Constants of the SVM and RBF models
    C = 10000000000000000 #Hard margin
    kernel='rbf'
    gamma=1.5
    K = [9, 12]
    lamb = 0

    #Start counts at 0
    count = 0
    count_k = [0, 0]
    count_k_sep = [0, 0]
    N_actual = 0

    # Count for each alternative of Q16
    count_alt = {'a':0,'b':0,'c':0,'d':0,'e':0}

    #Repeat N_exp times
    for i in range(N_exp):

        if (i % (N_exp/10) ==0):
            print(f'Running experiment number {i}...')
        #######################################################
        ###             TEST & TRAINING DATASETS
        #######################################################

        # Generate training dataset (compatible with the funciton made for hw8)
        x,y= final_dataset(N=N_train)

        # Generate a separate dataset for Eout
        x_out,y_out = final_dataset(N=N_test)
        
        # Increment N_actual
        N_actual += 1

        #######################################################
        ###              SVC WITH HARD MARGIN
        #######################################################
        #Instantiate SVC object
        svm = SVC(C=C,kernel=kernel,gamma=gamma)

        #Fit
        svm.fit(x,y)

        #Evaluate y_pred for the dataset points to calculate Ein
        y_pred_in = svm.predict(x)

        #Evaluate ein_svm
        ein_svm = zero_one_loss(y, y_pred_in)

        #If Ein > 0, add to count
        if ein_svm > 0:
            count +=1

        # Evaluate eout_svm
        y_pred_out = svm.predict(x_out)
        eout_svm = zero_one_loss(y_out, y_pred_out)


        #######################################################
        ###                REGULAR RBF
        #######################################################

        #Initialize ein_k and eout_k as empty lists
        ein_k = []
        eout_k = []
        # Repeat for all elements in K
        for i in range(len(K)):
            #Instantiate SVC object
            km = KMeans(n_clusters=K[i])

            #Fit
            km.fit(x,y)

            # Get centers
            mu = km.cluster_centers_.tolist()

            # Calculate matrix phi for trainig and test datasets
            phi = final_phi(x_list=x,mu_list=mu,gamma=gamma)
            phi_out = final_phi(x_list=x_out,mu_list=mu,gamma=gamma)

            # Instantiate LinearRegression objects
            linreg = LinearRegression(phi,y)

            # Fit
            linreg.learn(lamb=lamb)

            # Evaluate error in sample
            ein_k.append(linreg.test_learning(phi,y))

            # Evaluate error out of sample
            eout_k.append(linreg.test_learning(phi_out,y_out))

            # If eout_svm < eout_k, add to count
            if eout_svm < eout_k[i]:
                count_k[i] +=1

            # Check for ein_k == 0
            if ein_k[i] ==0:
                count_k_sep[i] += 1
        
        #######################################################
        ###                QUESTION 16
        #######################################################

        # Check each alternative of question 16
        
        # Alternative a
        if ((ein_k[1] < ein_k[0]) and (eout_k[1] > eout_k[0])):
            count_alt['a'] += 1
        
        # Alternative b
        if ((ein_k[1] > ein_k[0]) and (eout_k[1] < eout_k[0])):
            count_alt['b'] += 1

        # Alternative c
        if ((ein_k[1] > ein_k[0]) and (eout_k[1] > eout_k[0])):
            count_alt['c'] += 1

        # Alternative d
        if ((ein_k[1] < ein_k[0]) and (eout_k[1] < eout_k[0])):
            count_alt['d'] += 1

        # Alternative e
        if ((ein_k[1] == ein_k[0]) and (eout_k[1] == eout_k[0])):
            count_alt['e'] += 1

    # Print total count and percentage of Ein > 0
    print(f'Total runs: {N_actual}\n')
    print(f'QUESTION 13:')
    print(f'Total not separable: {count}')
    print(f'Percentage not separable: {round(count/N_actual*100,0)}%')
    print()
    print(f'QUESTION 14:')
    print(f'Total where kernel form has lower Eout than K1=9 clusters: {count_k[0]}')
    print(f'Percentage: {round(count_k[0]/N_actual*100,0)}%')
    print()
    print(f'QUESTION 15:')
    print(f'Total where kernel form has lower Eout than K2=12 clusters: {count_k[1]}')
    print(f'Percentage: {round(count_k[1]/N_actual*100,0)}%')
    print()
    print(f'QUESTION 16:')
    print(f'Most frequent alternative: {max(count_alt, key=count_alt.get)}')
    print()
    print(f'QUESTION 18:')
    print(f'Total separable with K1=9 clusters: {count_k_sep[0]}')
    print(f'Percentage: {round(count_k_sep[0]/N_actual*100,0)}%')

def question17(N_train=100,N_test=100,N_exp=500):
    """
    Solves question 17 of the final
    """

    # Constants of the SVM and RBF models
    gamma=[1.5, 2]
    K = 9
    lamb = 0

    # Count for each alternative of Q17
    count_alt = {'a':0,'b':0,'c':0,'d':0,'e':0}

    # Count iterations
    N_actual = 0

    #Repeat N_exp times
    for i in range(N_exp):

        if (i % (N_exp/10) ==0):
            print(f'Running experiment number {i}...')
        #######################################################
        ###             TEST & TRAINING DATASETS
        #######################################################

        # Generate training dataset (compatible with the funciton made for hw8)
        x,y= final_dataset(N=N_train)

        # Generate a separate dataset for Eout
        x_out,y_out = final_dataset(N=N_test)
        
        # Increment N_actual
        N_actual += 1

        #######################################################
        ###                REGULAR RBF
        #######################################################

        #Initialize ein and eout as empty lists
        ein = []
        eout = []
        
        # Repeat for all elements in K
        for i in range(len(gamma)):
            #Instantiate SVC object
            km = KMeans(n_clusters=K)

            #Fit
            km.fit(x,y)

            # Get centers
            mu = km.cluster_centers_.tolist()

            # Calculate matrix phi for trainig and test datasets
            phi = final_phi(x_list=x,mu_list=mu,gamma=gamma[i])
            phi_out = final_phi(x_list=x_out,mu_list=mu,gamma=gamma[i])

            # Instantiate LinearRegression objects
            linreg = LinearRegression(phi,y)

            # Fit
            linreg.learn(lamb=lamb)

            # Evaluate error in sample
            ein.append(linreg.test_learning(phi,y))

            # Evaluate error out of sample
            eout.append(linreg.test_learning(phi_out,y_out))
        
        #######################################################
        ###                CHECK ALTERNATIVES
        #######################################################

        # Check each alternative of question 17
        
        # Alternative a
        if ((ein[1] < ein[0]) and (eout[1] > eout[0])):
            count_alt['a'] += 1
        
        # Alternative b
        if ((ein[1] > ein[0]) and (eout[1] < eout[0])):
            count_alt['b'] += 1

        # Alternative c
        if ((ein[1] > ein[0]) and (eout[1] > eout[0])):
            count_alt['c'] += 1

        # Alternative d
        if ((ein[1] < ein[0]) and (eout[1] < eout[0])):
            count_alt['d'] += 1

        # Alternative e
        if ((ein[1] == ein[0]) and (eout[1] == eout[0])):
            count_alt['e'] += 1

    print(f'QUESTION 17:')
    print(f'Total count for each alternative: {count_alt}')
    print(f'Most frequent alternative: {max(count_alt, key=count_alt.get)}')

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