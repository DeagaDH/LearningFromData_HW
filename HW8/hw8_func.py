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
    
def questions2and3(filepath='features.test',q=2):
    """
    Solves questions 2 and 3 of HW8
    """

    #Initialize depending on desired question
    if q==2:
        y1_list = [0,2,4,6,8]
        y_min = y1_list[0]   
        error = 0
        n_sup = 0

    elif q==3:
        y1_list = [1,3,5,7,9]
        y_min = y1_list[0]  
        error = 10000000000
        n_sup = 0

    else:
        raise ValueError ('Question number must be 2 or 3!')

    # Define comparison function; q2 wants maximum value, but q3 wants minimum value

    def compare(val1, val2):
        """
        Compares values val1 and val2
        If q=2, returns true if val1 > val2
        If q=3, returns true if val1 < val2
        """

        if q==2:
            return val1 > val2

        else:
            return val1 < val2

    # Set C and Q values
    C = 0.01
    Q = 2

    for y1 in y1_list:
        # Read from dataset
        x,y= read_hw8_datasets('features.test',y1=y1)

        #Instantiate SVC object
        svm = SVC(C=C,kernel='poly',degree=Q,coef0=1,gamma=1)

        #Fit
        svm.fit(x,y)

        #Predict
        y_pred = svm.predict(x)

        # Evaluate error
        error_svm = zero_one_loss(y, y_pred)

        # Store error and support vector count
        if compare(error_svm,error):
            y_min = y1
            error = error_svm
            n_sup = svm.n_support_[0] + svm.n_support_[1]

    #Print results
    print(f'QUESTION {q}')
    print(f'Corresponding y1 = {y_min}')
    print(f'Ein = {round(error,3)}')
    print(f'N support vectors = {n_sup}')
    print()

    return (y_min,error,n_sup)

def questions5and6(train_path='features.train',test_path='features.test',Q=2):
    """
    Solves the problem given in questions 5 and 6, for a chosen Q value.
    The questions in the homework use Q=2 and Q=5.
    """
    
    # Initialize variables
    C_list = [0.0001,0.001,0.01,0.1,1]
    Ein =[]
    Eout =[]
    n_sup = []

    # Both questions employ a 1 vs 5 classifier
    y1=1
    y2=5

    # Read from dataset
    x,y= read_hw8_datasets(train_path,y1=y1,y2=y2)

    # Read second dataset for Eout
    x_out,y_out = read_hw8_datasets(test_path,y1=y1,y2=y2)

    for C in C_list:
        #Instantiate SVC object
        svm = SVC(C=C,kernel='poly',degree=Q,coef0=1,gamma=1)

        #Fit
        svm.fit(x,y)

        #Predict
        y_pred_in = svm.predict(x)
        y_pred_out = svm.predict(x_out)

        # Evaluate error
        Ein.append(zero_one_loss(y, y_pred_in))
        Eout.append(zero_one_loss(y_out, y_pred_out))

        # Count support vectors
        n_sup.append(svm.n_support_[0] + svm.n_support_[1])

    # Store data in a dataframe and return
    res = pd.DataFrame({'C':C_list,'Ein':Ein,'Eout':Eout,'N_sup':n_sup})

    return res

def questions7and8(train_path='features.train'):
    """
    Code to solve questions 7 and 8 implementing cross-validation
    """

    #Set up problem values
    Q=2
    nfolds=10
    y1=1
    y2=5
    N=100

    # List of C values to cross validate
    C_list = [0.0001, 0.001, 0.01, 0.1, 1]

    #Empty dictionary for results
    res = {}

    #Empty dictionary for error
    error = {}

    # initialize dictionaries
    for C in C_list:
        error[C] = []
        res[C] = 0

    # Read dataset
    x,y= read_hw8_datasets(train_path,y1=y1,y2=y2)

    #Make a scorer function from zero_one_loss
    accuracy_scorer = make_scorer(accuracy_score, greater_is_better=True)

    for i in range(N):
        if (i % 10 == 0):
            print(f'Iteration {i}...')
            
        #Initialize SVC objects
        svm_list = [SVC(C=C,kernel='poly',degree=Q,coef0=1,gamma=1) for C in C_list]

        #Shuffle the 10 folds
        cv = ShuffleSplit(n_splits=nfolds, test_size=0.2)

        #List of scores
        score_list = [cross_val_score(svm,x,y,cv=cv,scoring=accuracy_scorer) for svm in svm_list]

        for j in range(len(score_list)):
            score_list[j] = 1-np.average(score_list[j])

        #Best score
        best_score = min(score_list)

        #Corresponding C
        best_C = C_list[score_list.index(best_score)]

        # Add to count
        res[best_C] += 1

        # Add error
        error[best_C].append(best_score)

    return (res,error)

def questions9and10(train_path='features.train',test_path='features.test'):
    """
    Solves the problem given for quests 9 and 10 (evaluates both Ein and Eout for all
    given C)
    """

    # Initialize variables
    C_list = [10**x for x in range (-2,7,2)]
    Ein =[]
    Eout =[]

    # Both questions employ a 1 vs 5 classifier
    y1=1
    y2=5

    # Read from dataset
    x,y= read_hw8_datasets(train_path,y1=y1,y2=y2)

    # Read second dataset for Eout
    x_out,y_out = read_hw8_datasets(test_path,y1=y1,y2=y2)

    for C in C_list:
        #Instantiate SVC object
        svm = SVC(C=C,kernel='rbf',gamma=1)

        #Fit
        svm.fit(x,y)

        #Predict
        y_pred_in = svm.predict(x)
        y_pred_out = svm.predict(x_out)

        # Evaluate error
        Ein.append(zero_one_loss(y, y_pred_in))
        Eout.append(zero_one_loss(y_out, y_pred_out))

    # Store data in a dataframe and return
    res = pd.DataFrame({'C':C_list,'Ein':Ein,'Eout':Eout})

    return res

    