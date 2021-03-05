import random
import numpy as np 
from math import pi, sin

def random_point(xlim=[-1,1],ylim=[-1,1]):
    """
    Creates a random point with coordinates(x,y)
    Random values will be bounded by xlim and ylim
    """

    x = random.uniform(xlim[0],xlim[1])
    y = random.uniform(ylim[0],ylim[1])

    return (x,y)

def random_point_sine(xlim=[-1,1],m=pi):
    """
    Creates a random point with coordinates (x,sin(m*x))
    The random x value will be limited by xlim. Constant m may be modified.
    """

    x = random.uniform(xlim[0],xlim[1])
    y = sin(pi*x)

    return (x,y)

def random_point_sine_list(xlim=[-1,1],size=2,m=pi):
    """
    Creates a list of 'size' random points with coordinates (x,sin(m*x))
    The random x values will be limited by xlim. Constant m may be modified.
    Returns two lists, one of x values and one of y values
    """

    x=[]
    y=[]

    for i in range(size):
        x.append(random.uniform(xlim[0],xlim[1]))
        y.append(sin(pi*x[i]))

    return (x,y)

def midpoint(p1,p2):
    """
    Returns the midpoint between p1 and p2
    """

    p=() #Start with an empty tuple

    if (len(p1)!=len(p2)):
        raise ValueError('p1 and p2 have different dimensions!')

    for x1,x2 in zip(p1,p2):
        p += ((x1+x2)/2,)

    return p

def q7_func_a(x,b):
    return b

def q7_func_b(x,a):
    return a*x

def q7_func_c(x,a,b):
    return a*x+b

def q7_func_d(x,a):
    return a*x**2

def q7_func_e(x,a,b):
    return a*x**2+b

class line0:

    def __init__(self,p1=None,p2=None,angular=None,random=False,xlim=[-1,1],ylim=[-1,1]):
        """
        Initialize a line from either two points or its angular coefficient.
        This line will pass by the origin (0,0) minimize error from the two given points.
        If the angular coefficient is given, then it'll be use to define the line instead (y = ax)
        """

        if random: #Define randomly
            self.random_line0(xlim=xlim,ylim=ylim)

        else:
            #Use the redefine function with the given inputs
            self.redefine(p1=p1,p2=p2,angular=angular)
    
    def redefine(self,p1=None,p2=None,angular=None):
        """
        Redefines current line from either a single point (x,y)
        or a given value for the angular coefficient
        """

        #First case: given two points p1 and p2
        if (p1 != None and p2 != None):
            try:
                """
                a is evaluated to minimize the min. squared error between p1 = (x1,y1) and p2 = (x2,y2)
                error = (a*x1 - y1)² + (a*x2 - y2)²
                Derivating and making =0, solving for a
                2(a*x1 - y1)x1 + 2(a*x2 -y2)x2 = 0
                a = (x1y1 +x2y2)/(x1**2 + x2**2)
                """
                self.a= (p1[0]*p1[1]+p2[0]*p2[1])/(p1[0]**2 + p2[0]**2)
            except:
                raise ValueError('Invalid format for p. Use a tuple with just two entries, p=(x,y).')
        #With given angular and linear
        elif (angular != None):
            try:
                self.a=angular
            except:
                raise ValueError('Invalid format for angular. Use numbers as input!')
        else:
            raise ValueError('Invalid inputs! p be a tuple in the form p=(x,y).\nOtherwise, use angular=number!')
    
    def random_line0(self,xlim=[-1,1],ylim=[-1,1]):
        """
        Returns a line0 that does the best fit between two points.
        This is equivalent to making a line0 that goes through their midpoint
        """

        p1 = random_point(xlim,ylim)
        p2 = random_point(xlim,ylim)

        self.redefine(p1=p1,p2=p2)

    def get_y(self,x=0):
        """
        Calculates y value for a given x, for the current line.
        """

        return self.a*x

    def get_x(self,y=0):
        """
        Calculates x value for a given y, for the current line
        """

        return y/self.a

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
