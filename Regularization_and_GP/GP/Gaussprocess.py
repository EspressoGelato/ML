#Gaussian Process
#Sample from the distribution instead of virsualising
#the whole distribution


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_unit_gaussian_samples(D):
    p= plt.figure()
    xs=np.linspace(0,1,D)
    for i in range(1,11):
        ys=np.random.multivariate_normal(np.zeros(D),np.eye(D))
        p=plt.plot(xs,ys)
    return p
plot_unit_gaussian_samples(10)
plt.show()








'''

from bokeh.io import show
from bokeh.plotting import figure
import numpy as np
import matplotlib.pyplot as plt

def plot_unit_gaussian_samples(D):
    p = figure(plot_width=800, plot_height=500,
               title='Samples from a unit {}D Gaussian'.format(D))

    xs = np.linspace(0, 1, D)
    for i in range(11):
        ys = np.random.multivariate_normal(np.zeros(D), np.eye(D))
        p.line(xs, ys, line_width=1)
    return p

show(plot_unit_gaussian_samples(10))

'''