#kernel function

import numpy as np
import matplotlib.pyplot as plt


def kernel(xs,ys,sigma=1):
    # Squared exponential kernel designed to return the whole
    # covariance matrix

    dx=np.expand_dims(xs,1)-np.expand_dims(ys,0)
    return (sigma**2)*np.exp(-((dx/1)**2)/2)

def m(x):
    # Let the mean always be zero
    return np.zeros_like(x)

p=plt.figure()
D=100
xs=np.linspace(-5,5,D)
for i in range(0,11):
    ys=np.random.multivariate_normal(m(xs),kernel(xs,xs))
    #p=plt.scatter(xs,ys)
    p=plt.plot(xs,ys)
plt.show(p)














from bokeh.io import show
from bokeh.layouts import gridplot
from bokeh.models import ColorBar, ColumnDataSource, LinearColorMapper, LogColorMapper

from bokeh.plotting import figure





'''
from bokeh.io import show, output_file
from bokeh.models import ColumnDataSource
from bokeh.palettes import Viridis256 as palette
from bokeh.plotting import figure
from bokeh.sampledata.autompg import autompg as df
from bokeh.transform import linear_cmap
'''



'''
#Show kernel func


N=100
x=np.linspace(-2,2,N)
y=np.linspace(-2,2,N)
d=kernel(x,y)

color_mapper = LinearColorMapper(palette="Plasma256", low=0, high=1)

p = figure(plot_width=400, plot_height=400, x_range=(-2, 2), y_range=(-2, 2),
           title='Visualisation of k(x, x\')', x_axis_label='x',
           y_axis_label='x\'', toolbar_location=None)
p.image(image=[d], color_mapper=color_mapper, x=-2, y=-2, dw=4, dh=4)

color_bar = ColorBar(color_mapper=color_mapper, #ticker=BasicTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0))

p.add_layout(color_bar, 'right')

show(p)
'''