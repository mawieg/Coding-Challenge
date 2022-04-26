# -*- coding: utf-8 -*-
"""
Data science challenge for a big data analytics firm

This program constructs an ARMA(1,1) time series with 100 periods.  A graphical
user interface then allows to choose frame size and alphabet size for a SAX 
representation, and to view the resulting frequency distribution of categories,
as well as the time series itself. 

author: Martin Wiegand
last changed: 26.04.2022
"""

# load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.stats import norm
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk


############### Functions to compute SAX frequency distribution ###############

def make_timeseries():
    """Define example time series as ARMA(1,1) process."""
    np.random.seed(12345)
    n = 100 # number of periods
    ar1 = np.array([1, -0.0]) # AR1 term
    ma1 = np.array([1, 0.5]) # MA1 term
    x = pd.Series(ArmaProcess(ar1, ma1).generate_sample(n))
    return x, n

def normalize(x):
    """Normalize Series x to have mean 0 and sd 1."""
    mean = x.mean()
    sd = x.std()
    x_new = (x-mean) / sd
    return x_new

def reduce_PAA(x, f):
    """Aggregate Series x, with frame size f"""
    agg_index = [np.floor(i/f) for i in x.index]
    return x.groupby(agg_index).mean()    
    
def transform_to_SAX(x, a):
    """Transform time series x into SAX form with a categories."""
    # define breakpoints
    breakpoints_0_to_1 = [i/a for i in range(1,a)]
    breakpoints = norm.ppf(breakpoints_0_to_1)
    # give binary numbers as category names 
    cat_names = [bin(i)[2:] for i in range(0,a)] 
    bins = [-np.inf] + list(breakpoints) + [np.inf]
    # translate values into categories
    x_transformed = pd.cut(x, bins, labels = cat_names)
    return pd.Series(x_transformed)

def produce_SAX_data(f, a):
    """
    Make time series x, normalize and aggregate, then apply SAX transformation.
    """
    x, n = make_timeseries()
    x_agg = reduce_PAA(x, f)
    x_agg_norm = normalize(x_agg)
    x_SAX = transform_to_SAX(x_agg_norm, a)
    return x, x_SAX, n


############## Auxiliary functions for drawing plots in the GUI ###############

def draw_figure(canvas, figure, loc=(0, 0)):
    """Draw 'figure' on canvas."""
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_fig_agg(fig_agg):
    """Drop figure."""
    fig_agg.get_tk_widget().forget()
    plt.close('all')
    
def make_figure(window, figure):
    """Drop existing figure, make new one and store under 'fig_agg'."""
    global fig_agg
    if fig_agg is not None:
        delete_fig_agg(fig_agg)
    canvas_elem = window['canvas'].TKCanvas
    canvas_elem.Size=(300,300)
    fig_agg = draw_figure(canvas_elem, figure)
    figure_canvas = FigureCanvasTkAgg(figure)
    figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    

######################## The Graphical User Interface #########################

# Define layout
control_frame = sg.Frame(layout=[
        [sg.Button('Show time series', key='show_time', 
                   border_width=5, pad=(10,10))],
        [sg.Text('Frame size', size=(9, 1)), 
         sg.Text('Alphabet size', size=(10, 1))],
        [sg.InputText('', key='edit_f', size=(10,1),pad=(10,10)), 
         sg.InputText('', key='edit_a', size=(10,1),pad=(10,10))], 
        [sg.Button('Show frequency distribution', key='show_hist', 
                   border_width=5, pad=(10,10))],
    ],
    title='Control Area', relief=sg.RELIEF_SUNKEN, vertical_alignment='top'
)

graph_area = sg.Column(layout=[[sg.Canvas(key='canvas', size=(700, 750))]], 
                       background_color='#DAE0E6', pad=(0, 0),)

controls = sg.Button('Exit')

layout = [[control_frame],
          [graph_area],
          [controls]
]

window = sg.Window('Frequency distribution', layout, size=(700,750))


# Run event loop
fig_agg = None
while True:
    event, values = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    # Button to display the untransformed time series 
    if event == 'show_time':
        x = make_timeseries()[0]
        ts_lineplot = plt.figure(dpi=125)
        p_ts_lineplot = sns.lineplot(x = x.index, y = x)
        make_figure(window, ts_lineplot)
    # Button to display the frequency distribution of SAX categories
    if event == 'show_hist':
        n = make_timeseries()[1]
        try:
            f = float(values['edit_f'])
        except:
            f = 4
            sg.popup_error("Please enter frame size")
            break
        if f < 1 or f > n or f != int(f):
            sg.popup_error("Frame size has to be an integer between 1 and n.")
            break
        try:
            a = float(values['edit_a'])
        except:
            a = 5
            sg.popup_error("Please enter frame size")
            break
        if a < 1 or a > n or a != int(a):
            sg.popup_error(
                "Alphabet size has to be an integer between 1 and n."
                )
            break
        x, x_SAX, n = produce_SAX_data(int(f), int(a))
        histo = plt.figure(dpi=125)
        p_histo = sns.histplot(data=x_SAX)
        make_figure(window, histo)
        
window.close()





