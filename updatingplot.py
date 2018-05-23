import threading
import time
import functools as ft

import numpy as np
import ipywidgets as ipw
import bqplot as bq
import pandas as pd

# class PollThread(threading.thread):
#     """Polling thread which can be terminated"""
#     def __init__(self, *, sleep, target, args, kwargs, **more_kwargs):
#         self._stop = False
#         
#         @ft.wraps(target)
#         def wrapper(*wr_args, **wr_kwargs):
#             while not self._stop:
#                 target(*wr_args, **wr_kwargs)
#                 time.sleep(sleep)
#             
#         super().__init__(
#             target=wrapper,
#             args=args,
#             kwargs=kwargs,
#             **more_kwargs
#         )
#         
#     def start(self, *args, **kwargs):
#         super().start(*args, **kwargs)
#     
#     def stop(self, *args, **kwargs):
#         self.stop = True

class UpdatingPlot(ipw.VBox):
    def __init__(self, ar, y, x=None, sleep=1, xlim=None, ylim=None, xlabel=None, ylabel=None):
        self.x = x
        self.ar = ar
        self.sleep = sleep
        self.xlim = xlim
        self.ylim = ylim
        self.xlabel = xlabel
        self.ylabel = ylabel
        if isinstance(y, list):
            self.y = y
        else:
            self.y = [y]
        if not self.xlim:
            self.xlim = [0, 1]
        if not self.ylim:
            self.ylim = [0, 1]
        if not xlabel:
            self.xlabel = 'x'
        if not ylabel:
            self.ylabel = 'y'
            
        
        self.init_figure()
        self.init_layout()
        self.init_logic()
        
    def create_line(self, y, display_legend=False):
        color = self.colors[self.num_lines%len(self.colors)]
        self.lines.append(bq.Lines(
            x=[],
            y=[],
            scales={'x': self.xscale, 'y': self.yscale},
            interpolation='linear',
            display_legend=display_legend,
            colors=[color],
            labels=[y]
        ))
        self.scatters.append(bq.Scatter(
            x=[],
            y=[],
            scales={'x': self.xscale, 'y': self.yscale},
            colors=[color]
        ))
        self.labels.append(y)
        self.num_lines += 1
        
    def init_figure(self):
        self.colors = ['blue', 'red', 'green', 'yellow', 'black', 'purple', 'gray']
        self.xscale = bq.LinearScale(min=self.xlim[0], max=self.xlim[1])
        self.yscale = bq.LinearScale(min=self.ylim[0], max=self.ylim[1])
        xlabel = self.xlabel
        if isinstance(self.ylabel, list):
            ylabel = ''
        else:
            ylabel = self.ylabel
            
        self.xax = bq.Axis(
            scale=self.xscale,
            label=xlabel,
            grid_lines='none',
        )
        self.yax = bq.Axis(
            scale=self.yscale,
            label=ylabel,
            orientation='vertical',
            grid_lines='none',
        )
        self.num_lines = 0
        self.lines = []
        self.scatters = []
        self.labels = []
        
        if isinstance(self.y, list):
            for y in self.y:
                self.create_line(y, display_legend=True)
        else:
            self.create_line(self.y)
        
        self.fig = bq.Figure(marks=self.lines+self.scatters, axes=[self.xax, self.yax])
        
    def resize_fig(self, i):
        xmin = float(np.min(self.lines[i].x))
        xmax = float(np.max(self.lines[i].x))
        ymin = float(np.min(self.lines[i].y))
        ymax = float(np.max(self.lines[i].y))
        if xmin < self.xscale.min:
            self.xscale.min = xmin
        if xmax > self.xscale.max:
            self.xscale.max = xmax
        if ymin < self.yscale.min:
            self.yscale.min = ymin
        if ymax > self.yscale.max:
            self.yscale.max = ymax
        
    def init_layout(self):
        super().__init__([
            self.fig
        ])
        
    def watch_data(self):
        while not self.done:
            try:
                self.update_all()
            except KeyError:
                pass
            time.sleep(self.sleep)
            
        if self.ar.done():        
            self.done = False
            
    def update_one(self, i):
        self.lines[i].y = self.ar.data[self.y[i]]
        self.scatters[i].y = self.ar.data[self.y[i]]
        if self.x:
            self.lines[i].x = self.ar.data[self.x]
            self.scatters[i].x = self.ar.data[self.x]
        else:
            x = [i for i in range(len(self.lines[0].y))]
            self.lines[i].x = x
            self.scatters[i].x = x
                
    def update_all(self):
        for i in range(self.num_lines):
            self.update_one(i)
            self.resize_fig(i)
                
    def init_logic(self):
        self.done = False
        self.thread = threading.Thread(target=self.watch_data)
        self.thread.start()