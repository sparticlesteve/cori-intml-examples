# stdlib
import concurrent.futures as cf
import copy
import itertools
import pickle
import queue
import threading
import time
import warnings

# 3rd party
import bqplot as bq
import ipyparallel as ipp
from IPython.display import display, clear_output, update_display
import ipywidgets as ipw
import numpy as np
import pandas as pd
import qgrid


class ModelPlot(ipw.VBox):
    def __init__(self, y, x=None, xlim=None, ylim=None, xlabel=None, ylabel=None, title=None):
        super().__init__()

        self.x = x
        self.xlim = xlim or [0, 1]
        self.ylim = ylim or [0, 1]
        self.xlabel = xlabel or 'x'
        self.ylabel = ylabel or 'y'
        self.title = title or "{} vs {}".format(ylabel, xlabel)

        if isinstance(y, list):
            self.y = y
        else:
            self.y = [y]

        self.colors = ['blue', 'red', 'green', 'orange', 'black', 'purple', 'gray']
        self.xscale = bq.LinearScale(min=self.xlim[0], max=self.xlim[1])
        self.yscale = bq.LinearScale(min=self.ylim[0], max=self.ylim[1])

        if isinstance(self.ylabel, list):
            ylabel = ''
        else:
            ylabel = self.ylabel
            
        self.xax = bq.Axis(
            scale=self.xscale,
            label=self.xlabel,
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

        self.fig = bq.Figure(
            marks=self.lines + self.scatters, 
            axes=[self.xax, self.yax], 
            layout=ipw.Layout(height='550px', width='100%'))
        self.debug = ipw.Output(layout=ipw.Layout(height='100px', overflow_y='scroll'))
        self.children = [self.fig]

    def create_line(self, y, display_legend=False):
        try:
            color = self.colors[self.num_lines % len(self.colors)]
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
        except Exception as e:
            self.debug.append_stdout("Exception when adding a line and points to plot: {}".format(e.args))

    def resize_fig(self, i):
        try:
            self.xscale.min = min(self.xscale.min, np.min(self.lines[i].x))
            self.xscale.max = max(self.xscale.max, np.max(self.lines[i].x))
            self.yscale.min = min(self.yscale.min, np.min(self.lines[i].y))
            self.yscale.max = max(self.yscale.max, np.max(self.lines[i].y))
        except Exception as e:
            self.debug.append_stdout("Exception when resizing the figure: {}\n".format(e.args))

    def update_one(self, i, data):
        try:
            # check to see if we should fill in past data or just get the latest points
            if "history" in data and len(data["history"][self.y[i]]) > len(self.lines[i].y) + 1:
                self.lines[i].y = data["history"][self.y[i]]
                self.scatters[i].y = data["history"][self.y[i]]

                if self.x and self.x in data["history"]:
                    self.lines[i].x = data["history"][self.x]
                    self.scatters[i].x = data["history"][self.x]
                else:
                    self.lines[i].x = np.array([i for i in range(len(self.lines[0].y))])
                    self.scatters[i].x = np.array([i for i in range(len(self.lines[0].y))])
            # we are caught up, just add the latest data points
            elif "logs" in data:
                self.lines[i].y = np.append(self.lines[i].y, data["logs"][self.y[i]])
                self.scatters[i].y = np.append(self.scatters[i].y, data["logs"][self.y[i]])

                if self.x and self.x in data["logs"]:
                    self.lines[i].x = np.append(self.lines[i].x, data["logs"][self.x])
                    self.scatters[i].x = np.append(self.scatters[i].x, data["logs"][self.x])
                else:
                    self.lines[i].x = np.array([i for i in range(len(self.lines[0].y))])
                    self.scatters[i].x = np.array([i for i in range(len(self.lines[0].y))])
            else:
                self.debug.append_stdout("i {}, data {}\n".format(i, data))
        except Exception as e:
            self.debug.append_stdout("Exception plotting a line: {}\n".format(e.args))
            self.debug.append_stdout("Plot: {}, Data: {}".format(i, data))

    def update(self, data):
        try:
            for i in range(self.num_lines):
                self.update_one(i, data)
                self.resize_fig(i)
        except Exception as e:
            self.debug.append_stdout("Exception while plotting lines and resizing figure: {}\n".format(e.args))
            self.debug.append_stdout("Data: {}\n".format(data))


class ParamSpanWidget(ipw.VBox):
    def __init__(self, compute_func, vis_func, params, columns=None, ipp_cluster_id=None, 
                 product=False, output_layout=None, qgrid_layout=None):
        """
        compute_func: function 
        task to submit to IPyParallel for model output
        
        vis_func: function 
        function that produces a visualization of the model output (e.g. ModelPlot)
        
        params: dict
        grid search parameters
        
        ipp_cluster_id: str
        optional ipyparallel cluster id for connecting to a specific controller
        
        product: bool
        Whether to take the cartesian product of parameters (grid search). Otherwise, they must be the same length.
        """
        super().__init__()

        self.compute_func = compute_func
        self.vis_func = vis_func
        self.product = product
        self.output_layout = output_layout or \
            ipw.Layout(height='600px', border='1px solid', overflow_x='scroll', overflow_y='scroll')
        self.debug_layout = ipw.Layout(height='500px', border='1px solid', overflow_x='scroll', overflow_y='scroll')
        self.qgrid_layout = qgrid_layout or ipw.Layout()

        self.executor = cf.ThreadPoolExecutor()

        # connect to ipyparallel
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if ipp_cluster_id:
                self.ipp_client = ipp.Client(cluster_id="{}".format(ipp_cluster_id))
            else:
                self.ipp_client = ipp.Client()
        self.dview = self.ipp_client.direct_view()
        self.lview = self.ipp_client.load_balanced_view()

        list_params = {}
        for k in params:
            if type(params[k]) is np.ndarray:
                list_params[k] = params[k].tolist()
            else:
                list_params[k] = list(params[k])

        self.compute_params = list_params
        self.columns = ["status", "epoch"] + [k for k in params] + ["loss", "val_loss", "acc", "val_acc"]

        display_params = copy.deepcopy(list_params)
        for k in display_params:
            needs_str = False
            for i in range(len(display_params[k])):
                if isinstance(list, type(display_params[k][i])):
                    needs_str = True
            if needs_str:
                display_params[k] = [str(i) for i in display_params[k]]
        
        # setup the dataframe used to populate the table
        #self.compute_param_keys = params.keys()
        self.params_df = pd.DataFrame(display_params, columns=self.columns)
        self.params_df["status"] = ["Not Started"] * self.params_df.shape[0]
        self.params_df["epoch"] = [-1] * self.params_df.shape[0]

        # create the plot output and debug output widgets
        self.output = ipw.Output(layout=self.output_layout)
        self.debug = ipw.Output(layout=self.debug_layout)
        
        # create the table widget
        self.param_table = qgrid.QGridWidget(df=self.params_df, layout=self.qgrid_layout)
        self.param_table.grid_options['defaultColumnWidth'] = 200
        self.param_table.grid_options['forceFitColumns'] = True
        self.param_table.grid_options['editable'] = False

        # set the first row to be selected by default
        #self.param_table._selected_rows = [0]
        #self.param_table._handle_qgrid_msg_helper({'type': 'selection_changed', 'rows': [0]})
        #self.param_table._update_table()

        # add event listeners to the table
        self.add_handlers()

        # add buttons for stopping and restarting runs
        self._stop_btn = ipw.Button(description="Stop selected")
        self._stop_btn.on_click(self.stop_selected_models)
        self._restart_btn = ipw.Button(description="Restart selected")
        self._restart_btn.on_click(self.restart_selected_models)

        # Add the widgets to this container
        self.children = [self.output, ipw.HBox([self._stop_btn, self._restart_btn]), self.param_table]
        
        # store all the model related elements and futures
        table_size = self.param_table.get_changed_df().shape[0]
        self.model_submits = [None] * table_size
        self.model_runs = [None] * table_size
        self.model_plots = [self.vis_func() for i in range(table_size)]
        self.model_displays = [None] * table_size
        self.model_watchers = [None] * table_size
        self.model_updaters = [None] * table_size
        
        # thread Events and Queues for managing resources
        self.model_messages = [queue.Queue() for i in range(table_size)]
        self.table_ready = threading.Event()
        self.output_ready = threading.Event()
        self.table_ready.set()
        self.output_ready.set()

        self._active_plot = 0
        
        self.on_displayed(self._render_plots)

    def add_handlers(self):
        """Add event handlers to the table"""
        self.param_table.on('selection_changed', self.display_visualization)

    def remove_handlers(self):
        """Remove event handlers from the table"""
        self.param_table.off('selection_changed', self.display_visualization)

    def submit_computations(self):
        """Start threads that submit tasks to IPyParallel and update data"""
        try:            
            # paramset_id is the row index,
            # paramset is the dictionary of params
            paramitems = []
            for i in range([len(v) for v in self.compute_params.values()][0]):
                paramitems.append((i, {k: self.compute_params[k][i] for k in self.compute_params}))

            self.debug.append_stdout("Submitting: {}\n".format(paramitems))
            for paramset_id, paramset in paramitems:
                self.model_submits[paramset_id] = self.executor.submit(self.model_submit(paramset_id, paramset))
        except Exception as e:
            self.debug.append_stdout("Exception while submitting runs: {}\n".format(e.args))

    def model_submit(self, paramset_id, paramset):
        """Submit the model run to IPyParallel, then set a producer and consumer to handle data output"""
        try:
            # Launch model run via IPyParallel
            self.model_runs[paramset_id] = self.lview.apply(self.compute_func, **paramset)
            # Monitor incoming data
            self.model_watchers[paramset_id] = self.executor.submit(self.watch_data, paramset_id, self.model_runs[paramset_id])
            # Update table and plots
            self.model_updaters[paramset_id] = self.executor.submit(self.update_data, paramset_id, self.model_runs[paramset_id])
        except Exception as e:
            self.debug.append_stdout("Exception while submitting model runs, starting producer and consumer: {}\n".format(e.args))

    def watch_data(self, id, fut, interval=1):
        """Producer of data messages from IPyParallel"""
        try:
            last_status = ""
            done = False
            while not done and fut is not None:
                # once the future completes, don't exit until the last status message has come in
                if fut.ready() and last_status == "Ended Training":
                    done = True

                if fut.data and len(fut.data) > 0 and fut.data['status'] != last_status:
                    last_status = fut.data['status']
                    local_data = copy.deepcopy(fut.data)
                    #self.debug.append_stdout("Pushing {}\n".format(local_data))
                    self.model_messages[id].put(pickle.dumps(local_data))
                else:
                    time.sleep(1)
        except Exception as e:
            self.debug.append_stdout("Exception while updating table and plot: {}\n".format(e.args))

    def update_data(self, id, fut):
        """Consumer of data messages, updates table and plots"""
        try:
            while 1:
                raw_data = self.model_messages[id].get()
                #self.debug.append_stdout("data consumer for {}: {}\n".format(id, local_data))

                # Handle empty queue
                if raw_data is None:
                    # quit once all data has been processed
                    if fut is None or fut.ready():
                        break
                    # otherwise, the queue is empty, but the producer may not be finished
                    else:
                        continue

                local_data = pickle.loads(raw_data)

                self.debug.append_stdout("Received data for {}: {}\n".format(id, fut.data))
                #self.debug.append_stdout("Updating table with data\n")
                self.table_ready.wait()
                self.table_ready.clear()

                self.update_table(id, local_data)

                if local_data['status'] == "Ended Epoch":
                    #self.debug.append_stdout("Updating table data for {}: {}\n".format(id, local_data))
                    #self.debug.append_stdout("Plotting data for {}: {}\n".format(id, local_data))
                    self.model_plots[id].update(local_data)
                    #if self._active_plot == id:
                    #    self.display_plot_updates(id)

                #self.debug.append_stdout("Releasing data lock\n")
                self.table_ready.set()
                self.model_messages[id].task_done()
        except Exception as e:
            self.debug.append_stdout("Exception while updating {} table and plot: {}\n".format(id, e.args))
        finally:
            self.model_messages[id].task_done()

    def update_table(self, id, data):
        try:
            for k in data:
                self.debug.append_stdout("Updating {} for {} with {}\n".format(k, id, data[k]))
                if k == 'logs':
                    for kl in data['logs']:
                        if kl in self.params_df:
                            self.param_table._handle_qgrid_msg_helper({
                                'type': 'cell_change',
                                'column': kl,
                                'row_index': id,
                                'unfiltered_index': id,
                                'value': data['logs'][kl]
                            })
                elif k in self.params_df:
                    self.param_table._handle_qgrid_msg_helper({
                        'type': 'cell_change',
                        'column': k,
                        'row_index': id,
                        'unfiltered_index': id,
                        'value': data[k]
                    })
            self.param_table._update_table()
        except Exception as e:
            self.debug.append_stdout("Exception while updating table with data from {} : {}, {}\n".format(id, data, e.args))

    def _render_plots(self, *args):
        for i in range(self.params_df.shape[0]):
            with self.output:
                clear_output(wait=True)
                self.model_displays[i] = display(self.model_plots[i], display_id=True)

    def display_plot_updates(self, id):
        try:
            model_id = self.param_table.get_changed_df().index[id]
            self.output_ready.wait()
            self.output_ready.clear()
            with self.output:
                clear_output(wait=True)
                update_display(self.model_plots[model_id], display_id=self.model_displays[model_id])
            self.output_ready.set()
        except Exception as e:
            self.debug.append_stdout("Exception while rendering plot {}: {}\n".format(id, e.args))

    def display_visualization(self, event, widget_instance):
        try:
            self.debug.append_stdout("Event received: {}\n".format(event))

            # this means that all rows have been deselected
            if len(event['new']) == 0:
                return

            self._active_plot = event['new'][0]
            self.display_plot_updates(self._active_plot)
        except Exception as e:
            self.debug.append_stdout("Exception while switching to plot {}: {}\n".format(event, e.args))

    def stop_selected_models(self, event):
        srows = self.param_table.get_selected_rows()
        self.debug.append_stdout("Stop rows {}\n".format(srows))
    
    def restart_selected_models(self, event):
        srows = self.param_table.get_selected_rows()
        self.debug.append_stdout("Restart rows {}\n".format(srows))