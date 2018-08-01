# stdlib
import copy
import threading
import time
import traceback

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
        self.title = title or "{} vs {}".format(self.ylabel, self.xlabel)

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
            layout=ipw.Layout(height='550px', width='100%'),
            title=self.title)
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
                labels=[y],
                enable_hover=True
            ))
            self.scatters.append(bq.Scatter(
                x=[],
                y=[],
                scales={'x': self.xscale, 'y': self.yscale},
                colors=[color],
                enable_hover=True
            ))
            self.labels.append(y)
            self.num_lines += 1

            self.lines[-1].tooltip = bq.Tooltip(
                fields=['name'],
                show_labels=True)
            self.lines[-1].interactions = {
                'hover': 'tooltip',
                'click': 'tooltip'
            }
                        
            self.scatters[-1].tooltip = bq.Tooltip(
                fields=['y','x'],
                labels=[y, self.xlabel], 
                formats=['.4f', ''],
                show_labels=True)
            self.scatters[-1].interactions = {
                'hover': 'tooltip',
                'click': 'tooltip'
            }
        except Exception as e:
            self.debug.append_stdout("Exception when adding a line and points to plot: {}".format(e.args))

    def resize_fig(self):
        try:
            for i in range(len(self.lines)):
                if len(self.lines[i].x) > 0:
                    self.xscale.min = min(self.xscale.min, float(np.min(self.lines[i].x)))
                    self.xscale.max = max(self.xscale.max, float(np.max(self.lines[i].x)))
                    self.yscale.min = min(self.yscale.min, float(np.min(self.lines[i].y)))
                    self.yscale.max = max(self.yscale.max, float(np.max(self.lines[i].y)))
        except Exception as e:
            self.debug.append_stdout("Exception when resizing the figure: {}\n".format(e.args))

    def update(self, data):
        try:
            for i in range(self.num_lines):
                self.lines[i].y = np.array(data[self.y[i]])
                self.scatters[i].y = np.array(data[self.y[i]])

                if self.x and self.x in data:
                    self.lines[i].x = np.array(data[self.x])
                    self.scatters[i].x = np.array(data[self.x])
                else:
                    self.lines[i].x = np.array([i for i in range(len(self.lines[0].y))])
                    self.scatters[i].x = np.array([i for i in range(len(self.lines[0].y))])
                
            self.resize_fig()
        except Exception as e:
            self.debug.append_stdout("Exception while plotting lines and resizing figure: {}\n".format(e.args))
            self.debug.append_stdout("Data: {}\n".format(data))


class ParamSpanWidget(ipw.VBox):
    def __init__(self, compute_func, vis_func, params, columns=None, ipp_cluster_id=None, 
                 output_layout=None, qgrid_layout=None):
        """
        compute_func: function 
        task to submit to IPyParallel for model output
        
        vis_func: function 
        function that produces a visualization of the model output (e.g. ModelPlot)

        params: dict
        grid search parameters, either lists/numpy arrays or list of lists/2D numpy arrays, where the outer lists have the same length
        
        ipp_cluster_id: str
        optional ipyparallel cluster id for connecting to a specific controller
        """
        super().__init__()

        self.compute_func = compute_func
        self.vis_func = vis_func
        self.output_layout = output_layout or \
            ipw.Layout(height='600px', border='1px solid', overflow_x='scroll', overflow_y='scroll')
        self.debug_layout = ipw.Layout(height='500px', border='1px solid', overflow_x='scroll', overflow_y='scroll')
        self.qgrid_layout = qgrid_layout or ipw.Layout()

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
        self._num_models = self.param_table.get_changed_df().shape[0]
        self.model_plots = [self.vis_func(title="Model {}: {}".format(i, 
                {k: self.compute_params[k][i] for k in self.compute_params})) for i in range(self._num_models)]
        self.model_displays = [None for i in range(self._num_models)]
        self.model_data = [
            ModelTaskData(["epoch","loss","val_loss","acc","val_acc"],["status","epoch"]) for i in range(self._num_models)]
        self._model_controller = ModelController(ipp_cluster_id=ipp_cluster_id)

        # select the first row by default
        self._active_plot = 0
        self.param_table._handle_qgrid_msg_helper({'type': 'selection_changed', 'rows': [0]})
        
        self._stop_updates = threading.Event()
        self._stop_updates.clear()
        self._update_thread = threading.Thread(target=self.update_data)
        self._update_thread.start()

    def add_handlers(self):
        """Add event handlers to the table"""
        self.param_table.on('selection_changed', self.display_visualization)

    def remove_handlers(self):
        """Remove event handlers from the table"""
        self.param_table.off('selection_changed', self.display_visualization)
        
    def submit_computations(self):
        """Start all models"""
        try:            
            for i in range(self._num_models):
                self._model_controller.start_model(
                    i,
                    self.compute_func, 
                    {k: self.compute_params[k][i] for k in self.compute_params})
        except Exception as e:
            self.debug.append_stdout("Exception while submitting runs: {}\n".format(e.args))

    def update_data(self, interval=1):
        try:
            while not self._stop_updates.is_set():
                active_models = self._model_controller.get_running_models()

                for model_id in active_models:
                    data = active_models[model_id].data
                    table_updated = False
                    
                    if len(data) == 0:
                        continue
                    
                    if "history" in data and len(data["history"]["epoch"]) > 0:
                        current_data_length = self.model_data[model_id].num_data_rows
                        history_data_length = len(data["history"]["epoch"])

                        if current_data_length < history_data_length:
                            if current_data_length == 0:
                                i = 0
                            else:
                                i = current_data_length - 1
                        
                            while i < history_data_length:
                                self.model_data[model_id].append_plot_data_row(
                                    {k: data["history"][k][i] for k in data["history"]})
                                i += 1

                            # apply plot data update
                            if model_id == self._active_plot:
                                self.model_plots[model_id].update(self.model_data[model_id].get_plot_data())

                            for k in data["history"]:
                                self.param_table._handle_qgrid_msg_helper({
                                    'type': 'cell_change',
                                    'column': k,
                                    'row_index': model_id,
                                    'unfiltered_index': model_id,
                                    'value': data["history"][k][-1]
                                })
                            table_updated = True

                    if "status" in data and self.param_table.get_changed_df()["status"][model_id]:
                        self.param_table._handle_qgrid_msg_helper({
                            'type': 'cell_change',
                            'column': "status",
                            'row_index': model_id,
                            'unfiltered_index': model_id,
                            'value': data["status"]
                        })
                        table_updated = True
                        
                        if data["status"] == "Ended Training":
                            self._model_controller.set_model_completed(model_id)
                    
                    if "epoch" in data:
                        self.param_table._handle_qgrid_msg_helper({
                            'type': 'cell_change',
                            'column': "epoch",
                            'row_index': model_id,
                            'unfiltered_index': model_id,
                            'value': data["epoch"]
                        })
                        table_updated = True

                    if table_updated:
                        self.param_table._update_table()                                        

                time.sleep(interval)
        except Exception as e:
            self.debug.append_stdout("Exception while applying updates from futures: {}\n".format(traceback.format_exc(e)))


    def display_visualization(self, event, widget_instance):
        try:
            self.debug.append_stdout("Event received: {}\n".format(event))

            # this means that all rows have been deselected
            if len(event['new']) == 0:
                return

            row_id = event['new'][0]
            model_id = self.param_table.get_changed_df().index[row_id]
            self._active_plot = model_id
                        
            # only update the plot if there is more data since the last viewing
            if self.model_data[model_id].num_data_rows > len(self.model_plots[model_id].lines[0].y):
                plot_data = self.model_data[model_id].get_plot_data()
                self.model_plots[model_id].update(plot_data)
            
            with self.output:
                clear_output(wait=True)
                if self.model_displays[model_id] is None:
                    self.model_displays[model_id] = display(self.model_plots[model_id], display_id=True)
                else:
                    update_display(self.model_plots[model_id], display_id=self.model_displays[model_id])
        except Exception as e:
            self.debug.append_stdout("Exception while switching to plot {}: {}\n".format(event, e.args))

    def stop_selected_models(self, event):
        srows = self.param_table.get_selected_rows()
        self.debug.append_stdout("Stop rows {}\n".format(srows))

        #for row_id in srows:
        #    model_id = self.param_table.get_changed_df().index[row_id]
    
    def restart_selected_models(self, event):
        srows = self.param_table.get_selected_rows()
        self.debug.append_stdout("Restart rows {}\n".format(srows))
        
        #for row_id in srows:
        #    model_id = self.param_table.get_changed_df().index[row_id]

    def get_resource_usage(self, model_id):
        pass
    
    def get_models_status(self):
        status = self.param_table.get_changed_df()[["status"]]


class ModelController(object):
    def __init__(self, ipp_cluster_id=None):
        self._futures = []
        self._completed = []
        self._active_models = {}
        self._completed_models = {}
        self._ipp_client = ipp.Client(cluster_id=ipp_cluster_id)
        self._lview = self._ipp_client.load_balanced_view()

    def start_model(self, model_id, compute_func, params):
        self._futures.append(self._lview.apply(compute_func, **params))
        self._active_models[model_id] = len(self._futures) - 1

    def stop_model(self, model_id):
        pass
    
    def restart_model(self, model_id, compute_func, params):
        #self._futures[model_id] = self._lview.apply(compute_func, **params)
        pass
    
    def set_model_completed(self, model_id):
        if model_id not in self._completed:
            self._completed.append(model_id)
    
    def get_completed_models(self):
        return {k: self._futures[self._completed_models[k]] for k in self._completed_models}
    
    def get_running_models(self):
        for i in range(len(self._futures)):
            if self._futures[i] is not None and self._futures[i].done() and i in self._completed:
                self._futures[i] = None
                self._completed_models[i] = self._completed.index(i)
                del self._active_models[i]
        
        return {k: self._futures[self._active_models[k]] for k in self._active_models}


class ModelTaskData(object):
    def __init__(self, plot_columns, status_columns):
        super(ModelTaskData, self).__init__()
        
        self._plot_data = ModelPlotTable(plot_columns)
        self._status_data = {k: None for k in status_columns}
        self._updated = True

    @property
    def has_updates(self):
        return self._updated

    @property
    def num_data_rows(self):
        return len(self._plot_data.rows[0])
    
    def get_plot_data(self):
        return self._plot_data.to_dict()
    
    def append_plot_data_row(self, d):
        self._plot_data.append_row(d)
        self._updated = True
    
    def set_status_data(self, d):
        self._status_data.update(d)
        self._updated = True
    
    def get_status_data(self):
        return self._status_data


class ModelPlotTable(object):
    def __init__(self, column_names):
        super(ModelPlotTable, self).__init__()

        self._id = None
        self._num_columns = len(column_names)
        self._num_rows = 0
        self._column_map = {column_names[i]: i for i in range(len(column_names))}
        self._column_data = [list() for c in column_names]
    
    @property
    def columns(self):
        return list(self._column_map.keys())
    
    @property
    def rows(self):
        return self._column_data
        
    def append_column(self, name, vals=None):
        if name in self._column_map:
            raise KeyError("column {} is already in this table".format(name))

        if vals:
            if len(vals) == self._num_rows:
                self._column_data.append(list(vals))
            else:
                raise ValueError("Number of rows must match table")
        else:
            data = [None] * self._num_rows
            self._column_data.append(data)

        self._column_map[name] = len(self._column_data) - 1
        self._updated = True

    def append_row(self, column_data):
        for column_name in self._column_map:
            column_index = self._column_map[column_name]
            if column_name in column_data:
                self._column_data[column_index].append(column_data[column_name])
            else:
                self._column_data[column_index].append(None)

    def to_dict(self):
        return {k: self._column_data[v] for k, v in self._column_map.items()}
