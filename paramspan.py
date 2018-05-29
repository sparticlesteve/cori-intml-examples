import datetime
import warnings
import itertools as it
import time
import concurrent.futures as cf

import dill

import numpy as np
import pandas as pd

from IPython.display import display
import ipyparallel as ipp
import ipywidgets as ipw
import ipyvolume as ipv
import qgrid


## View

def param_grid(**kwargs):
    """Generate Cartesian product of keyword arguments"""
    try:
        prod_bool = kwargs.pop('_product')
    except KeyError:
        prod_bool = False
    name_list, values_list = zip(*kwargs.items())
    if prod_bool:
        prod = it.product(kwargs.items())
        value_combinations = list(it.product(*values_list))
    else:
        value_combinations = list(zip(*values_list))
    return name_list, value_combinations

def param_df(**kwargs):
    """Generate pandas DataFrame from Cartesian product of keyword arguments"""
    names, values = param_grid(**kwargs)
    return pd.DataFrame(values, columns=names)

def param_qgrid(qgrid_layout=None, **kwargs):
    """Generate Qgrid table from Cartesian product of keyword arguments"""
    if not qgrid_layout:
        qgrid_layout=ipw.Layout()
    return qgrid.QGridWidget(df=param_df(**kwargs), layout=qgrid_layout)

def dict_list_from_df(df):
    """Turn each row into a dictionary indexed by column names"""
    return [{col: val for col, val in zip(df.columns, df.loc[ind,:])} for ind in df.index]

class ParamSpanWidget(ipw.VBox):
    def __init__(self, compute_func, vis_func, product=False, live=True, output_layout=None, qgrid_layout=None):
        """
        live: bool
        Whether to pass future to visualize upon task start. Otherwise, results are given after task end.
        
        product: bool
        Whether to take the cartesian product of parameters (grid search). Otherwise, they must be the same length.
        """
        self.compute_func = compute_func
        self.vis_func = vis_func
        self.product = product
        self.live = live
        self.output_layout = output_layout
        self.qgrid_layout = qgrid_layout

        super().__init__()

        self.init_executor()
        self.init_ipp()
        self.init_widgets()
        self.init_layout()

    def init_executor(self):
        self.executor = cf.ThreadPoolExecutor()
        
    def init_ipp(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ipp_client = ipp.Client()
        self.dview = self.ipp_client.direct_view()
        self.lview = self.ipp_client.load_balanced_view()
        
    def init_widgets(self):
        if not self.output_layout:
            self.output_layout = ipw.Layout(height='500px', border='1px solid', overflow_x='scroll', overflow_y='scroll')
        if not self.qgrid_layout:
            qgrid_layout = ipw.Layout()

        self.output = ipw.Output(layout=self.output_layout)
        # param_table is empty until set_params is called
        self.param_table = param_qgrid(self.qgrid_layout, **{'':[]})

    def init_logic(self):
        self.param_table.observe(self.visualize_wrapper, names='_selected_rows')

    def init_layout(self):
        self.children = [
            self.output,
            self.param_table
        ]

    def set_params(self, **all_params):
        """Provide parameter set to search over
        all_params = {
            'param1': [val1, val2, ...],
            'param2': [val3, val4, ...],
            ...
        }
        """
        self.param_table = param_qgrid(
            self.qgrid_layout, 
            _product=self.product,
            **all_params
        )
        self.init_logic()
        self.init_layout()

    def submit_and_store(self, paramset_id, paramset):
        """Submit one job and store the results, indexed by id."""
        fut = self.lview.apply(self.compute_func, **paramset)
        self.compute_futures[paramset_id] = fut
        self.results[paramset_id] = fut.result()
        
    def submit_computations(self, *change):
        # def compute_wrapper(compute_func, name, paramset_id, params, library):
        #     """Perform computation and send results to MongoDB for one set of params"""
        #     results = compute_func(**params)

        # Loop over all sets of parameters
        # paramset_id is the row index,
        # paramset is the dictionary of params
        paramitems = self.param_table.df.T.to_dict().items()
        self.results = [None]*len(paramitems)
        self.compute_futures = [None]*len(paramitems)
        self.save_futures = [None]*len(paramitems)
        for paramset_id, paramset in paramitems:
            # Submit task to IPyParallel
            # fut =  self.lview.apply(compute_wrapper, self.compute_func, self.name, paramset_id, paramset, self.library)
            # One thread per task so that results can be saved asynchronously when they return
            # Keep the outer future to make sure all results have been saved
            self.save_futures[paramset_id] = self.executor.submit(self.submit_and_store, paramset_id, paramset)

    def visualize_wrapper(self, *change):
        """Call visualization function and capture output"""
        # Do nothing if selection is empty:
        # Empty list evaluates to False
        if not self.param_table.get_selected_rows():
            return

        # Get params from selected row (take first row if >1 selected)
        # The ordering is not necessarily the same as in widget,
        # so weird things might happen if multiple rows are selected.
        paramset_id = self.param_table.get_selected_df().index[0]
        paramset = self.param_table.df.loc[paramset_id, :]

        # Clear screen if the results of this computation
        # are not available. Assume empty dict by default
        compute_results = []
        
        future = self.compute_futures[paramset_id]
        # If live, then just pass future to vis_func and let it receive data.
        if self.live:
            @self.output.capture(clear_output=True, wait=True)
            def wrapper():
                display(self.vis_func(future))
            wrapper()
        
        # Otherwise, wait until the task finishes and pass returned values.
        else:
            # future is initially None
            if future and future.done():
                compute_results = self.results[paramset_id]

                # Avoid using output context to ensure that
                # only this function's output is included.
                @self.output.capture(clear_output=True, wait=True)
                def wrapper():
                    display(self.vis_func(**compute_results))
            else:
                @self.output.capture(clear_output=True, wait=True)
                def wrapper():
                    print("Task {} not done: {}".format(paramset_id, self.compute_futures[paramset_id]))
                    
            wrapper()
            
    def get_entries(self):
        """Get all results stored in database from parameter spans with this name"""
        return [entry for entry in self.library.list_symbols() if entry[:len(self.name)+1] == self.name+'-']
      
# User functions

def exp_compute(N, mean, std, color):
    import numpy as np
    from datetime import datetime
    x = np.random.normal(loc=mean, scale=std, size=N)
    realmean = np.mean(x)
    realstd = np.std(x)
    return {
        'engine_id': engine_id,
        'date': datetime.now().ctime(),
        'x': x,
        'N': N,
        'realmean': realmean,
        'realstd': realstd,
        'color': color
    }
    
def exp_viz(engine_id, date, x, N, realmean, realstd, color):
    print("Computed on engine {} at {}".format(engine_id, date))
    plt.figure(figsize=[8,5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.distplot(x, color=color)
    plt.title(r'$N={}$, $\mu={:.2f}$, $\sigma={:.2f}$'.format(N, realmean, realstd))
    plt.show()
    print("Data: {}".format(x))
