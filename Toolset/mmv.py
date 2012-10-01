#!/usr/bin/env python
# Copyright (2011) Sandia Corporation. Under the terms of Contract
# DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain
# rights in this software.

# The MIT License

# Copyright (c) Sandia Corporation

# License for the specific language governing rights and limitations under
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


"""
Create the next to simplest chaco plot: separate the model and the view, and add
a control on the model.
"""
from numpy import linspace, sin, loadtxt
from numpy import loadtxt
import scipy.stats as ss
import linecache as lc
import math
import sys
import os

exename = os.path.basename(__file__)
usage = """{0}
usage: {0} <{{file,dir}}.out> <@COLUMN1> <@COLUMN2>""".format(exename)

import Source.Payette_sim_index as psi

verbose = False

try:
    from chaco.api import ArrayPlotData, Plot, DataLabel
    from chaco.tools.api import PanTool, ZoomTool
    from enable.component_editor import ComponentEditor
    from traits.api import HasTraits, Instance, DelegatesTo, \
        on_trait_change, Array, Range
    from traitsui.api import View, Item

except ImportError:
    from enthought.chaco.api import ArrayPlotData, Plot, DataLabel
    from enthought.chaco.tools.api import PanTool, ZoomTool
    from enthought.enable.component_editor import ComponentEditor
    from enthought.traits.api import HasTraits, Instance, DelegatesTo, \
        on_trait_change, Array, Range
    from enthought.traits.ui.api import View, Item

except:
    sys.exit("Enthought tools must be installed to use {0}".format(exename))

if __name__ == "__main__":
    if not sys.argv[1:] or "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print usage
        msg = (None if sys.argv[1:]
               else "Valid output file or directory must be given")
        sys.exit(msg)

# ===============================================================
class swan:
    def __init__(self):

        # 'TIME' should always be used
        input_columns = ["TIME"]

        argv = [x for x in sys.argv[1:]]
        for arg in [x for x in argv]:
            if arg[0] == "@":
                input_columns.append(arg[1:])
                argv.remove(arg)
            continue

        if len(input_columns) != 3:
            print usage
            sys.exit("Please specify two columns for comparison")

        self.input_cols = input_columns
        self.xlabel = self.input_cols[1]
        self.ylabel = self.input_cols[2]

        if len(argv) != 1:
            print usage
            sys.exit("Input syntax error")
        arg = os.path.realpath(argv[0])

        if not os.path.exists(arg):
            print usage
            sys.exit("{0} not found".format(os.path.basename(arg)))

        if os.path.isfile(arg) and arg.endswith(".out"):
            output_files = [arg]

        else:
            if os.path.isdir(arg):
                sim_index = psi.SimulationIndex(base_dir=arg)

            else:
                sim_index = psi.SimulationIndex(index_file=arg)

            output_files = sim_index.output_files()
            not_found = [x for x in output_files if not os.path.isfile(x)]
            if not_found:
                sys.exit("The following output files were not found:\n{0}"
                         .format(", ".join(not_found)))

        self.output_files = output_files
        self.data = []
        self.datanames = []
        self.timepoints = []
        self.timepointsnames = []
        self.update_data()

    def update_data(self):
        # Make sure that the headers on each file are the same
        header = lc.getline(self.output_files[0],1)
        same_headers = [header == lc.getline(out_f,1) for out_f in self.output_files]
        if not all(same_headers):
            sys.exit("Not all the output files have the same header.")

        # Make sure that the columns requested are located in the header
        headers = [ x.upper() for x in header.split()]
        input_columns_idx = [x for x in self.input_cols]
        for idx, col in enumerate(self.input_cols):
            if not col.upper() in headers:
                sys.exit("'{0}' not found in headers.\n{1}".format(col,headers))
            input_columns_idx[idx] = headers.index(col)

        self.data = []
        for file_name in self.output_files:
            dbt, dbx, dby = loadtxt(file_name, skiprows=1, unpack=True,
                                    usecols=tuple(input_columns_idx))
            dbt = [float(x) for x in dbt]
            dbx = [float(x) for x in dbx]
            dby = [float(x) for x in dby]
            maxt = max(dbt)
            mint = min(dbt)
            ranget = maxt-mint
            if ranget > 1.0:
                dbt = [(x-mint)/ranget for x in dbt]

            self.data.append([dbt, dbx, dby])
            self.datanames.append(["time{0:03d}".format(len(self.data)),
                                   "x{0:03d}".format(len(self.data)),
                                   "y{0:03d}".format(len(self.data))])
            self.timepoints.append([ [dbx[0]],[dby[0]] ] )
            self.timepointsnames.append([ "x2{0:03d}".format(len(self.data)),
                                          "y2{0:03d}".format(len(self.data))])

    def reset_timepoints(self,t):
        linreg_x = []
        linreg_y = []
        for idx in range(0,len(self.data)):
            if len(self.timepoints) != len(self.data):
                sys.exit("timepoints: {0} != {1}"
                         .format(len(self.timepoints),len(self.data)))
            # find the index closest to the value of 't'
            list_of_diffs = [ abs(x-t) for x in self.data[idx][0] ]
            tdx = list_of_diffs.index(min(list_of_diffs))

            self.timepoints[idx][0] = [self.data[idx][1][tdx]]
            self.timepoints[idx][1] = [self.data[idx][2][tdx]]
            linreg_x.append(self.timepoints[idx][0][0])
            linreg_y.append(self.timepoints[idx][1][0])
        if len(self.data) > 1:
            linreg = ss.linregress(linreg_x,linreg_y)
            if verbose:
                print("y = {0:.5e}*x + {1:.5e}".format(linreg[0],linreg[1]))

swan_db = swan()
my_time = swan_db.data[0][0]
my_x = swan_db.data[0][1]
my_y = swan_db.data[0][2]
# ===============================================================

class MyData(HasTraits):
    """ The model
    """
    x = my_x
    y = my_y
    x2 = [x[0]]
    y2 = [y[0]]
    param = Range(low=min(my_time),high=max(my_time))

    def __init__(self, *args, **kw):
        super(MyData, self).__init__(*args, **kw)
        self.param = min(my_time)

    @on_trait_change('param')
    def compute_y2(self):
        swan_db.reset_timepoints(self.param)
        list_of_diffs = [ abs(x-self.param) for x in my_time ]
        idx = list_of_diffs.index(min(list_of_diffs))
        self.x2 = [ self.x[idx] ]
        self.y2 = [ self.y[idx] ]

class MyPlot(HasTraits):
    """ The view
    """
    # The model
    data = Instance(MyData, ())
    # add a pointer to a parameter that we will want to control in the view
    param = DelegatesTo('data')

    # Plot object
    plot = Instance(Plot)
    # Data backing the plot
    plotdata = Instance(ArrayPlotData)

    traits_view = View(
            Item('plot', editor = ComponentEditor(), show_label = False),
            Item('param'),
                        width = 500, height = 500,
                        resizable = True, title = "Material Time History Viewer")

    def __init__(self, data, *args, **kw):
        """ Set up the view
        """
        self.data = data
#        self.plotdata = ArrayPlotData(x=self.data.x,y=self.data.y,
#                                      x2=self.data.x2,y2=self.data.y2)
        self.plotdata = ArrayPlotData()

        for idx in range(0,len(swan_db.data)):
            for jdx in range(0,len(swan_db.data[0])):
                self.plotdata.set_data(swan_db.datanames[idx][jdx],
                                       swan_db.data[idx][jdx])
            for jdx in range(0,len(swan_db.timepoints[0])):
                self.plotdata.set_data(swan_db.timepointsnames[idx][jdx],
                                       swan_db.timepoints[idx][jdx])

        #self.plotdata.set_data("x",self.data.x)
        #self.plotdata.set_data("y",self.data.y)
        #self.plotdata.set_data("x2",self.data.x2)
        #self.plotdata.set_data("y2",self.data.y2)

        super(MyPlot, self).__init__(*args, **kw)

    def _plot_default(self):
        """ Set up the plot
        """
        # Create a plot and hook it up to the data it will represent
        plot = Plot(self.plotdata)
        #vars = [ ["x","y","line"], ["x2","y2","scatter"] ]
        # Add a line plot showing y as a function of x
        #for var in vars:
        #    plot.plot((var[0],var[1]), type = var[2], color = "blue")

        plot.tools.append(PanTool(plot, constrain_key="shift"))
        zoom = ZoomTool(component=plot, tool_mode="box", always_on=False)
        plot.overlays.append(zoom)


        for idx in range(0,len(swan_db.data)):
            fac = 1.0
            if len(swan_db.data) != 1:
                fac = float(idx)/float(len(swan_db.data)-1)
            facRG = min(fac,0.5)
            facGB = max(fac,0.5) - 0.5
            R = (math.cos(facRG*math.pi)*math.cos(facGB*math.pi))
            G = (math.sin(facRG*math.pi)*math.cos(facGB*math.pi))
            B = (math.sin(facGB*math.pi))
            plot.plot((swan_db.datanames[idx][1],
                       swan_db.datanames[idx][2]),
                      type="line",color=(R,G,B))
            plot.plot((swan_db.timepointsnames[idx][0],
                       swan_db.timepointsnames[idx][1]),
                      type="scatter",color=(R,G,B))

        return plot

    @on_trait_change('param')
    def update_plot_data(self,x,y):
        """ Update the data behind the plot. Since they are hooked, no need to
        tell the plot to redraw.
        """
#        self.plotdata.set_data("x", self.data.x)
#        self.plotdata.set_data("y", self.data.y)
        #self.plotdata.set_data("x2", self.data.x2)
        #self.plotdata.set_data("y2", self.data.y2)
        for idx in range(0,len(swan_db.data)):
            for jdx in range(0,len(swan_db.timepoints[0])):
                self.plotdata.set_data(swan_db.timepointsnames[idx][jdx],
                                       swan_db.timepoints[idx][jdx])

if __name__ == "__main__":
    d = MyData()
    app_view = MyPlot(data = d)
    app_view.configure_traits()
