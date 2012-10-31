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

import numpy as np
from scipy.special import jn

from enthought.chaco.example_support import COLOR_PALETTE
from enthought.enable.example_support import DemoFrame, demo_main
from enthought.enable.api import Window, Component, ComponentEditor
from enthought.traits.api import HasTraits, Instance, Array, List, Str, Int, Float, Bool, Dict, on_trait_change
from enthought.traits.ui.api import Item, Group, View, RangeEditor
from enthought.chaco.api import create_line_plot, add_default_axes, \
        add_default_grids, PlotLabel, Plot, \
        create_scatter_plot, DataLabel, ArrayPlotData
from enthought.chaco.tools.api import PanTool, ZoomTool, \
        TraitsTool, DragZoom

SIZE = (700, 600)
LS = ['dot dash', 'dash', 'dot', 'long dash']

class Viz_Plot2D(HasTraits):
    container = Instance(Plot)
    plot_info = Dict(Int, Dict(Str, List(Str)))
    plot_data = List(Array)
    overlay_headers = Dict(Str, List(Str))
    overlay_plot_data = Dict(Str, Array)
    variables = List(Str)
    plot_indices = List(Int)
    x_idx = Int
    y_idx = Int
    Time = Float
    high_time = Float
    low_time = Float
    time_data_labels = Dict(Int,List)
    runs_shown = List(Bool)
    x_scale = Float
    y_scale = Float

    traits_view = View(
        Group(
            Item('container', editor=ComponentEditor(size=SIZE),
                 show_label=False),
            Item('Time',
                 editor = RangeEditor(low_name='low_time',
                                      high_name='high_time',
                                      format='%.1f',
                                      label_width=28,
                                      mode='auto' )),
            orientation="vertical"),
        resizable=True,
        width=SIZE[0], height=SIZE[1]+100)

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.time_data_labels = {}
        self.x_idx = 0
        self.y_idx = 0
        self.runs_shown = [True] * len(self.variables)
        self.x_scale, self.y_scale = 1., 1.
        pass

    @on_trait_change('Time')
    def change_data_markers(self):
        ti = self.find_time_index()
        for d in range(len(self.plot_data)):
            if d not in self.time_data_labels: continue

            for i, y_idx in enumerate(self.plot_indices):
                self.time_data_labels[d][i].data_point = (
                    self.plot_data[d][ti, self.x_idx] * self.x_scale,
                    self.plot_data[d][ti, y_idx] * self.y_scale)

        self.container.invalidate_and_redraw()
        return

    def create_container(self):
        container = Plot(padding=80, fill_padding=True,
                         bgcolor="white", use_backbuffer=True,
                         border_visible=True)
        return container

    def find_time_index(self):
        list_of_diffs = [abs(x - self.Time) for x in self.plot_data[0][:, 0]]
        tdx = list_of_diffs.index(min(list_of_diffs))
        return tdx

    def change_axis(self, index):
        """Change the x-axis of the current plot

        Parameters
        ----------
        index : int
            The column containing the new x-axis data

        Returns
        -------
        None

        """
        self.x_idx = index
        self.change_plot(self.plot_indices)
        return

    def create_data_label(self, xp, yp, d, di):
        nform = "[%(x).2g, %(y).2g]"
        if self.nfiles() - 1 or self.overlay_plot_data:
            lform = "({0}) {1}".format(self.get_file_name(d), nform)
        else:
            lform = nform
        label = DataLabel(component=self.container, data_point=(xp, yp),
                          label_position="bottom right",
                          border_visible=False,
                          bgcolor="transparent",
                          label_format=lform,
                          marker_color=tuple(COLOR_PALETTE[(d + di) % 10]),
                          marker_line_color="transparent",
                          marker="diamond",
                          arrow_visible=False)

        self.time_data_labels[d].append(label)
        self.container.overlays.append(label)
        return

    def create_plot(self, x, y, di, d, plot_name, ls):
        self.container.data.set_data("x " + plot_name, x)
        self.container.data.set_data("y " + plot_name, y)
        self.container.plot(
            ("x " + plot_name, "y " + plot_name),
            line_width=2.0, name=plot_name,
            color=tuple(COLOR_PALETTE[(d + di) % 10]),
            bgcolor="white", border_visible=True, line_style=ls)
        return

    def change_plot(self, indices, x_scale=None, y_scale=None):
        self.plot_indices = indices
        self.container = self.create_container()
        self.high_time = float(max(self.plot_data[0][:, 0]))
        self.low_time = float(min(self.plot_data[0][:, 0]))
        self.container.data = ArrayPlotData()
        self.time_data_labels = {}
        if len(indices) == 0:
            return
        x_scale, y_scale = self.get_axis_scales(x_scale, y_scale)

        # loop through plot data and plot it
        overlays_plotted = False
        for d in range(len(self.plot_data)):

            if not self.runs_shown[d]:
                continue

            # The legend entry for each file is one of the following forms:
            #   1) [file basename] VAR
            #   2) [file basename:] VAR variables
            # depending on if variabes were perturbed for this run.
            variables = self.variables[d]
            if len(variables) > 30:
                variables = ", ".join(variables.split(",")[:-1])
            if variables: variables = ": {0}".format(variables)

            self.time_data_labels[d] = []
            ti = self.find_time_index()
            mheader = self._mheader()
            xname = mheader[self.x_idx]
            self.y_idx = get_index(mheader, mheader[indices[0]])

            # indices is an integer list containing the columns of the data to
            # be plotted. The indices are wrt to the FIRST file in parsed, not
            # necessarily the same for every file. Here, we loop through the
            # indices, determine the name from the first file's header and
            # find the x and y index in the file of interest
            fnam, header = self.get_info(d)
            for i, idx in enumerate(indices):
                yname = mheader[idx]

                # get the indices for this file
                xp_idx = get_index(header, xname)
                yp_idx = get_index(header, yname)
                if xp_idx is None or yp_idx is None:
                    continue

                x = self.plot_data[d][:, xp_idx] * x_scale
                y = self.plot_data[d][:, yp_idx] * y_scale
                if self.nfiles() - 1 or self.overlay_plot_data:
                    entry = "({0}) {1}{2}".format(fnam, yname, variables)
                else:
                    entry = "{0} {1}".format(yname, variables)
                self.create_plot(x, y, yp_idx, d, entry, "solid")

                # create point marker
                xp = self.plot_data[d][ti, xp_idx] * x_scale
                yp = self.plot_data[d][ti, yp_idx] * y_scale
                self.create_data_label(xp, yp, d, yp_idx)

                if not overlays_plotted:
                    # plot the overlay data
                    overlays_plotted = True
                    ls_ = 0
                    for fnam, head in self.overlay_headers.items():
                        # get the x and y indeces corresponding to what is
                        # being plotted
                        xo_idx = get_index(head, xname)
                        yo_idx = get_index(head, yname)
                        if xo_idx is None or yo_idx is None:
                            continue
                        xo = self.overlay_plot_data[fnam][:, xo_idx] * x_scale
                        yo = self.overlay_plot_data[fnam][:, yo_idx] * y_scale
                        # legend entry
                        entry = "({0}) {1}".format(fnam, head[yo_idx])
                        self.create_plot(xo, yo, yo_idx, d, entry, LS[ls_ % 4])
                        ls_ += 1
                        continue

        add_default_grids(self.container)
        add_default_axes(self.container, htitle=mheader[self.x_idx])

        self.container.index_range.tight_bounds = False
        self.container.index_range.refresh()
        self.container.value_range.tight_bounds = False
        self.container.value_range.refresh()

        self.container.tools.append(PanTool(self.container))

        zoom = ZoomTool(self.container, tool_mode="box", always_on=False)
        self.container.overlays.append(zoom)

        dragzoom = DragZoom(self.container, drag_button="right")
        self.container.tools.append(dragzoom)

        self.container.legend.visible = True

        self.container.invalidate_and_redraw()
        return

    def _mheader(self):
        """Returns the "master" header - the header of the first file

        Returns
        -------
        header : list
        """
        return self.get_info(0)[1]

    def min_x(self):
        return np.amin(self.plot_data[0][:, self.x_idx])

    def max_x(self):
        return np.amax(self.plot_data[0][:, self.x_idx])

    def min_y(self):
        return np.amin(self.plot_data[0][:, self.y_idx])

    def max_y(self):
        return np.amax(self.plot_data[0][:, self.y_idx])

    def get_info(self, i):
        """Return the info for index i

        Parameters
        ----------
        i : int
            The location in self.plot_info

        Returns
        -------
        fnam : str
            the file name
        header : list
            the file header

        """
        return self.plot_info[i].items()[0]

    def get_file_name(self, i):
        return self.get_info(i)[0]

    def nfiles(self):
        return len(self.plot_info)

    def get_axis_scales(self, x_scale, y_scale):
        """Get/Set the scales for the x and y axis

        Parameters
        ----------
        x_scale : float, optional
        y_scale : float, optional

        Returns
        -------
        x_scale : float
        y_scale : float

        """
        # get/set x_scale
        if x_scale is None: x_scale = self.x_scale
        else: self.x_scale = x_scale

        # get/set y_scale
        if y_scale is None: y_scale = self.y_scale
        else: self.y_scale = y_scale

        return x_scale, y_scale

def get_index(list_, name):
    """Return the index for name in list_"""
    try:
        return [x.lower() for x in list_
                if x not in ("#", "$",)].index(name.lower())
    except ValueError:
        return None

if __name__ == "__main__":
    p = Viz_Plot2D()
    p.configure_traits()
