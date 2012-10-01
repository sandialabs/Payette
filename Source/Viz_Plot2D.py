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

from numpy import arange
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

size=(700,600)

class Viz_Plot2D(HasTraits):
    container = Instance(Plot)
    plot_data = List(Array)
    overlay_plot_data = Dict(Str, Array)
    overlay_headers = Dict(Str, List(Str))
    run_names = List(Str)
    headers = List(Str)
    plot_indices = List(Int)
    axis_index = Int
    Time = Float
    high_time = Float
    low_time = Float
    time_data_labels = Dict(Int,List)
    runs_shown = List(Bool)

    traits_view = View(
                    Group(
                        Item('container', editor=ComponentEditor(size=size),
                             show_label=False),
                        Item('Time',
                             editor = RangeEditor( low_name    = 'low_time',
                                                   high_name   = 'high_time',
                                                   format      = '%.1f',
                                                   label_width = 28,
                                                   mode        = 'auto' )),
                        orientation = "vertical"),
                    resizable=True,
                    width=size[0], height=size[1]+100
                    )

    def __init__(self, **traits):
         HasTraits.__init__(self, **traits)
         self.time_data_labels = {}
         self.axis_index = 0
         self.runs_shown = [True] * len(self.run_names)

    @on_trait_change('Time')
    def change_data_markers(self):
         ti = self.find_time_index()
         for d in range(len(self.plot_data)):
            if d not in self.time_data_labels:
                continue

            for i, di in enumerate(self.plot_indices):
               self.time_data_labels[d][i].data_point = (self.plot_data[d][ti,self.axis_index], self.plot_data[d][ti,di])

         self.container.invalidate_and_redraw()

    def create_container(self):
         container = Plot(padding = 80, fill_padding = True,
                          bgcolor = "white", use_backbuffer=True,
                          border_visible = True)
         return container

    def find_time_index(self):
         list_of_diffs = [ abs(x-self.Time) for x in self.plot_data[0][:,0] ]
         tdx = list_of_diffs.index(min(list_of_diffs))
         return tdx

    def change_axis(self, index):
         self.axis_index = index
         self.change_plot(self.plot_indices)

    def create_data_label(self, xp, yp, d, di):
        if len(self.run_names) > 1:
            label_format = '[' + self.run_names[d] + '] (%(x).5g, %(y).5g)'
        else:
            label_format = '(%(x).5g, %(y).5g)'

        label = DataLabel(component=self.container, data_point=(xp, yp),
                           label_position="bottom right",
                           border_visible=False,
                           bgcolor="transparent",
                           label_format = label_format,
                          marker_color=tuple(COLOR_PALETTE[(d+di)%10]),
                           marker_line_color="transparent",
                           marker = "diamond",
                           arrow_visible=False)

        self.time_data_labels[d].append(label)
        self.container.overlays.append(label)

    def create_plot(self, x, y, di, d, plot_name, line_style):
         self.container.data.set_data("x " + plot_name, x)
         self.container.data.set_data("y " + plot_name, y)
         self.container.plot(("x " + plot_name, "y " + plot_name),
                             line_width=2.0,
                             name = plot_name,
                             color=tuple(COLOR_PALETTE[(d+di)%10]),
                             bgcolor = "white",
                             border_visible = True,
                             line_style=line_style)

    def change_plot(self, indices):
         self.plot_indices = indices
         self.container = self.create_container()
         self.high_time = float(max(self.plot_data[0][:,0]))
         self.low_time = float(min(self.plot_data[0][:,0]))
         self.container.data = ArrayPlotData()
         line_styles = ['dot dash', 'dash', 'dot', 'long dash']
         self.time_data_labels = {}
         if len(indices) == 0:
            return
         for d in range(len(self.plot_data)):
            if not self.runs_shown[d]:
                continue

            if len(self.run_names) > 1:
                run_name = " " + str(d) + " " + self.run_names[d]
            else:
                run_name = ""

            self.time_data_labels[d] = []
            x = self.plot_data[d][:,self.axis_index]
            ti = self.find_time_index()
            for i, di in enumerate(indices):
               y = self.plot_data[d][:,di]
               xp = self.plot_data[d][ti,self.axis_index]
               yp = self.plot_data[d][ti,di]
               self.create_plot(x, y, di, d, self.headers[di] + run_name, 'solid')
               self.create_data_label(xp, yp, d, di)
               if d == 0:
                   for j, k in enumerate(self.overlay_plot_data.keys()):
                       if self.headers[self.axis_index] == self.overlay_headers[k][self.axis_index] and \
                          self.headers[di] == self.overlay_headers[k][di]:
                           xo = self.overlay_plot_data[k][:,self.axis_index]
                           yo = self.overlay_plot_data[k][:,di]
                           self.create_plot(xo, yo, di, d, self.headers[di] + ' (' + k + ')',
                                            line_styles[j%4])

         add_default_grids(self.container)
         add_default_axes(self.container, htitle=self.headers[self.axis_index])

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

if __name__ == "__main__":
    p = Viz_Plot2D()
    p.configure_traits()
