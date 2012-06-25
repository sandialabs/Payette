
from numpy import arange
from scipy.special import jn

try:

  from chaco.example_support import COLOR_PALETTE
  from enable.example_support import DemoFrame, demo_main
  from enable.api import Window, Component, ComponentEditor
  from traits.api import HasTraits, Instance, Array, List, Str, Int, Float, on_trait_change
  from traits.ui.api import Item, Group, View, RangeEditor
  from chaco.api import create_line_plot, add_default_axes, \
          add_default_grids, OverlayPlotContainer, PlotLabel, \
          create_scatter_plot, Legend, DataLabel
  from chaco.tools.api import PanTool, ZoomTool, LegendTool, \
          TraitsTool, DragZoom

except ImportError:

  from enthought.chaco.example_support import COLOR_PALETTE
  from enthought.enable.example_support import DemoFrame, demo_main
  from enthought.enable.api import Window, Component, ComponentEditor
  from enthought.traits.api import HasTraits, Instance, Array, List, Str, Int, Float, on_trait_change 
  from enthought.traits.ui.api import Item, Group, View, RangeEditor
  from enthought.chaco.api import create_line_plot, add_default_axes, \
          add_default_grids, OverlayPlotContainer, PlotLabel, \
          create_scatter_plot, Legend, DataLabel
  from enthought.chaco.tools.api import PanTool, ZoomTool, LegendTool, \
          TraitsTool, DragZoom

size=(700,600)

class Viz_Plot2D(HasTraits):
    container = Instance(Component)
    plot_data = Array
    headers = List(Str)
    plot_indices = List(Int)
    axis_index = Int
    Time = Float
    high_time = Float
    low_time = Float
    time_data_labels = List
    
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
         self.high_time = float(max(self.plot_data[:,0]))
         self.low_time = float(min(self.plot_data[:,0]))
         self.time_data_labels = []
         self.axis_index = 0

    @on_trait_change('Time')
    def change_data_markers(self):
         ti = self.find_time_index()
         for i, di in enumerate(self.plot_indices):
             self.time_data_labels[i].data_point = (self.plot_data[ti,self.axis_index], self.plot_data[ti,di])
         self.container.invalidate_and_redraw()
    
    def _container_default(self):
         container = OverlayPlotContainer(padding = 80, fill_padding = True,
                                          bgcolor = "lightgray", use_backbuffer=True)
         return container

    def find_time_index(self):
         list_of_diffs = [ abs(x-self.Time) for x in self.plot_data[:,0] ]
         tdx = list_of_diffs.index(min(list_of_diffs))
         return tdx

    def change_axis(self, index):
         self.axis_index = index
         self.change_plot(self.plot_indices)

    def change_plot(self, indices):
         self.plot_indices = indices
         self.container.components[:] = []
         value_mapper = None
         index_mapper = None
         plots = {}
         self.time_data_labels = []
         if len(indices) == 0:
            return
         x = self.plot_data[:,self.axis_index]
         for i, di in enumerate(indices):
            y = self.plot_data[:,di]
            plot = create_line_plot((x,y), color=tuple(COLOR_PALETTE[i%10]), width=2.0)
            plot.index.sort_order = "ascending"
            plot.bgcolor = "white"
            plot.border_visible = True

            ti = self.find_time_index()
            label = DataLabel(component=plot, data_point=(self.plot_data[ti,self.axis_index], self.plot_data[ti,di]),
                              label_position="bottom right",
                              border_visible=False,
                              bgcolor="transparent",
	                      marker_color=tuple(COLOR_PALETTE[i%10]),
                              marker_line_color="transparent",
                              marker = "diamond",
                              arrow_visible=False)
            plot.overlays.append(label)
            self.time_data_labels.append(label)

            if i != 0:
               plot.value_mapper = value_mapper
               value_mapper.range.add(plot.value)
               plot.index_mapper = index_mapper
               index_mapper.range.add(plot.index)
            else:
               value_mapper = plot.value_mapper
               index_mapper = plot.index_mapper
               add_default_grids(plot)
               add_default_axes(plot, htitle=self.headers[self.axis_index])
               plot.index_range.tight_bounds = False
               plot.index_range.refresh()
               plot.value_range.tight_bounds = False
               plot.value_range.refresh()
               plot.tools.append(PanTool(plot))

               # The ZoomTool tool is stateful and allows drawing a zoom
               # box to select a zoom region.
               zoom = ZoomTool(plot, tool_mode="box", always_on=False)
               plot.overlays.append(zoom)

               # The DragZoom tool just zooms in and out as the user drags
               # the mouse vertically.
               dragzoom = DragZoom(plot, drag_button="right")
               plot.tools.append(dragzoom)

               # Add a legend in the upper right corner, and make it relocatable
               legend = Legend(component=plot, padding=10, align="ur")
               legend.tools.append(LegendTool(legend, drag_button="right"))
               plot.overlays.append(legend)

            self.container.add(plot)
            plots[self.headers[di]] = plot

         legend.plots = plots
         self.container.invalidate_and_redraw()
    
if __name__ == "__main__":
    p = Viz_Plot2D()
    p.configure_traits()
