import linecache
import Source.Payette_utils as pu
import Source.Payette_sim_index as psi
from Viz_Plot2D import Viz_Plot2D

try:

  from traits.api import HasStrictTraits, Instance, String, Button, \
                         Bool, List, Str, Property, HasPrivateTraits, on_trait_change
  from traitsui.api import View, Item, HSplit, VGroup, Handler, TabularEditor, HGroup, UItem
  from traitsui.tabular_adapter import TabularAdapter

except ImportError:
  # modification for importing MacPorts install of enthought tools
  from enthought.traits.api import (
    HasStrictTraits, Instance, String, Button,
    Bool, List, Str, Property, HasPrivateTraits, on_trait_change)
  from enthought.traits.ui.api import (
    View, Item, HSplit, VGroup, Handler, TabularEditor, HGroup, UItem)
  from enthought.traits.ui.tabular_adapter import TabularAdapter


Change_X_Axis_Enabled = True

class ChangeAxisHandler(Handler):

    def closed(self, info, is_ok):
        global Change_X_Axis_Enabled
        Change_X_Axis_Enabled = True

class ChangeAxis(HasStrictTraits):

    Change_X_Axis = Button('Change X-axis')
    Plot_Data = Instance(Viz_Plot2D)
    headers = List(Str)

    def __init__(self, **traits):
        HasStrictTraits.__init__(self, **traits)

    def _Change_X_Axis_fired(self):
        global Change_X_Axis_Enabled
        Change_X_Axis_Enabled = False
        ms = SingleSelect(choices=self.headers, plot=self.Plot_Data)
        ms.configure_traits(handler=ChangeAxisHandler())

    view = View(VGroup(
                      Item('Change_X_Axis',
                           enabled_when='Change_X_Axis_Enabled==True',
                           show_label = False),
                      show_border=True,
                      padding=15
                      ),
                width=150,
                height=50,
                resizable=True
               )

class SingleSelectAdapter(TabularAdapter):
    columns = [ ('Payette Outputs', 'myvalue') ]

    myvalue_text = Property

    def _get_myvalue_text(self):
        return self.item

class SingleSelect(HasPrivateTraits):
    choices  = List(Str)
    selected = Str
    plot = Instance(Viz_Plot2D)

    view = View(
        HGroup(
            UItem('choices',
                  editor     = TabularEditor(
                                   show_titles  = True,
                                   selected     = 'selected',
                                   editable     = False,
                                   multi_select = False,
                                   adapter      = SingleSelectAdapter())
        )),
        width=224,
        height=668,
        resizable=True,
        title='Change X-axis'
    )

    @on_trait_change( 'selected' )
    def _selected_modified ( self, object, name, new ):
        self.plot.change_axis(object.choices.index(object.selected))

class MultiSelectAdapter(TabularAdapter):
    columns = [ ('Payette Outputs', 'myvalue') ]

    myvalue_text = Property

    def _get_myvalue_text(self):
        return self.item

class MultiSelect(HasPrivateTraits):
    choices  = List(Str)
    selected = List(Str)
    plot = Instance(Viz_Plot2D)

    view = View(
        HGroup(
            UItem('choices',
                  editor     = TabularEditor(
                                   show_titles  = True,
                                   selected     = 'selected',
                                   editable     = False,
                                   multi_select = True,
                                   adapter      = MultiSelectAdapter())
        )),
        width=224,
        height=668,
        resizable=True
    )

    @on_trait_change( 'selected' )
    def _selected_modified ( self, object, name, new ):
        ind = []
        for i in object.selected:
           ind.append(object.choices.index(i))
        self.plot.change_plot(ind)

class Viz_ModelPlot(HasStrictTraits):

    Plot_Data = Instance(Viz_Plot2D)
    headers = List(Str)
    Multi_Select = Instance(MultiSelect)
    Change_Axis = Instance(ChangeAxis)
    file_path = String

    def __init__(self, **traits):
        HasStrictTraits.__init__(self, **traits)
        self.headers = pu.get_header(self.file_path)
        data = pu.read_data(self.file_path)
        self.Plot_Data = Viz_Plot2D(plot_data=data, headers=self.headers, axis_index=0)
        self.Multi_Select = MultiSelect(choices=self.headers, plot=self.Plot_Data)
        self.Change_Axis = ChangeAxis(Plot_Data=self.Plot_Data, headers=self.headers)

def create_Viz_ModelPlot(window_name, **kwargs):
    """Create the plot window

    Parameters
    ----------
    kwargs : dict
      "output file" : file path to simulation output file
      "index file" : file path to simulation index file

    Notes
    -----
    When run_payette is called for a single job simulation, the output is
    written to a file "simulation_name.out". If, however, run_payette is
    called for a multiple job simulation, an index file of the multiple output
    files is created. The index file has information about where the
    individual output files are located.
    """
    output_file = kwargs.get("output file")
    index_file = kwargs.get("index file")

    if output_file is None and index_file is None:
        pu.report_and_raise_error(
            "no output or index file given")
    elif output_file is not None and index_file is not None:
        pu.report_and_raise_error(
            "specify either an output or index file, not both")

    if index_file is not None:
        sim_index = psi.SimulationIndex(index_file=arg)
        output_files = sim_index.output_files()
        not_found = [x for x in output_files if not os.path.isfile(x)]
        if not_found:
            pu.report_and_raise_error(
                "The following output files were not found:\n{0}"
                .format(", ".join(not_found)))
        pu.report_and_raise_error(
            "ERROR: index file reading not yet supported")

    elif output_file is not None:
        output_files = [output_file]

    view = View(HSplit(
                       VGroup(
                             Item('Multi_Select',
                                  show_label=False, width=224, height=668, springy=True, resizable=True),
                             Item('Change_Axis',
                                  show_label=False),
                             ),
                       Item('Plot_Data',
                            show_label=False, width=800, height=768, springy=True, resizable=True)
                      ),
                style='custom',
                width=1124,
                height=868,
                resizable=True,
                title=window_name)

    main_window = Viz_ModelPlot(file_path=output_files[0])

    main_window.configure_traits(view=view)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation_name")
    parser.add_argument("file")

    args = parser.parse_args()

    kwargs = {"output file": args.file}
    create_Viz_ModelPlot(args.simulation_name, **kwargs)
