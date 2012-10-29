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

import os
import linecache
import os
import Source.Payette_utils as pu
import Source.Payette_sim_index as psi
from Viz_Plot2D import Viz_Plot2D

from enthought.traits.api import (HasStrictTraits, Instance, String, Button,
                                  Bool, List, Dict, Str, Int, Property,
                                  HasPrivateTraits, on_trait_change, Trait)
from enthought.traits.ui.api import View, Item, HSplit, VGroup, Handler, TabularEditor, HGroup, UItem
from enthought.traits.ui.tabular_adapter import TabularAdapter
from enthought.pyface.api import FileDialog, OK

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

    view = View(Item('Change_X_Axis',
                     enabled_when='Change_X_Axis_Enabled==True',
                     show_label=False
                     )
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

class SingleSelectOverlayFilesAdapter(TabularAdapter):
    columns = [ ('Overlay File Name', 'myvalue') ]

    myvalue_text = Property

    def _get_myvalue_text(self):
        return self.item

class SingleSelectOverlayFiles(HasPrivateTraits):
    choices  = List(Str)
    selected = Str

    view = View(
        HGroup(
            UItem('choices',
                  editor     = TabularEditor(
                                   show_titles  = True,
                                   selected     = 'selected',
                                   editable     = False,
                                   multi_select = False,
                                   adapter      = SingleSelectOverlayFilesAdapter())
        )),
        width=224,
        height=100
    )

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
        height=568,
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
    finfo = Dict(Int, Dict(Str, List(Str)))
    Multi_Select = Instance(MultiSelect)
    Change_Axis = Instance(ChangeAxis)
    Reset_Zoom = Button('Reset Zoom')
    Reload_Data = Button('Reload Data')
    Load_Overlay = Button('Open Overlay')
    Close_Overlay = Button('Close Overlay')
    Single_Select_Overlay_Files = Instance(SingleSelectOverlayFiles)
    file_paths = List(String)
    file_variables = List(String)

    def __init__(self, **traits):
        HasStrictTraits.__init__(self, **traits)

        # put together information to be sent to Viz_Plot2D
        # information needed:
        # finfo : dict
        #   {0: {file_0: header_0}}
        #   {1: {file_1: header_1}}
        #   ...
        #   {n: {file_n: header_n}}
        # variables : list
        #   list of variables that changed from one simulation to another
        # x_idx : int
        #   column containing x variable to be plotted

        data = []
        for idx, file_path in enumerate(self.file_paths):
            if idx == 0: mheader = pu.get_header(file_path)
            fnam = os.path.basename(file_path)
            self.finfo[idx] = {fnam: pu.get_header(file_path)}
            data.append(pu.read_data(file_path))
        self.Plot_Data = Viz_Plot2D(
            plot_data=data, variables=self.file_variables,
            x_idx=0, finfo=self.finfo)
        self.Multi_Select = MultiSelect(choices=mheader, plot=self.Plot_Data)
        self.Change_Axis = ChangeAxis(Plot_Data=self.Plot_Data, headers=mheader)
        self.Single_Select_Overlay_Files = SingleSelectOverlayFiles(choices=[])

    def _Reset_Zoom_fired(self):
        self.Plot_Data.change_plot(self.Plot_Data.plot_indices)

    def _Reload_Data_fired(self):
        data = []
        for idx, file_path in enumerate(self.file_paths):
            if idx == 0: mheader = pu.get_header(file_path)
            fnam = os.path.basename(file_path)
            self.finfo[idx] = {fnam: pu.get_header(file_path)}
            data.append(pu.read_data(file_path))
        self.Plot_Data.plot_data = data
        self.Plot_Data.finfo = self.finfo
        self.Multi_Select.choices = mheader
        self.Change_Axis.headers = mheader
        self.Plot_Data.change_plot(self.Plot_Data.plot_indices)

    def _Close_Overlay_fired(self):
        if self.Single_Select_Overlay_Files.selected:
            index = self.Single_Select_Overlay_Files.choices.index(self.Single_Select_Overlay_Files.selected)
            self.Single_Select_Overlay_Files.choices.remove(self.Single_Select_Overlay_Files.selected)
            del self.Plot_Data.overlay_plot_data[self.Single_Select_Overlay_Files.selected]
            if not self.Single_Select_Overlay_Files.choices:
                self.Single_Select_Overlay_Files.selected = ""
            else:
                if index >= len(self.Single_Select_Overlay_Files.choices):
                    index = len(self.Single_Select_Overlay_Files.choices) - 1
                self.Single_Select_Overlay_Files.selected = self.Single_Select_Overlay_Files.choices[index]
            self.Plot_Data.change_plot(self.Plot_Data.plot_indices)

    def _Load_Overlay_fired(self):
        dialog = FileDialog(action="open")
        dialog.open()
        info = {}
        if dialog.return_code == OK:
            for eachfile in dialog.paths:
                try:
                    overlay_data = pu.read_data(eachfile)
                    overlay_headers = pu.get_header(eachfile)
                except:
                    print "Error reading overlay data in file " + eachfile
                fnam = os.path.basename(eachfile)
                self.Plot_Data.overlay_plot_data[fnam] = overlay_data
                self.Plot_Data.overlay_headers[fnam] = overlay_headers
                self.Single_Select_Overlay_Files.choices.append(fnam)
                continue
            self.Plot_Data.change_plot(self.Plot_Data.plot_indices)
        return


def create_Viz_ModelPlot(window_name, handler=None, metadata=None, **kwargs):
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


    if metadata is not None:
        metadata.plot.configure_traits(view=view)
        return

    output_file = kwargs.get("output file")
    index_file = kwargs.get("index file")

    if output_file is None and index_file is None:
        pu.report_and_raise_error(
            "no output or index file given")
    elif output_file is not None and index_file is not None:
        pu.report_and_raise_error(
            "specify either an output or index file, not both")

    if index_file is not None:
        sim_index = psi.SimulationIndex(index_file=index_file)
        output_files = []
        variables = []
        idx = sim_index.get_index()
        for run,info in idx.iteritems():
            output_files.append(info['outfile'])
            s = []
            for var, val in info['variables'].iteritems():
                s.append("%s=%.2g" % (var, val))
                continue
            s = ", ".join(s)
            variables.append(s)

        not_found = [x for x in output_files if not os.path.isfile(x)]
        if not_found:
            pu.report_and_raise_error(
                "The following output files were not found:\n{0}"
                .format(", ".join(not_found)))

    elif output_file is not None:
        output_files = [output_file]
        variables = [""]

    view = View(HSplit(
                       VGroup(
                             Item('Multi_Select', show_label=False),
                             Item('Change_Axis',
                                  show_label=False),
                             Item('Reset_Zoom',
                                  show_label = False),
                             Item('Reload_Data',
                                  show_label = False),
                             VGroup(
                                    HGroup(
                                          Item('Load_Overlay',
                                               show_label = False, springy=True),
                                          Item('Close_Overlay',
                                               show_label = False, springy=True),
                                          ),
                                    Item('Single_Select_Overlay_Files', show_label=False, resizable=False),
                                    show_border=True)
                             ),
                       Item('Plot_Data',
                            show_label=False, width=800, height=768, springy=True, resizable=True)
                      ),
                style='custom',
                width=1124,
                height=868,
                resizable=True,
                title=window_name)

    main_window = Viz_ModelPlot(file_paths=output_files, file_variables=variables)

    main_window.configure_traits(view=view, handler=handler)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation_name")
    parser.add_argument("file")

    args = parser.parse_args()

    kwargs = {"output file": args.file}
    create_Viz_ModelPlot(args.simulation_name, **kwargs)
