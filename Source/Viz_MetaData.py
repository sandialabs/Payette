import uuid

from enthought.traits.api import HasStrictTraits, String, List, File, Date, Time, Bool, Enum, Instance, Directory, Tuple

from Viz_ModelData import PayetteModel
from Viz_ModelPlot import Viz_ModelPlot 

class VizMetaData(HasStrictTraits):
    name = String
    base_directory = Directory
    index_file = File
    out_file = File
    log_file = File
    surface_file = File
    data_type = Enum('Simulation', 'Optimization', 'Visualization', 'Imported')
    model_type = String
    path_files = List(Tuple(String, File))
    created_date = Date
    created_time = Time
    object_id = String
    session_id = String
    successful = Bool(False)
    model = Instance(PayetteModel)
    plot = Instance(Viz_ModelPlot)

    def __init__(self, **traits):
        HasStrictTraits.__init__(self, **traits)

        if len(self.object_id) < 1:
            self.object_id = str(uuid.uuid1())
