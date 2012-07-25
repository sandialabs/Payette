from enthought.traits.api import HasStrictTraits, String, List, File, Date, Time, Bool, Enum, Instance, Directory

from Viz_ModelData import PayetteModel

class VizMetaData(HasStrictTraits):
    name = String
    base_directory = Directory
    index_file = File
    out_file = File
    log_file = File
    surface_file = File
    data_type = Enum('Simulation', 'Optimization', 'Visualization', 'Imported')
    model_type = String
    curve_files = List(File)
    created_date = Date
    created_time = Time
    object_id = String
    session_id = String
    successful = Bool(False)
    model = Instance(PayetteModel)
