import numpy as np
import tables


def save_result(filename, signal_electric, signal_magnetic, time, index=0):
    data = np.core.records.fromarrays([
        time[:-1],
        signal_electric[:,0],
        signal_electric[:,1],
        signal_electric[:,2],
        signal_magnetic[:,0],
        signal_magnetic[:,1],
        signal_magnetic[:,2],
    ], names="time,Ex,Ey,Ez,Hx,Hy,Hz")
    with tables.open_file(filename, "a") as h5file:
        group = h5file.create_group("/", "sim_{}".format(index))
        table = h5file.create_table(group, "data", obj=data
                                    , filters=tables.Filters(complevel=3, fletcher32=True))
        table.flush()
    return filename