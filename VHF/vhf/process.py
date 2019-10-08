import os

import numpy as np
import tables

from .lienard_wiechert import EMFieldCalculator

CALCULATOR = EMFieldCalculator()


def compute_velocity(track):  # m/ns
    diff_x = np.diff(track["x"])
    diff_y = np.diff(track["y"])
    diff_z = np.diff(track["z"])
    return ((np.array([diff_x, diff_y, diff_z])) / (np.diff(track["time"]))).T


def compute_acceleration(velocity, time):  ## m/ns^2
    velocity = velocity.T
    time = time[2:] - time[:-2]
    return (np.diff(velocity) / time).T


def process_track(track, observed_point):
    outtype = np.dtype([('x', np.double), ('y', np.double), ('z', np.double), ('time', np.double)])
    if (track.size < 3):
        return None
    velocity = compute_velocity(track)
    acceleration = compute_acceleration(velocity, track["time"])
    r0 = track[["x", "y", "z"]][1:-1]
    r0 = np.array([r0["x"], r0["y"], r0["z"]]).T
    efield, hfield = CALCULATOR.get_em_field(observed_point, r0, velocity[:-1], acceleration)
    outcome = np.zeros(track.size - 2, outtype)
    time = track["time"][1:-1] + CALCULATOR.get_retarded_time(observed_point, r0)
    return efield, hfield, time


def process_event(tracks, id_tracks, observed_point, event_number=0, verbose=False):
    tracks_position = np.hstack((np.zeros(1), np.cumsum(id_tracks["count"])))
    result = []
    for i in range(id_tracks.size):
        if verbose:
            print("Process track id:", id_tracks[i])
        left, right = int(tracks_position[i]), int(tracks_position[i + 1])
        _, mask = np.unique(tracks[left:right]['time'], return_index=True)
        temp = tracks[left:right]
        data = process_track(temp[mask], observed_point)
        result.append(data)
    result = list(filter(lambda x: x is not None, result))
    return result


def find_max_time(listData):
    max_time = 0
    for item in listData:
        temp = item[-1][-1]
        if (temp > max_time):
            max_time = temp
    return max_time


def join_track_signal(tracks, time_step=1):
    max_time = find_max_time(tracks)
    grid = np.arange(0, max_time + time_step, time_step)
    value_electric = np.zeros( (grid.size - 1, 3), dtype="d")
    value_magnetic = np.zeros( (grid.size - 1, 3), dtype="d")
    for efield, hfield, time in tracks:
        index = np.searchsorted(grid, time, side="left")
        left = index[0]
        if (index[0] < grid.size -1):
            value_electric[left] += efield[0]
            value_magnetic[left] += hfield[0]
        for i, j in enumerate(index[1:]):
            value_electric[left:j] += efield[i+1]
            value_magnetic[left:j] += hfield[i+1]
            left = j
    return value_electric, value_magnetic, grid


def join_event_signal(events, time_step=1):
    max_size = 0
    for signal, _ in events:
        temp = signal.size
        if temp > max_size:
            max_size = temp

    grid = np.arange(0, (max_size + 1) * time_step, time_step)
    value_electric = np.zeros( (grid.size - 1, 3), dtype="d")
    value_magnetic = np.zeros( (grid.size - 1, 3), dtype="d")
    for  efield, hfield in events:
        i = efield.shape[0]
        value_electric[:i] += efield
        value_magnetic[:i] += hfield
    return value_electric, value_magnetic, grid


from multiprocessing import Pool


def process_event_parallel(params):
    tracks, id_tracks, observed_point, time_step = params
    tracks = process_event(tracks, id_tracks, observed_point, verbose=False)
    return join_track_signal(tracks, time_step)


def params_generator(h5file, observed_point, time_step):
    for group in h5file.root:
        tracks = h5file.get_node(group, "tracks").read()
        id_tracks = h5file.get_node(group, "tracks_id").read()
        yield [tracks, id_tracks, observed_point, time_step]


def process_file(filename, observed_point, time_step=1):
    with tables.open_file(filename) as h5file:
        with Pool() as p:
            signals = p.imap_unordered(process_event_parallel, params_generator(h5file, observed_point, time_step))
            signal_electric, signal_magnetic, time = join_event_signal([i[:2] for i in signals], time_step)
    return signal_electric, signal_magnetic, time

def process_all_files(path, observed_point, time_step=1):
    files = os.listdir(path)
    results = []
    for file in files:
        filename = os.path.join(path, file)
        temp = process_file(filename, observed_point)
        results.append(temp)
    signal_electric, signal_magnetic, time = join_event_signal([i[:2] for i in results], time_step)
    return signal_electric, signal_magnetic, time