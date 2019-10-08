import tables
import matplotlib.pyplot as plt


def main():
    with tables.open_file("result.hdf5") as h5file:
        data = h5file.get_node("/sim_0", "data").read()
        plt.plot(data["time"]/1000, data["Ex"])
        plt.xlabel("Microsecond")
        plt.ylabel("Volt/meter")
        plt.show()

if __name__ == '__main__':
    main()