import h5py


class Convertor:

    def __init__(self):
        print("Convertor object created")

    def read_matlab(self,filename):
        def conv(path=''):
            p = path or '/'
            paths[p] = ret = {}
            for k, v in f[p].items():
                if type(v).__name__ == 'Group':
                    ret[k] = conv(f'{path}/{k}')  # Nested struct
                    continue
                v = v[()]  # It's a Numpy array now
                if v.dtype == 'object':
                    # HDF5ObjectReferences are converted into a list of actual pointers
                    ret[k] = [r and paths.get(f[r].name, f[r].name) for r in v.flat]
                else:
                    # Matrices and other numeric arrays
                    ret[k] = v if v.ndim < 2 else v.swapaxes(-1, -2)
            return ret

        paths = {}
        with h5py.File(filename, 'r') as f:
            return conv()