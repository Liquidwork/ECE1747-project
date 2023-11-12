import pickle
import zlib

def compress(X, y, params):
    # Combine X, y, and params into a dictionary
    data = {'X': X, 'y': y, 'params': params}

    serialData = pickle.dumps(data)
    compresData = zlib.compress(serialData)

    return compresData

def decompress(data):
    decompresData = zlib.decompress(data)
    deserialData = pickle.loads(decompresData)

    X = deserialData['X']
    y = deserialData['y']
    params = deserialData['params']

    return X, y, params