import pandas as pd
import numpy as np

data = pd.read_csv('fer2013.csv')

data = data['pixels']

#remove spaces
data = [ dat.split() for dat in data]
#convert to numpy array with data type as float
data = np.array(data)
data = data.astype('float64')

#normalize data
data = [[np.divide(d,255.0) for d in dat] for dat in data]

np.save('Scaled.bin.npy',data)