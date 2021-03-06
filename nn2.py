import cPickle as pickle
from datetime import datetime
import os
import sys

from matplotlib import pyplot
from lasagne import layers
from nolearn.lasagne import BatchIterator,NeuralNet,TrainSplit
from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import theano
import glob				 # import for file looping
from scipy import misc	  # import for image reading
import numpy as np		  # import because you're a data scientist
import urllib			   # urllib used for downloading 
import hashlib			  # used for md5 checking
from lasagne.nonlinearities import *

try:
	from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
	from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
	Conv2DLayer = layers.Conv2DLayer
	MaxPool2DLayer = layers.MaxPool2DLayer

sys.setrecursionlimit(10000)  # for pickle...
np.random.seed(192)

def image_smash(filename):   # Taken an image, flatten it, and smash it
	# This function takes an image filename, loads it into a matrix, smashes it to 1D row
	data=misc.imread(filename,flatten=True)
	data_row = np.ravel(data)
	return data_row


# In[107]:

def smash_stack(file_list):
	# This function initializes the file matrix and begins populating it row by row
	structured_image_matrix = np.zeros((len(file_list),len(image_smash(file_list[0]))))   #initialize matrix
	for row in range(0,len(file_list)):
		structured_image_matrix[row,:]=image_smash(file_list[row])
	return structured_image_matrix


# In[108]:

def dlProgress(count, blockSize, totalSize):
	percent = int(count*blockSize*100/totalSize)
	sys.stdout.write("\r mars_comp.tar.gz:%d%% complete" % percent)
	sys.stdout.flush()


# In[109]:

def file_check(filename):
	error=1
	if hashlib.md5(filename).hexdigest() == "03098c7ef6a012cc569ffe0166716312":
		error=0
	return error

def float32(k):
	return np.cast['float32'](k)

def load():
	if os.path.exists('mars_comp.tar.gz'):
		print "local file exists"
		#check local file?
		error = file_check("mars_comp.tar.gz")
	else:
		print "Downloading dataset"
		data_location = "https://s3.amazonaws.com/global-comp-mars-set/final_set/mars_comp.tar.gz"
		urllib.urlretrieve(data_location, "mars_comp.tar.gz",reporthook=dlProgress)
		print "Download Complete!"
		error = file_check("mars_comp.tar.gz")	
	if error==0:
		print "MD5 check passed"
	# In[93]:
	if os.path.exists('volyes_train/') and os.path.exists('volno_train/') and os.path.exists('unknown/'):
		print 'use existing files'
	else:
		from subprocess import call
		call(["tar","xf","mars_comp.tar.gz"])   #This extract the files to your local folder
	# In[94]:
	files_yes = glob.glob("volyes_train/*.png")
	files_no = glob.glob("volno_train/*.png")
	files_guess = sorted(glob.glob("unknown/*.png"))
	# In[95]:
	print "There are "+str(len(files_yes))+" yes images"
	print "There are "+str(len(files_no))+" no images"
	print "There are "+str(len(files_guess))+" unknown images that we will predict"
	# In[96]:
	X_yes = smash_stack(files_yes)
	X_no = smash_stack(files_no)
	X_unknown = smash_stack(files_guess)
	# In[97]:
	X = np.vstack((X_yes,X_no))											  #features for training
	y = np.vstack((np.ones((X_yes.shape[0],1)),np.zeros((X_no.shape[0],1)))) #labels for training
	y = y.reshape(-1,)
	X,y = shuffle(X,y,random_state=911)
	X = X/255.
	X_unknown = X_unknown/255.
	return X,y,X_unknown

def load2d():
	X, y, Xtest= load()
	X = X.reshape(-1, 1, 91, 91)
	Xtest = Xtest.reshape(-1,1,91,91)
	return X, y, Xtest

def plot_sample(x, y, axis):
	img = x.reshape(91, 91)
	axis.imshow(img, cmap='gray')
	if y is not None:
		axis.scatter(y[0::2] * 45 + 45, y[1::2] * 45 + 45, marker='x', s=10)


def plot_weights(weights):
	fig = pyplot.figure(figsize=(6, 6))
	fig.subplots_adjust(
		left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

	for i in range(16):
		ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
		ax.imshow(weights[:, i].reshape(96, 96), cmap='gray')
	pyplot.show()

class FlipBatchIterator(BatchIterator):
	def transform(self, Xb, yb):
		Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
		# Flip half of the images in this batch at random:
		bs = Xb.shape[0]
		indices = np.random.choice(bs, bs / 3, replace=False)
		Xb1 = Xb[indices, :, :, ::-1]
		yb1 = yb[indices]
		indices2 = np.random.choice(bs, bs/3, replace=False)
		Xb2 = Xb[indices2,:,::-1,:]
		yb2 = yb[indices2]
		return np.vstack((Xb,Xb1,Xb2)), np.hstack((yb,yb1,yb2))

class AdjustVariable(object):
	def __init__(self, name, start=0.03, stop=0.001):
		self.name = name
		self.start, self.stop = start, stop
		self.ls = None
	def __call__(self, nn, train_history):
		if self.ls is None:
			self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
		epoch = train_history[-1]['epoch']
		new_value = np.cast['float32'](self.ls[epoch - 1])
		getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
	def __init__(self, patience=100):
		self.patience = patience
		self.best_valid = np.inf
		self.best_valid_epoch = 0
		self.best_weights = None
	def __call__(self, nn, train_history):
		current_valid = train_history[-1]['valid_loss']
		current_epoch = train_history[-1]['epoch']
		if current_valid < self.best_valid:
			self.best_valid = current_valid
			self.best_valid_epoch = current_epoch
			self.best_weights = nn.get_all_params_values()
		elif self.best_valid_epoch + self.patience < current_epoch:
			print("Early stopping.")
			print("Best valid loss was {:.6f} at epoch {}.".format(
				self.best_valid, self.best_valid_epoch))
			nn.load_params_from(self.best_weights)
			raise StopIteration()

net = NeuralNet(
	layers=[
		('input', layers.InputLayer),
		('conv1', Conv2DLayer),
		('pool1', MaxPool2DLayer),
		('dropout1', layers.DropoutLayer),
		('conv2', Conv2DLayer),
		('pool2', MaxPool2DLayer),
		('dropout2', layers.DropoutLayer),
		('conv3', Conv2DLayer),
		('pool3', MaxPool2DLayer),
		('dropout3', layers.DropoutLayer),
		('hidden4', layers.DenseLayer),
		('dropout4', layers.DropoutLayer),
		('hidden5', layers.DenseLayer),
		('dropout5',layers.DropoutLayer),
		('output', layers.DenseLayer),
		],
	input_shape=(None, 1, 91, 91),
	conv1_num_filters=10, conv1_filter_size=(2, 2), pool1_pool_size=(2, 2),
	dropout1_p=0.2,
	conv2_num_filters=20, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
	dropout2_p=0.3,
	conv3_num_filters=40, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
	dropout3_p=0.35,
	hidden4_num_units=200,
	hidden4_nonlinearity=sigmoid,
	dropout4_p=0.5,
	hidden5_num_units=110,
	dropout5_p=0.4,
	output_num_units=2, output_nonlinearity=softmax,

	update_learning_rate=theano.shared(float32(0.025)),
	update_momentum=theano.shared(float32(0.9)),

	#regression=True,
	batch_iterator_train=FlipBatchIterator(batch_size=256),
	on_epoch_finished=[
		AdjustVariable('update_learning_rate', start=0.025, stop=0.0001),
		AdjustVariable('update_momentum', start=0.9, stop=0.99),
		EarlyStopping(patience=80),
		],
	max_epochs=1500,
	verbose=1,
	train_split=TrainSplit(eval_size=0.2),
	)
if __name__ == '__main__':
	X, y , Xtest= load2d()  # load 2-d data
	net.fit(X.astype(theano.config.floatX), y.astype(np.int32).reshape(-1,))
	with open('net2.pickle', 'wb') as f:
		pickle.dump(net, f, -1)
	ypred = net.predict(Xtest.astype(theano.config.floatX))
	with open('net2.res.pkl', 'wb') as f:
		pickle.dump(ypred,f)
