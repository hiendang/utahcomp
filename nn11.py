import cPickle as pickle
from datetime import datetime
import os
import sys

from matplotlib import pyplot

from nolearn.lasagne import BatchIterator,NeuralNet
try:
	from nolearn.lasagne import TrainSplit
except:
	print 'cannot import TrainSplit'
from pandas import DataFrame
from pandas.io.parsers import read_csv
import theano
import glob				 # import for file looping
from scipy import misc	  # import for image reading
import numpy as np		  # import because you're a data scientist
import urllib			   # urllib used for downloading 
import hashlib			  # used for md5 checking

from lasagne.nonlinearities import *
from lasagne import layers

from sklearn.metrics import roc_auc_score,log_loss,accuracy_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.utils import shuffle

try:
	from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
	from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
	Conv2DLayer = layers.Conv2DLayer
	MaxPool2DLayer = layers.MaxPool2DLayer

sys.setrecursionlimit(10000)  # for pickle...
from rotation import *
flip_func = [array_tf_0f,array_tf_90f,array_tf_180f,array_tf_270f]
rotate_func = [array_tf_0,array_tf_90,array_tf_180,array_tf_270]
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

def load2d(rotate_function=None):
	X, y, Xtest= load()
	X = X.reshape(-1, 1, 91, 91)
	Xtest = Xtest.reshape(-1,1,91,91)
	if rotate_function is not None:
		X = rotate_function(X)
		Xteste = rotate_function(Xtest)
	return X.astype(theano.config.floatX), y.reshape(-1,).astype(np.int32), Xtest.astype(theano.config.floatX)

class FlipBatchIterator(BatchIterator):	
	def transform(self, Xb, yb):
		Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)
		# Flip half of the images in this batch at random:
		flip_func_id = np.random.randint(len(flip_func))
		#Xb = flip_func[flip_func_id](Xb)
		bs = Xb.shape[0]		
		indices =np.random.choice(bs,bs/2,replace=False)
		Xb[indices] = flip_func[flip_func_id](Xb[indices, :, :, :])
		#indices =np.random.choice(bs,bs/3,replace=False)
		#Xb[indices] = Xb[indices, :, ::-1, :]
		return Xb,yb

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
	def __init__(self, patience=100,Xvalid=None,yvalid=None,verbose=True):
		self.patience = patience
		self.best_valid = 0
		self.best_valid_epoch = 0
		self.best_weights = None
		self.Xvalid = Xvalid
		self.yvalid = yvalid
		self.verbose=verbose
	def __call__(self, nn, train_history):
		ypred_valid = nn.predict_proba(self.Xvalid)
		current_valid = roc_auc_score(self.yvalid,ypred_valid[:,1])
		current_epoch = train_history[-1]['epoch']
		valid_log_loss = log_loss(self.yvalid,ypred_valid)
		valid_acc = accuracy_score(self.yvalid, np.argmax(ypred_valid,axis=1))
		if self.verbose:			
			print "%04d   |   %5f   |   %5f   |   %5f   |  %5f  | %4.2f"%(current_epoch,train_history[-1]['train_loss'],valid_log_loss,valid_acc,current_valid,train_history[-1]['dur'])
		if current_valid > self.best_valid:
			self.best_valid = current_valid
			self.best_valid_epoch = current_epoch
			self.best_weights = nn.get_all_params_values()
		elif self.best_valid_epoch + self.patience < current_epoch:
			print("Early stopping.")
			print("Best valid loss was {:.6f} at epoch {}.".format(
				self.best_valid, self.best_valid_epoch))
			nn.load_params_from(self.best_weights)
			raise StopIteration()


def build_nn4():
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
		conv1_num_filters=4, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
		dropout1_p=0.25,
		conv2_num_filters=8, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
		dropout2_p=0.3,
		conv3_num_filters=16, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
		dropout3_p=0.35,
		hidden4_num_units=210,
		hidden4_nonlinearity=leaky_rectify,
		dropout4_p=0.5,
		hidden5_num_units=130,
		dropout5_p=0.35,
		output_num_units=2, output_nonlinearity=softmax,

		update_learning_rate=theano.shared(float32(0.02)),
		update_momentum=theano.shared(float32(0.9)),

		#regression=True,
		batch_iterator_train=FlipBatchIterator(batch_size=128),
		on_epoch_finished=[
			AdjustVariable('update_learning_rate', start=0.02, stop=0.00000001),
			AdjustVariable('update_momentum', start=0.9, stop=0.99),			
			],
		max_epochs=1000,
		verbose=0,
		train_split=TrainSplit(eval_size=0.0),
		)
	return net

def nn_features(X,y,Xtest,model=build_nn4,random_state=100,n_folds=4):
	from lasagne.layers import noise
	from theano.sandbox.rng_mrg import MRG_RandomStreams
	noise._srng =MRG_RandomStreams(seed=random_state)
	try:		
		skf = StratifiedKFold(y, n_folds=n_folds,shuffle=True,random_state=random_state)
		ypred_test = None;
		ypred_train = np.zeros(X.shape[0],);		
		for train_index, test_index in skf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]			
			y_train = y_train.reshape(-1,)
			nn = model()			
			print "%-7s|  %-12s|  %-12s|  %-12s| %-9s |  %-4s"%('epoch','train loss','valid loss','accuracy','roc ','dur')
			nn.on_epoch_finished.append(EarlyStopping(patience=150,Xvalid=X_test,yvalid=y_test,verbose=True))
			nn.fit(X_train,y_train)
			ypred = nn.predict_proba(Xtest)[:,1]
			ypred_valid = nn.predict_proba(X_test)[:,1]
			ypred_test = ypred if ypred_test is None else ypred_test + ypred
			ypred_train[test_index] = ypred_valid
	except KeyboardInterrupt:		
		return ypred_train, ypred_test
	return ypred_train, ypred_test*1./n_folds

if __name__ == '__main__':
	random_state=1
	if len(sys.argv) < 2:
		print '[LOG] no random_seed is given -> use 1'
	else:
		random_state=int(sys.argv[1])
		print '[LOG] using %d as random seed'%random_state
	np.random.seed(random_state)
	X, y , Xtest= load2d()
	rtrain_total = None
	rtest_total = None
	for i in range(4):
		X = array_tf_90(X)
		Xtest = array_tf_90(Xtest)
		rtrain,rtest = nn_features(X,y,Xtest,model=build_nn4,random_state=np.random.randint(10000),n_folds=5)
		rfile = 'net11.res.rotated.%d.pkl'%i
		with open(rfile,'wb') as f:
			pickle.dump((rtrain,rtest), f)
		if i == 0:
			rtrain_total = rtrain
			rtest_total = rtest
		else:
			rtrain_total += rtrain
			rtest_total += rtest
	
	print 'roc auc score is %f '%(roc_auc_score(y,rtrain))
	with open('net11.res.pickle', 'wb') as f:
		pickle.dump((rtrain_total,rtest_total), f)
	
