import os
import glob                 # import for file looping
from scipy import misc      # import for image reading
import numpy as np          # import because you're a data scientist
import urllib               # urllib used for downloading 
import sys
import hashlib              # used for md5 checking


# In[106]:

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


# In[111]:

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
files_guess = glob.glob("unknown/*.png")


# In[95]:

print "There are "+str(len(files_yes))+" yes images"
print "There are "+str(len(files_no))+" no images"
print "There are "+str(len(files_guess))+" unknown images that we will predict"


# In[96]:

X_yes = smash_stack(files_yes)
X_no = smash_stack(files_no)
X_unknown = smash_stack(files_guess)


# In[97]:

X = np.vstack((X_yes,X_no))                                              #features for training
y = np.vstack((np.ones((X_yes.shape[0],1)),np.zeros((X_no.shape[0],1)))) #labels for training
y = y.reshape(-1,)

# In[104]:
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedKFold
import xgboost as xgb
from sklearn.metrics import roc_auc_score
def xgb_features(X,y,Xtest,params=None,random_state=0,n_folds=4,early_stop=100):
	try:		
		if params['objective'] == 'reg:logistic':
			yt = MinMaxScaler().fit_transform(y*1.)		
		else:
			yt = y
		skf = StratifiedKFold(yt, n_folds=n_folds,shuffle=True,random_state=random_state)
		ypred_test = np.zeros(Xtest.shape[0])
		ypred_train =np.zeros(X.shape[0])
		seed = random_state;
		dtest = xgb.DMatrix(data=Xtest)
		for train_index, test_index in skf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = yt[train_index], yt[test_index]			
			dtrain = xgb.DMatrix(data=X_train,label=y_train)
			dvalid = xgb.DMatrix(data=X_test,label=y_test)
			evallist = [(dtrain,'train'),(dvalid,'valid')]
			num_round = 5000
			params['seed'] = seed+1
			seed+=1
			plst = params.items()
			bst = xgb.train( plst, dtrain, num_round,evallist,early_stopping_rounds=early_stop)
			ypred = bst.predict(dtest,ntree_limit=bst.best_iteration)
			ypred_valid = bst.predict(dvalid)
			print ("\tcross validation gini score %s: %f"%(params['objective'],roc_auc_score(y_test,ypred_valid)))
			ypred_test += ypred
			ypred_train[test_index] = ypred_valid
	except KeyboardInterrupt:
		#ypred_test = np.zeros(Xtest.shape[0]);
		ypred_train = np.zeros(X.shape[0]);
		return ypred_train, ypred_test		
	return ypred_train, ypred_test*1./n_folds

def sklearn_features(X,y,Xtest,model,random_state=0,n_folds=4):
	try:
		print (model)
		skf = StratifiedKFold(y, n_folds=n_folds,shuffle=True,random_state=random_state)
		ypred_test = np.zeros(Xtest.shape[0]);
		ypred_train = np.zeros(X.shape[0]);
		seed = random_state;
		for train_index, test_index in skf:
			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]
			model.random_state=seed
			seed+=1
			model.fit(X_train,y_train)
			ypred = model.predict(Xtest)
			ypred_valid = model.predict(X_test)
			print ("\tcross validation gini score: %f"%roc_auc_score(y_test,ypred_valid))
			ypred_test = ypred if ypred_test is None else ypred_test + ypred
			ypred_train[test_index] = ypred_valid
	except KeyboardInterrupt:
		ypred_test = np.zeros(Xtest.shape[0]);
		ypred_train = np.zeros(X.shape[0]);
		return ypred_train, ypred_test
	return ypred_train, ypred_test*1./n_folds

params = {}
params["objective"] = 'binary:logistic'
params["eta"] = 0.008	# v1 0.015
params["min_child_weight"] = 6	# v1 5
params["subsample"] = 0.9#
params["colsample_bytree"]=0.666
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 12
params['eval_metric']='auc'
rtrain_xgb,rtest_xgb = xgb_features(X,y,X_unknown,params=params,n_folds=7,early_stop=150,random_state=665)#random_state v1 42 v2 983
import cPickle
fi = open('xgb_res3.pkl','wb')
cPickle.dump((rtrain_xgb,rtest_xgb),fi)
fi.close()
