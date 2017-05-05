import lmdb
import re, fileinput, math
import numpy as np
import random

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
sys.path.append("/home/richardb/Work/caffe/python")
import caffe

# Command line to check created files:
# python -mlmdb stat --env=./Downloads/caffe-master/data/liris-accede/train_score_lmdb/

data = 'face_only.txt'
lmdb_data_name = 'train_data_lmdb'
lmdb_label_name = 'train_score_lmdb'

Inputs = []
Label1 = []
Label2 = []
Label3 = []

for line in fileinput.input(data):
	entries = re.split(' ', line.strip())
	Inputs.append(entries[0])
	Label1.append(entries[1])
	Label2.append(entries[2])
	Label3.append(entries[3])



print('Writing labels')

# Size of buffer: 1000 elements to reduce memory consumption
for idx in range(int(math.ceil(len(Label2)/1000.0))):
	in_db_label = lmdb.open(lmdb_label_name, map_size=int(1e12))
	with in_db_label.begin(write=True) as in_txn:
		for label_idx, label1_ in enumerate(Label1[(1000*idx):(1000*(idx+1))]):

			label2_ = Label2[label_idx]
			label3_ = Label3[label_idx]

			label_ = np.zeros((1,3))
			label_[0] = label1_
			label_[1] = label2_
			label_[2] = label3_

			im_dat = caffe.io.array_to_datum(np.array(label_).astype(float).reshape(1,1,3))
			in_txn.put('{:0>10d}'.format(1000*idx + label_idx), im_dat.SerializeToString())
			print(im_dat)
			string_ = str(1000*idx+label_idx+1) + ' / ' + str(len(Label2))
			sys.stdout.write("\r%s" % string_)
			sys.stdout.flush()
	in_db_label.close()
print('')

print('Writing image data')

for idx in range(int(math.ceil(len(Inputs)/1000.0))):
	in_db_data = lmdb.open(lmdb_data_name, map_size=int(1e12))
	with in_db_data.begin(write=True) as in_txn:
		for in_idx, in_ in enumerate(Inputs[(1000*idx):(1000*(idx+1))]):
			im = caffe.io.load_image(in_)
			im_dat = caffe.io.array_to_datum(im.astype(float).transpose((2, 0, 1)))
			in_txn.put('{:0>10d}'.format(1000*idx + in_idx), im_dat.SerializeToString())

			string_ = str(1000*idx+in_idx+1) + ' / ' + str(len(Inputs))
			sys.stdout.write("\r%s" % string_)
			sys.stdout.flush()
	in_db_data.close()
print('')
