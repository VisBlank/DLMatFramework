import scipy.misc
import random
import h5py
import numpy as np


class HandleData:
    __xs = []
    __ys = []
    __file = []
    __file_val = []
    __dataset_imgs = []
    __dataset_label = []
    __dataset_imgs_val = []
    __dataset_label_val = []
    __num_images = 0
    __train_xs = []
    __train_ys = []
    __val_xs = []
    __val_ys = []
    __num_train_images = 0
    __num_val_images = 0
    __train_batch_pointer = 0
    __val_batch_pointer = 0
    __train_perc = 0
    __val_perc = 0

    # points to the end of the last batch
    __train_batch_pointer = 0
    __val_batch_pointer = 0

    def __init__(self, path='DrivingData.h5', path_val='', train_perc=0.8, val_perc=0.2, shuffle=True):
        self.__train_perc = train_perc
        self.__val_perc = val_perc
        print("Loading training data")
        # Read hdf5
        self.__file = h5py.File(path, 'a')
        # Check if the dataset exist
        existTrain = "/Train/Labels" in self.__file
        if existTrain:
            # Initialize pre-existing datasets
            self.__dataset_imgs = self.__file["/Train/Images"]
            self.__dataset_label = self.__file["/Train/Labels"]

            self.__xs = list(self.__dataset_imgs)
            self.__ys = list(self.__dataset_label)

            self.__num_images = len(self.__xs)


            # Create a zip list with images and angles
            c = list(zip(self.__xs, self.__ys))

            # Shuffle data
            if shuffle == True:
                random.shuffle(c)

            # Check if validation set is not given
            if not path_val:
                self.__xs, self.__ys = zip(*c)

                # Training set 80%
                self.__train_xs = self.__xs[:int(len(self.__xs) * train_perc)]
                self.__train_ys = self.__ys[:int(len(self.__xs) * train_perc)]

                # Validation set 20%
                self.__val_xs = self.__xs[-int(len(self.__xs) * val_perc):]
                self.__val_ys = self.__ys[-int(len(self.__xs) * val_perc):]
            else:
                print('Load validation dataset')
                # Read hdf5
                self.__file_val = h5py.File(path_val, 'a')
                # Check if the dataset exist
                existTrain = "/Train/Labels" in self.__file_val
                if existTrain:
                    # Training set 100%
                    self.__train_xs = self.__xs
                    self.__train_ys = self.__ys

                    # Initialize pre-existing datasets
                    self.__dataset_imgs_val = self.__file_val["/Train/Images"]
                    self.__dataset_label_val = self.__file_val["/Train/Labels"]

                    self.__val_xs = list(self.__dataset_imgs_val)
                    self.__val_ys = list(self.__dataset_label_val)

            # Get number of images
            self.__num_train_images = len(self.__train_xs)
            self.__num_val_images = len(self.__val_xs)
            print("Number training images: %d" % self.__num_train_images)
            print("Number validation images: %d" % self.__num_val_images)

    def shuffleData(self):
        # Shuffle data
        c = list(zip(self.__xs, self.__ys))
        random.shuffle(c)
        self.__xs, self.__ys = zip(*c)

        # Training set 80%
        self.__train_xs = self.__xs[:int(len(self.__xs) * self.__train_perc)]
        self.__train_ys = self.__ys[:int(len(self.__xs) * self.__train_perc)]

        # Validation set 20%
        self.__val_xs = self.__xs[-int(len(self.__xs) * self.__val_perc):]
        self.__val_ys = self.__ys[-int(len(self.__xs) * self.__val_perc):]

    def LoadTrainBatch(self, batch_size, crop_up=0):
        x_out = []
        y_out = []

        # If batch_size is -1 load the whole thing
        if batch_size == -1:
            batch_size = self.__num_train_images

        for i in range(0, batch_size):
            # Load image
            # image = scipy.misc.imread(train_xs[(train_batch_pointer + i) % num_train_images], mode="RGB")
            image = self.__train_xs[(self.__train_batch_pointer + i) % self.__num_train_images]
            # Crop top, resize to 66x200 and divide by 255.0
            image = scipy.misc.imresize(image[-crop_up:], [66, 200]) / 255.0
            x_out.append(image)
            y_out.append([self.__train_ys[(self.__train_batch_pointer + i) % self.__num_train_images]])
            self.__train_batch_pointer += batch_size
        return x_out, y_out

    def LoadValBatch(self, batch_size, crop_up=0):
        x_out = []
        y_out = []

        # If batch_size is -1 load the whole thing
        if batch_size == -1:
            batch_size = self.__num_val_images

        for i in range(0, batch_size):
            # Load image
            # image = scipy.misc.imread(val_xs[(val_batch_pointer + i) % num_val_images], mode="RGB")
            image = self.__val_xs[(self.__val_batch_pointer + i) % self.__num_val_images]
            # Crop top, resize to 66x200 and divide by 255.0
            image = scipy.misc.imresize(image[-crop_up:], [66, 200]) / 255.0
            x_out.append(image)
            y_out.append([self.__val_ys[(self.__val_batch_pointer + i) % self.__num_val_images]])
            self.__val_batch_pointer += batch_size
        return x_out, y_out

    def get_num_images(self):
        return self.__num_images

    def save_hdf5(self, list_tups_train, filename='newHdf5.h5'):
        ys = []
        imgs = []
        for (tup_element) in list_tups_train:
            img, steer = tup_element
            ys.append(steer)
            imgs.append(img)

        # Convert to numpy arrays
        list_labels_ndarray = np.asarray(ys)
        list_imgs_ndarray = np.asarray(imgs)
        # Should have shape 1111,
        list_labels_ndarray = list_labels_ndarray.reshape(list_labels_ndarray.size)

        file = h5py.File(filename, 'w')  # 'a' is append
        # Create Training group
        group = file.create_group("Train")
        # Dataset must be resizable and chunked
        dataset_label = group.create_dataset("Labels", list_labels_ndarray.shape, maxshape=(None,), chunks=True)
        dataset_imgs = group.create_dataset("Images", list_imgs_ndarray.shape, maxshape=(None, 256, 256, 3),
                                            chunks=True)

        # Copy data
        dataset_label[...] = list_labels_ndarray
        dataset_imgs[...] = list_imgs_ndarray

        # Close and save data
        file.flush()
        file.close()