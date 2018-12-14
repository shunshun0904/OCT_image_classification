# OCT_image_classification

This repository is the python script file to imprement conlutional VAE.


## Usage

I describe these python files below.

1. First of all, you must make the numpy file stored image pixels from exel files and image directry.

```
python data_to_numpy.py
```

### detail about making file

- x_train_augmentor.npy: augmented images pixel data
- ae_x_test.npy: test images pixel data
- y_train_augmentor.npy: label of augmented images pixel data (1: normal ,0:abnormal)
- ae_y_test.npy: label of test images pixel data (1: normal ,0:abnormal)


2. after that, you can run C-VAE

```
python vae_CNN.py
```




