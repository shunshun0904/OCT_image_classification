import numpy as np
import os
from PIL import Image
import glob
import re
from augmentor import Augmentation
from data_resize import Tomography


Augmentation().run()

# 縮小前の画像があるdir
FROM_DIR = "./train/output"


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


class Make_train():
    def image_list(self):
        test = np.empty((0, 224, 224, 1), int)
        for infile in sorted(glob.glob(os.path.join(FROM_DIR, "*.png")),key=numericalSort):
            im = Image.open(infile)
            gray_im = im.convert('L')
            gray_array = np.array(gray_im)
            im_reshape = np.reshape(gray_array, (1,224,224,1))
            test = np.concatenate([test,im_reshape])
            if len(test)==2:
                break
        normal_label = np.ones(len(test), dtype="int32")

        return test, normal_label

if __name__ == '__main__':
    print("run augmentor.py")
    x_train_tom ,y_train = Make_train().image_list()
    _, x_test_tom, _, y_test = Tomography().make()

    np.save('./x_train_augmentor.npy',x_train_tom)
    np.save('./ae_x_test.npy',x_test_tom)
    np.save('./y_train_augmentor.npy',y_train)
    np.save('./ae_y_test.npy',y_test)