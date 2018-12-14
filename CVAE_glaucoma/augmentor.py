import numpy as np
from PIL import Image
import Augmentor
from data_resize import Tomography



x_train_tom,x_test_tom,y_tran,y_test = Tomography().make()
print("run data_resize.py")


image_size = x_train_tom.shape[1]
img_rows,img_cols = 224,224
for i in range(len(x_train_tom)):
    image = x_train_tom[i:i+1,:,:]
    image = np.reshape(image, (224,224))
    pilImg = Image.fromarray(np.uint8(image))
    pilImg.save('./train/normal_%d.png' %i)



class Augmentation():
    def run(self):
        images_dir = "./train"
        p = Augmentor.Pipeline(images_dir)
        p.rotate(probability=0.8, max_left_rotation=15, max_right_rotation=15)
        p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=5)
        p.shear(probability=0.8, max_shear_left=3, max_shear_right=3)
        p.zoom(probability=0.8, min_factor=0.95, max_factor=1.15)
        p.skew_corner(probability=0.8)
        #p.crop_random(probability=0.5, percentage_area=0.8)
        p.flip_left_right(probability=0.8)
        p.sample(6000)


