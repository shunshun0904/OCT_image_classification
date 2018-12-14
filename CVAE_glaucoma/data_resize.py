from PIL import Image
import glob
import re
import pandas as pd
import numpy as np
import os
from excel_preprocess import Dataframe



folder_list = [["./data/4gatu-right/Glaucoma3D_R_Tomograms",
                "./data/4gatu-right/Glaucoma3D_R_SLO",
                "./data/4gatu-right/UI_Glaucoma3D_R_NFL+GCL+IPL_DeviationMap_gray"
                ],
               ["./data/5gatu-right/Glaucoma3D_R_Tomograms",
                "./data/5gatu-right/Glaucoma3D_R_SLO",
                "./data/5gatu-right/UI_Glaucoma3D_R_NFL+GCL+IPL_DeviationMap_gray"
                ],
               ["./data/4gatu-left/Glaucoma3D_L_Tomograms",
                "./data/4gatu-left/Glaucoma3D_L_SLO",
                "./data/4gatu-left/UI_Glaucoma3D_L_NFL+GCL+IPL_DeviationMap_gray"
                ],
               ["./data/5gatu-left/Glaucoma3D_L_Tomograms",
                "./data/5gatu-left/Glaucoma3D_L_SLO",
                "./data/5gatu-left/UI_Glaucoma3D_L_NFL+GCL+IPL_DeviationMap_gray"
                ]
               ]

folder_list_2 = [["./data/4gatu-right/UI_Glaucoma3D_R_NFL+GCL+IPL_DeviationMap_gray",
                  "./data/4gatu-right/Glaucoma3D_R_Tomograms",
                  "./data/4gatu-right/Glaucoma3D_R_SLO"
                  ],
                 ["./data/5gatu-right/UI_Glaucoma3D_R_NFL+GCL+IPL_DeviationMap_gray",
                  "./data/5gatu-right/Glaucoma3D_R_Tomograms",
                  "./data/5gatu-right/Glaucoma3D_R_SLO",
                  ],
                 ["./data/4gatu-left/UI_Glaucoma3D_L_NFL+GCL+IPL_DeviationMap_gray",
                  "./data/4gatu-left/Glaucoma3D_L_Tomograms",
                  "./data/4gatu-left/Glaucoma3D_L_SLO"

                  ],
                 ["./data/5gatu-left/UI_Glaucoma3D_L_NFL+GCL+IPL_DeviationMap_gray",
                  "./data/5gatu-left/Glaucoma3D_L_Tomograms",
                  "./data/5gatu-left/Glaucoma3D_L_SLO"
                  ]
                 ]

folder_list_3 = [[
    "./data/4gatu-right/Glaucoma3D_R_SLO",
    "./data/4gatu-right/UI_Glaucoma3D_R_NFL+GCL+IPL_DeviationMap_gray",
    "./data/4gatu-right/Glaucoma3D_R_Tomograms"
],
    ["./data/5gatu-right/Glaucoma3D_R_SLO",
     "./data/5gatu-right/UI_Glaucoma3D_R_NFL+GCL+IPL_DeviationMap_gray",
     "./data/5gatu-right/Glaucoma3D_R_Tomograms"
     ],
    [
        "./data/4gatu-left/UI_Glaucoma3D_L_NFL+GCL+IPL_DeviationMap_gray",
        "./data/4gatu-left/Glaucoma3D_L_SLO",
        "./data/4gatu-left/Glaucoma3D_L_Tomograms"
    ],
    [
        "./data/5gatu-left/UI_Glaucoma3D_L_NFL+GCL+IPL_DeviationMap_gray",
        "./data/5gatu-left/Glaucoma3D_L_SLO",
        "./data/5gatu-left/Glaucoma3D_L_Tomograms"
    ]
]

normal_comp , abnormal_comp = Dataframe().choice()
print("run excel_preprocess.py")

def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def make_data(h, w, folder_list, normal_comp):
    all_image = np.empty((0, h, w), int)
    for i in folder_list:
        count = 0
        for j in i:
            if count == 0:
                df_flat = normal_comp.iloc[:, 1:2].values.flatten()
                arr = np.empty((0, h * w), int)
                file_all = []
                file = []

                for infile in sorted(glob.glob(os.path.join(j, "*.tif")), key=numericalSort):
                    for i in range(len(normal_comp)):
                        huga = df_flat[i:i + 1].astype(str).tolist()
                        if huga[0] in infile:
                            file.append(infile)
                            file_all.append(huga[0])

                file_true = file[0::2]
                test1 = np.empty((0, h, w), int)
                for infile in file_true:
                    im = Image.open(infile)
                    im_array = np.asarray(im)
                    gray_im = im.convert('L')
                    gray_resize = gray_im.resize((h, w))
                    im_array = np.asarray(gray_resize)
                    im_reshape = np.reshape(im_array, (1, h, w))
                    # test_2 = np.concatenate([im_reshape ,im_reshape])
                    # test_2 = np.concatenate([test_2 ,im_reshape])
                    test_trans = im_reshape.transpose(0, 1, 2)
                    test1 = np.concatenate([test1, test_trans])

            if count == 1:

                arr = np.empty((0, 224 * 224), int)
                file_2 = []
                file = []

                for infile in sorted(glob.glob(os.path.join(j, "*.tif")), key=numericalSort):
                    for i in range(len(file_all)):
                        if file_all[i] in infile:
                            file.append(infile)
                            file_2.append(file_all[i])
                file_true = file[0::2]
                test4 = np.empty((0, h, w), int)
                for infile in file_true:
                    im = Image.open(infile)
                    im_array = np.asarray(im)
                    gray_im = im.convert('L')
                    gray_resize = gray_im.resize((224, 224))
                    im_array = np.asarray(gray_resize)
                    im_reshape = np.reshape(im_array, (1, 224, 224))
                    # test_2 = np.concatenate([im_reshape ,im_reshape])
                    # test_2 = np.concatenate([test_2 ,im_reshape])
                    test_trans = im_reshape.transpose(0, 1, 2)
                    test4 = np.concatenate([test4, test_trans])

            if count == 2:

                arr = np.empty((0, 224 * 224), int)
                file = []

                for infile in sorted(glob.glob(os.path.join(j, "*.tif")), key=numericalSort):
                    for i in range(len(file_2)):
                        if file_2[i] in infile:
                            file.append(infile)
                file_true = file[0::2]
                test3 = np.empty((0, h, w), int)
                for infile in file_true:
                    im = Image.open(infile)
                    im_array = np.asarray(im)
                    gray_im = im.convert('L')
                    gray_resize = gray_im.resize((224, 224))
                    im_array = np.asarray(gray_resize)
                    im_reshape = np.reshape(im_array, (1, 224, 224))
                    # test_2 = np.concatenate([im_reshape ,im_reshape])
                    # test_2 = np.concatenate([test_2 ,im_reshape])
                    test_trans = im_reshape.transpose(0, 1, 2)
                    test3 = np.concatenate([test3, test_trans])
            count += 1
        all_image = np.concatenate([all_image, test3])
    return file_true, all_image


class Tomography():

    def make(self):
        #file_normal, normal_image = make_data(224, 224, folder_list, normal_comp)
        #file_abnormal, abnormal_image = make_data(224, 224, folder_list, abnormal_comp)
        #file_normal_2, normal_image_2 = make_data(224, 224, folder_list_2, normal_comp)
        #file_abnormal_2, abnormal_image_2 = make_data(224, 224, folder_list_2, abnormal_comp)
        file_normal_3, normal_image_3 = make_data(224, 224, folder_list_3, normal_comp)
        file_abnormal_3, abnormal_image_3 = make_data(224, 224, folder_list_3, abnormal_comp)

        normal_label = np.ones(len(normal_image), dtype="int32")
        abnormal_label = np.zeros(len(abnormal_image), dtype="int32")

        y = np.concatenate([normal_label, abnormal_label])
        #x_map = np.concatenate([normal_image, abnormal_image])
        #x_slo = np.concatenate([normal_image_2, abnormal_image_2])
        x_tom = np.concatenate([normal_image_3, abnormal_image_3])


        ae_x_train = x_tom[:453,:,:]
        ae_x_test = x_tom[453:,:,:]
        ae_y_train = y[:453,]
        ae_y_test =y[453:,]

        return ae_x_train, ae_x_test ,ae_y_train ,ae_y_test

