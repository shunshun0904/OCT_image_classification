import numpy as np
from PIL import Image
import os
import glob
import pandas as pd
import re


## remove something,for example 'PQ'...
def make_df(file, number):
    data = pd.read_excel(file, sheet_name=number, delim_whitespace=True, encoding="shift-jis")
    data_PQ = data[data['SLO'] != 'PQ']
    number = data_PQ.iloc[:, 0:1]
    data_drop_num = data_PQ.drop(columns='匿名個人番号')
    get_index = data_drop_num.iloc[:, :5].dropna(how='all')
    data = pd.concat([number, data_drop_num], axis=1)
    abnormal_index = get_index.index.tolist()
    data_abnormal = data.loc[abnormal_index]
    data_normal = data.drop(index=abnormal_index)
    return data_normal, data_abnormal


## finish making the normal and abnormal version dataframe
def connect_df():
    april_right_no, april_right_ab = make_df('./data/1804_oct.xlsx', 'right.csv')
    april_left_no, april_left_ab = make_df('./data/1804_oct.xlsx', 'left.csv')
    may_right_no, may_right_ab = make_df('./data/1805_oct.xlsx', 'right.csv')
    may_left_no, may_left_ab = make_df('./data/1805_oct.xlsx', 'left.csv')
    normal_right = pd.concat([april_right_no, may_right_no])
    abnormal_right = pd.concat([april_right_ab, may_right_ab])
    normal_left = pd.concat([april_left_no, may_left_no])
    abnormal_left = pd.concat([april_left_ab, may_left_ab])
    return normal_right, abnormal_right, normal_left, abnormal_left


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def image_pixel(file_list, img_rows, img_cols, df):
    df_flat = df.iloc[:, 0:1].values.flatten()
    arr = np.empty((0, 224 * 224), int)
    exel = []
    file_all = []
    for k in file_list:
        FROM_DIR = k
        listing = os.listdir(k)
        file = []
        for infile in sorted(glob.glob(os.path.join(FROM_DIR, "*.tif")), key=numericalSort):
            for i in range(len(df)):
                huga = df_flat[i:i + 1].astype(str).tolist()
                if huga[0] in infile:
                    file.append(infile)
                    file_all.append(huga[0])

        file_true = file[0::2]
        for infile in file_true:
            im = Image.open(infile)
            img = im.resize((img_rows, img_cols))
            im_array = np.asarray(img)
            im_reshape = np.reshape(im_array, (1, img_rows * img_cols))
            arr = np.append(arr, im_reshape, axis=0)

    pixel = pd.DataFrame(arr)
    # file_all_true = file_all[0::2]
    return pixel, file_all


class Dataframe():
    def choice(self):
        file_list_right = ["./data/4gatu-right/Glaucoma3D_R_Tomograms",
                           "./data/5gatu-right/Glaucoma3D_R_Tomograms"]
        file_list_left = ["./data/4gatu-left/Glaucoma3D_L_Tomograms",
                          "./data/5gatu-left/Glaucoma3D_L_Tomograms"]

        right_folder = [["./data/4gatu-right/Glaucoma3D_R_Tomograms",
                         "./data/5gatu-right/Glaucoma3D_R_Tomograms"]
            , ["./data/4gatu-right/Glaucoma3D_R_SLO",
               "./data/5gatu-right/Glaucoma3D_R_SLO"]
            , ["./data/4gatu-right/UI_Glaucoma3D_R_NFL+GCL+IPL_DeviationMap_gray",
               "./data/5gatu-right/UI_Glaucoma3D_R_NFL+GCL+IPL_DeviationMap_gray"]]
        left_folder = [["./data/4gatu-left/Glaucoma3D_L_Tomograms",
                        "./data/5gatu-left/Glaucoma3D_L_Tomograms"]
            , ["./data/4gatu-left/Glaucoma3D_L_SLO",
               "./data/5gatu-left/Glaucoma3D_L_SLO"]
            , ["./data/4gatu-left/UI_Glaucoma3D_L_NFL+GCL+IPL_DeviationMap_gray",
               "./data/5gatu-left/UI_Glaucoma3D_L_NFL+GCL+IPL_DeviationMap_gray"]]

        count = 0
        for i, j in zip(right_folder, left_folder):
            count += 1
            img_rows, img_cols = 224, 224
            normal_right, abnormal_right, normal_left, abnormal_left = connect_df()
            abnormal_pixel_right, abnormal_num_right = image_pixel(i, img_rows, img_cols, abnormal_right)
            abnormal_pixel_left, abnormal_num_left = image_pixel(j, img_rows, img_cols, abnormal_left)
            normal_pixel_right, normal_num_right = image_pixel(i, img_rows, img_cols, normal_right)
            normal_pixel_left, normal_num_left = image_pixel(j, img_rows, img_cols, normal_left)

            abnormal_exel_image_right = abnormal_right[abnormal_right['匿名個人番号'].isin(abnormal_num_right)]
            abnormal_exel_image_left = abnormal_left[abnormal_left['匿名個人番号'].isin(abnormal_num_left)]
            normal_exel_image_right = normal_right[normal_right['匿名個人番号'].isin(normal_num_right)]
            normal_exel_image_left = normal_left[normal_left['匿名個人番号'].isin(normal_num_left)]
            if count == 1:
                abnormal_tomo = pd.concat([abnormal_exel_image_right, abnormal_exel_image_left])
                normal_tomo = pd.concat([normal_exel_image_right, normal_exel_image_left])
            if count == 2:
                abnormal_SLO = pd.concat([abnormal_exel_image_right, abnormal_exel_image_left])
                normal_SLO = pd.concat([normal_exel_image_right, normal_exel_image_left])
                abnormal_2 = pd.merge(abnormal_tomo, abnormal_SLO, on='匿名個人番号')
                normal_2 = pd.merge(normal_tomo, normal_SLO, on='匿名個人番号')
            if count == 3:
                abnormal_gla = pd.concat([abnormal_exel_image_right, abnormal_exel_image_left])
                normal_gla = pd.concat([normal_exel_image_right, normal_exel_image_left])
                abnormal_3 = pd.merge(abnormal_2, abnormal_gla, on='匿名個人番号')
                normal_3 = pd.merge(normal_2, normal_gla, on='匿名個人番号')

        normal_comp = normal_tomo[normal_3.duplicated('匿名個人番号', keep=False)]
        abnormal_comp = abnormal_tomo[abnormal_3.duplicated('匿名個人番号', keep=False)]

        ##疾患の中でも緑内障のみ選択
        abnormal_comp_glaucoma = abnormal_comp[(abnormal_comp["oct1"] == "Gla") | (abnormal_comp["oct1"] == "gla")]


        return normal_comp, abnormal_comp_glaucoma
