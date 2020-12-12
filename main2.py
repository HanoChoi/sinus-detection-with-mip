from medpy.io import load, save
import glob
import numpy as np
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    Input,
    MaxPool2D, Dense, Flatten, BatchNormalization, Dropout
)
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tensorflow.keras import backend as K
import sys

def main():
    #id = '27565425'
    input_folder_path = './input'
    output_folder_path = './output'
    model_path = './model'
    #line_width = 4
    #is_skeletal_or_box = False
    id = sys.argv[1]
    line_width = int(sys.argv[2])
    is_skeletal_or_box = False
    if sys.argv[3]=='True':
        is_skeletal_or_box = True
    print(id, line_width, is_skeletal_or_box)

    sinus_name = ['Ethmoid','Maxillary','Sphenoid','Nasopharynx']
    colors = ['red', 'blue', 'green', 'cyan']
    class_names = ['E_x1', 'E_y1', 'E_x2', 'E_y2',
                   'M_x1', 'M_y1', 'M_x2', 'M_y2',
                   'S_x1', 'S_y1', 'S_x2', 'S_y2',
                   'N_x1', 'N_y1', 'N_x2', 'N_y2']

    '''
    Ethmoid sinus  1
    Maxillary sinus  2
    Sphenoid sinus 3
    Nasopharynx 5
    '''


    def dcm_to_3d_npy(data_path, output_path, id):
        c, d = load(data_path)
        #np.save(output_path + '/' + id, c)
        return c, d


    def create_folder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)


    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


    def my_loss(y_true, y_pred):
        result = 0
        for i in range(4):
            result += rmse(y_true[4 * i:4 * (i + 1)], y_pred[4 * i:4 * (i + 1)])
        return result


    def my_loss2(y_true, y_pred):
        result = 0
        for i in range(4):
            result += K.mean(K.square(y_true[4 * i:4 * (i + 1)] - y_pred[4 * i:4 * (i + 1)]))
        return result


    def cut_npy(target, target_shapes, return_location=False):
        gap_indexs = []
        for i in range(3):
            gap_indexs.append((target.shape[i] - target_shapes[i]) // 2)

        a = gap_indexs[0]
        b = gap_indexs[0] + target_shapes[0]
        c = 1
        d = target_shapes[1] + 1
        e = gap_indexs[2]
        f = gap_indexs[2] + target_shapes[2]
        location = [a, b, c, d, e, f]
        target = target[:, c:d, e:f]
        if return_location:
            return target, location
        else:
            return target

    def make_block(npy_3d: np.array, npy_label: np.array, axis1):
        # 3dnpy, label_npy
        ###x = mip, y = box
        npy_3d = cut_npy(npy_3d, [40, 200, 300])
        npy_label = cut_npy(npy_label, [40, 200, 300])
        x_mip = np.min(npy_3d, axis=axis1)
        x_mip = x_mip.reshape(x_mip.shape[:2] + (1,))  # (512,512) -> (512,512,1)..
        # print("x_mip.shape : ", x_mip.shape)
        y_mip = []
        for i in [1, 2, 3, 5]:  # 부비동 종류
            locations = np.where(npy_label == i)
            # print('location shape : ', locations)
            locs = []
            try:
                if axis1 == 0:
                    locs = [x1, y1, x2, y2] = [locations[1].min(), locations[2].min(), locations[1].max(),
                                               locations[2].max()]
                elif axis1 == 1:
                    locs = [x1, y1, x2, y2] = [locations[0].min(), locations[2].min(), locations[0].max(),
                                               locations[2].max()]
                elif axis1 == 2:
                    locs = [x1, y1, x2, y2] = [locations[0].min(), locations[1].min(), locations[0].max(),
                                               locations[1].max()]
            except:
                locs = [200, 300, 200, 300]
            # print('locs : ',locs)
            y_mip.extend(locs)

        y_mip2 = np.stack(y_mip)
        return x_mip, y_mip2


    def make_block_x(npy_3d: np.array, axis1):
        # 3dnpy, label_npy
        ###x = mip, y = box
        npy_3d = cut_npy(npy_3d, [40, 200, 300])
        x_mip = np.min(npy_3d, axis=axis1)
        x_mip = x_mip.reshape(x_mip.shape[:2] + (1,))  # (512,512) -> (512,512,1)..
        # print("x_mip.shape : ", x_mip.shape)
        return x_mip

    # def make_block_x2(npy_3d: np.array, axis1):
    #     # 3dnpy, label_npy
    #     ###x = mip, y = box
    #     x_mip = np.min(npy_3d, axis=axis1)
    #     if axis1==1 or axis1==2:
    #         x_mip = cv.resize(x_mip, dsize=((40,)+x_mip.shape[1]))
    #     x_mip = x_mip.reshape(x_mip.shape[:2] + (1,))  # (512,512) -> (512,512,1)..
    #     print("x_mip.shape : ", x_mip.shape)
    #     return x_mip


    def get_intersection(a: list, b: list):
        intersection = []
        for i in range(6):
            if i < 4:
                intersection.append(max(a[i], b[i]))
            else:
                intersection.append(min(a[i], b[i]))
        return intersection


    def get_volume(a: list):
        result = 1
        for i in range(3):
            result *= abs(a[i + 3] - a[i])
        return result


    def DSC(a: list, b: list, intersection: list):
        return (2 * get_volume(intersection)) / (get_volume(a) + get_volume(b))

    numpy1, dcm_info = dcm_to_3d_npy(input_folder_path, output_folder_path,id)
    #numpy1 = np.load(output_folder_path+'/'+id+'.npy')
    #print(numpy1.shape, numpy1.T.shape, numpy1[:,:,1].shape)
    #print(output_folder_path+'/'+id+'.nii')
    #save(numpy1, output_folder_path+'/'+id+'.nii', dcm_info)
    numpy1 = numpy1.T
    blank_numpy = np.zeros(numpy1.shape)

    sinus_target_shape = [40,200,300]
    HEIGHT_LEN = sinus_target_shape[0]

    models = []
    for i in range(3):
        model = keras.models.load_model(model_path+'/mip_mse3_model%d_10_200'%(i),custom_objects={'my_loss': my_loss, 'rmse':rmse, 'my_loss2':my_loss2})
        models.append(model)

    np_x= numpy1
    pred_result = np.zeros((4,6))
    for main_axis in range(3):
        temp_index = [0,1,2]
        temp_index.remove(main_axis)

        MIDDLE_GAP = (np_x.shape[0]-HEIGHT_LEN)//2
        x0_predict= make_block_x(np_x, axis1=main_axis)#정규화 전
        if main_axis == 1 or main_axis == 2:
            x0_predict_shape = x0_predict.shape
            temp_x_mip = np.ones((200, x0_predict_shape[1], 1)) * (-300)
            temp_x_mip[:x0_predict_shape[0], :x0_predict_shape[1], :] = x0_predict

            x0_predict = temp_x_mip
        x0_predict = (x0_predict+1024)/4000
        x0_predict = x0_predict.reshape((1,)+x0_predict.shape)
        #print('pred temp : ',models[main_axis].predict(x0_predict)*511//1)
        pred = (models[main_axis].predict(x0_predict)*511//1).reshape((4,4))
        print(input_folder_path)
        print('predict : ')
        print(pred)
        #temp_index[0], temp_index[1], temp_index[0]+3, temp_index[1]+3
        for index in range(4):
            pred_result[index][temp_index[0]] = max(pred[index][0],0)
            pred_result[index][temp_index[1]] = max(pred[index][1],0)
            pred_result[index][temp_index[0]+3] = max(pred[index][2],0)
            pred_result[index][temp_index[1]+3] = max(pred[index][3],0)


        plt.subplot(121)
        #(512,512,1) 등에서 마지막 1 뗌
        plt.imshow(x0_predict[0].reshape(x0_predict[0].shape[:2]),cmap='gray')
        plt.xlabel('predict')
        ax = plt.gca()

        for i in range(pred.shape[0]):
            rect1 = xlu, ylu, xrd, yrd = pred[i]
            rect = patches.Rectangle((xlu,ylu),
                          xrd-xlu,
                          yrd-ylu,
                          linewidth=2,
                          edgecolor=colors[i],
                          fill = False)

            ax.add_patch(rect)
        #plt.show()

    pred//=2
    print('pred_result : ')
    print(pred_result)

    #make nii
    blank_target, location1 = cut_npy(blank_numpy,sinus_target_shape, return_location=True)

    for i in [1,0,2,3]:
        now = a,b,c,d,e,f = pred_result[i].astype(np.int)
        #print('now :',now)
        tag = [1,2,3,5][i]
        if is_skeletal_or_box:
            blank_target[a:d+1,b:b+line_width,c:c+line_width] = tag
            blank_target[a:d+1,b:b+line_width,f-line_width:f] = tag
            blank_target[a:d+1,e-line_width:e,c:c+line_width] = tag
            blank_target[a:d+1,e-line_width:e,f-line_width:f] = tag

            blank_target[a:a+line_width,b:e+1,c:c+line_width] = tag
            blank_target[a:a+line_width,b:e+1,f-line_width:f] = tag
            blank_target[d-line_width:d,b:e+1,c:c+line_width] = tag
            blank_target[d-line_width:d,b:e+1,f-line_width:f] = tag

            blank_target[a:a+line_width,b:b+line_width,c:f+1] = tag
            blank_target[a:a+line_width,e-line_width:e,c:f+1] = tag
            blank_target[d-line_width:d,b:b+line_width,c:f+1] = tag
            blank_target[d-line_width:d,e-line_width:e,c:f+1] = tag
        else:
            blank_target[a:d,b:e,c:f] = tag
            blank_target[a+line_width:d-line_width, b+line_width:e-line_width,c+line_width:f-line_width] = 0
    #print(blank_target.shape, location1)

    a,b,c,d,e,f = location1
    blank_numpy[:,c:d,e:f] = blank_target
    blank_numpy = blank_numpy.T
    save_path = output_folder_path+'/'+id+'_%s.nii'%('skeleton' if is_skeletal_or_box else 'box')
    save(blank_numpy, save_path, dcm_info)
    print(save_path,'saved(%s version)'%('skeleton' if is_skeletal_or_box else 'box'))

if __name__=='__main__':
    main()
