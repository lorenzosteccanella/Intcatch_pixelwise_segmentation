import cv2
import numpy as np
import scipy
import glob
import matplotlib.pyplot as plt
from Line_of_horizont_fitting import Line_of_horizont_fitting
import imageio
from time import time, sleep

class Utils:
    @staticmethod
    def accuracy_on_images(x, y, model1):
        recall_list=[]
        precision_list=[]
        specificity_list=[]
        accuracy_list=[]
        f1score_list=[]
        
        line_of_horizont=Line_of_horizont_fitting()
        for label, img in zip(y, x):

            label=line_of_horizont.get_binary_image(label, 0.5)

            pred=line_of_horizont.predict_segmentation(img, model1)
            pred=line_of_horizont.get_binary_image(pred, 0.5)

            True_neg=len(np.where((label==0)&(pred==0))[0])
            False_neg=len(np.where((label==1)&(pred==0))[0])
            True_pos=len(np.where((label==1)&(pred==1))[0])
            False_pos=len(np.where((label==0)&(pred==1))[0])
            precision=True_pos/(True_pos+False_pos)
            recall=True_pos/(True_pos+False_neg)
            specificity=1-(True_neg/(True_neg+False_pos))
            accuracy=(True_pos+True_neg)/(True_pos+True_neg+False_pos+False_neg)
            f1score=2*(precision*recall)/(precision+recall)
            
            recall_list.append(recall)
            precision_list.append(precision)
            specificity_list.append(specificity)
            accuracy_list.append(accuracy)
            f1score_list.append(f1score)
            
        return recall_list, precision_list, specificity_list, accuracy_list, f1score_list
            
    
    
    @staticmethod
    def test_speed_from_video(filename, model, inp_w, inp_h, steps=10):
        lineofhorizont = Line_of_horizont_fitting()
        reader = imageio.get_reader(filename,  'ffmpeg')
        fps = reader.get_meta_data()['fps']
        now=time()
        start_time=now
        for i in range(steps):
            elapsed_time = now - start_time
            print(i, elapsed_time)
            or_image=reader.get_data(i)
            or_height, or_width, or_depth = or_image.shape

            fit_line, predict=lineofhorizont.horizont_line_pipeline(or_image, model, inp_w, inp_h, 5)

            predict = predict.reshape(or_height,or_width,1)
            predict1 = predict*255
            predict= np.uint8(np.concatenate((predict,predict,predict1),axis=2))
            imageOUT = cv2.bitwise_or(or_image,predict)
            imageOUT=cv2.line(imageOUT, (int(fit_line[2]-fit_line[0]*or_width), int(fit_line[3]-fit_line[1]*or_width)),
                              (int(fit_line[2]+fit_line[0]*or_width), int(fit_line[3]+fit_line[1]*or_width)), 
                              (255, 0, 255), 20)
            now=time()
    
    @staticmethod
    def test_from_video(filename, model, inp_w, inp_h, steps=10):
        lineofhorizont = Line_of_horizont_fitting()
        reader = imageio.get_reader(filename,  'ffmpeg')
        fps = reader.get_meta_data()['fps']
        now=time()
        start_time=now
        for i in range(steps):
            elapsed_time = now - start_time
            print(i, elapsed_time)
            or_image=reader.get_data(i)
            plt.imshow(or_image)
            plt.show()
            or_height, or_width, or_depth = or_image.shape

            fit_line, predict=lineofhorizont.horizont_line_pipeline(or_image, model, inp_w, inp_h, 5)

            predict = predict.reshape(or_height,or_width,1)
            predict1 = predict*255
            predict= np.uint8(np.concatenate((predict,predict,predict1),axis=2))
            imageOUT = cv2.bitwise_or(or_image,predict)
            imageOUT=cv2.line(imageOUT, (int(fit_line[2]-fit_line[0]*or_width), int(fit_line[3]-fit_line[1]*or_width)),
                              (int(fit_line[2]+fit_line[0]*or_width), int(fit_line[3]+fit_line[1]*or_width)), 
                              (255, 0, 255), 20)
            now=time()
            plt.imshow(imageOUT)
            plt.show()

    @staticmethod
    def test_from_folder(path, model, inp_w, inp_h, steps=10):
        lineofhorizont = Line_of_horizont_fitting()
        path_images=glob.glob(path)
        images=[]
        now=time()
        start_time=now
        for path_img in path_images:
            elapsed_time = now - start_time
            print(elapsed_time)
            or_image=cv2.imread(path_img)
            or_image=cv2.cvtColor(or_image, cv2.COLOR_BGR2RGB)
            or_height, or_width, or_depth = or_image.shape

            fit_line, predict=lineofhorizont.horizont_line_pipeline(or_image, model, inp_w, inp_h, 5)

            predict = predict.reshape(or_height,or_width,1)
            predict1 = predict*255
            predict= np.uint8(np.concatenate((predict,predict,predict1),axis=2))
            imageOUT = cv2.bitwise_or(or_image,predict)
            imageOUT=cv2.line(imageOUT, (int(fit_line[2]-fit_line[0]*or_width), int(fit_line[3]-fit_line[1]*or_width)), 
                              (int(fit_line[2]+fit_line[0]*or_width), int(fit_line[3]+fit_line[1]*or_width)), 
                              (255, 0, 255), 20)
            now=time()
            plt.imshow(imageOUT)
            plt.show()
