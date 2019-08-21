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
    def median_accuracy_line_of_horizont(x, y, model, inp_w, inp_h, steps=1, visualization=False):
        avg_distance=[]
        max_distance=[]
        
        line_of_horizont=Line_of_horizont_fitting()
        for label, img in zip(y, x):

            image = np.uint8(255 * img)
            
            label_med=line_of_horizont.median_blur(label,5)
            label_med=line_of_horizont.get_binary_image(label_med, 0.5)
            fit_line= line_of_horizont.horizont_line_from_binary_image(label_med)

            label=line_of_horizont.get_binary_image(label, 0.5)
            height,width=label.shape
            label_line = np.zeros([height,width], dtype = "uint8")
            cv2.line(label_line, (int(fit_line[2]-fit_line[0]*width), 
                                             int(fit_line[3]-fit_line[1]*width)), 
                     (int(fit_line[2]+fit_line[0]*width), 
                      int(fit_line[3]+fit_line[1]*width)), (255, 255, 255), 1)

            image_pred = line_of_horizont.resize_image(img, inp_w, inp_h)
            pred=line_of_horizont.predict_segmentation(image_pred, model)
            pred=line_of_horizont.get_binary_image(pred, 0.5)
            pred=line_of_horizont.resize_image(pred, width, height)

            fit_line, predict=line_of_horizont.horizont_line_pipeline(image, model, inp_w, inp_h, steps, 5)

            pred_line = np.zeros([height,width], dtype = "uint8")
            cv2.line(pred_line, (int(fit_line[2]-fit_line[0]*width), 
                                             int(fit_line[3]-fit_line[1]*width)), 
                     (int(fit_line[2]+fit_line[0]*width), 
                      int(fit_line[3]+fit_line[1]*width)), (255, 255, 255), 1)

            distance=[]
            for j in range (width):
                for i in range (height):
                    if(label_line[i,j]==255):
                        y1=i
                    if(pred_line[i,j]==255):
                        y2=i
                distance.append(abs(y1-y2))

            avg_y= int((y1+y2)/2)
            
            avg_distance.append(np.mean(distance)/width)
            max_distance.append(max(distance))
            
            if(visualization):
                print("avg_distance: ", (np.mean(distance)/width)," - max_distance: ", (max(distance)))
                plt.imshow(label_line)
                plt.show()
                plt.imshow(pred_line)
                plt.show()
        
        return avg_distance, max_distance
    
    
    @staticmethod
    def accuracy_on_line_of_horizont_area(x, y, model, inp_w, inp_h, steps=1, visualization=False):
        recall_list=[]
        precision_list=[]
        specificity_list=[]
        accuracy_list=[]
        f1score_list=[]

        line_of_horizont=Line_of_horizont_fitting()
        for label, img in zip(y, x):

            image = np.uint8(255 * img)
            label=line_of_horizont.get_binary_image(label, 0.5)
            height,width=label.shape

            image_pred = line_of_horizont.resize_image(img, inp_w, inp_h)
            pred=line_of_horizont.predict_segmentation(image_pred, model)
            pred=line_of_horizont.get_binary_image(pred, 0.5)
            pred=line_of_horizont.resize_image(pred, width, height)

            fit_line, predict=line_of_horizont.horizont_line_pipeline(image, model, inp_w, inp_h, steps)

            line_annotation_image = np.zeros([height,width], dtype = "uint8")
            cv2.line(line_annotation_image, (int(fit_line[2]-fit_line[0]*width), 
                                             int(fit_line[3]-fit_line[1]*width)), 
                     (int(fit_line[2]+fit_line[0]*width), 
                      int(fit_line[3]+fit_line[1]*width)), (255, 255, 255), 1)

            for i in range (height):
                if(label[i,0]==1):
                    y1=i
                    break

            for i in range (height):
                if(label[i,width-1]==1):
                    y2=i
                    break

            avg_y= int((y1+y2)/2)

            annotation_image = label[avg_y-100:avg_y+100, 0:width]
            pred_image = pred[avg_y-100:avg_y+100, 0:width]

            label=annotation_image
            pred=pred_image

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

            if(visualization):
                print("Recall: ", recall," - Precision: ", precision, " - Specificity: ", specificity, " - Accuracy: ", 
                      accuracy, " - F1score: ", f1score)
                plt.imshow(label)
                plt.show()
                plt.imshow(pred)
                plt.show()

        return recall_list, precision_list, specificity_list, accuracy_list, f1score_list
    
    
    @staticmethod
    def accuracy_on_images(x, y, model, inp_w, inp_h, steps=1, visualization=False):
        recall_list=[]
        precision_list=[]
        specificity_list=[]
        accuracy_list=[]
        f1score_list=[]
        
        line_of_horizont=Line_of_horizont_fitting()
        for label, img in zip(y, x):

            label=line_of_horizont.get_binary_image(label, 0.5)
            height,width=label.shape
            image_pred = line_of_horizont.resize_image(img, 160, 160)
            pred=line_of_horizont.predict_segmentation(image_pred, model)
            pred=line_of_horizont.get_binary_image(pred, 0.5)
            pred=line_of_horizont.resize_image(pred, width, height)

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
            
            if(visualization):
                print("Recall: ", recall," - Precision: ", precision, " - Specificity: ", specificity, " - Accuracy: ", 
                      accuracy, " - F1score: ", f1score)
                plt.imshow(label)
                plt.show()
                plt.imshow(pred)
                plt.show()
            
        return recall_list, precision_list, specificity_list, accuracy_list, f1score_list
            
    
    
    @staticmethod
    def test_speed_from_video(filename, model, inp_w, inp_h, n_iteration, steps=1):
        lineofhorizont = Line_of_horizont_fitting()
        reader = imageio.get_reader(filename,  'ffmpeg')
        fps = reader.get_meta_data()['fps']
        n_steps=0
        frame_to_discard = 10
        now=time()
        for i in range(n_iteration):
            if i == frame_to_discard:
                start_time=now
            if i > frame_to_discard:
                #n_steps+=1
                elapsed_time = now - start_time
                #print(n_steps, elapsed_time)
            or_image=reader.get_data(i)
            or_height, or_width, or_depth = or_image.shape

            fit_line, predict=lineofhorizont.horizont_line_pipeline(or_image, model, inp_w, inp_h, steps)

            predict = predict.reshape(or_height,or_width,1)
            predict1 = predict*255
            predict= np.uint8(np.concatenate((predict,predict,predict1),axis=2))
            imageOUT = cv2.bitwise_or(or_image,predict)
            imageOUT=cv2.line(imageOUT, (int(fit_line[2]-fit_line[0]*or_height), int(fit_line[3]-fit_line[1]*or_width)), 
                              (int(fit_line[2]+fit_line[0]*or_height), int(fit_line[3]+fit_line[1]*or_width)), 
                              (255, 0, 255), 5)
            now=time()
            
        reader.close()
        return ((i-frame_to_discard))/elapsed_time
    
    @staticmethod
    def test_speed_from_video_v2(reader, model, inp_w, inp_h, n_iteration, steps=1):
        n_steps=0
        frame_to_discard = 10
        now=time()
        lineofhorizont = Line_of_horizont_fitting()
        for i in range(n_iteration):
            if i == frame_to_discard:
                start_time=now
            if i > frame_to_discard:
                #n_steps+=1
                elapsed_time = now - start_time
                #print(n_steps, elapsed_time)
            or_image=reader.get_data(i)
            or_height, or_width, or_depth = or_image.shape

            fit_line, predict=lineofhorizont.horizont_line_pipeline(or_image, model, inp_w, inp_h, steps)

            predict = predict.reshape(or_height,or_width,1)
            predict1 = predict*255
            predict= np.uint8(np.concatenate((predict,predict,predict1),axis=2))
            imageOUT = cv2.bitwise_or(or_image,predict)
            imageOUT=cv2.line(imageOUT, (int(fit_line[2]-fit_line[0]*or_height), int(fit_line[3]-fit_line[1]*or_width)), 
                              (int(fit_line[2]+fit_line[0]*or_height), int(fit_line[3]+fit_line[1]*or_width)), 
                              (255, 0, 255), 5)
            now=time()

        return ((i-frame_to_discard))/elapsed_time
    
    @staticmethod
    def test_from_video(filename, model, inp_w, inp_h, n_iteration, steps=1):
        lineofhorizont = Line_of_horizont_fitting()
        reader = imageio.get_reader(filename,  'ffmpeg')
        fps = reader.get_meta_data()['fps']
        now=time()
        start_time=now
        for i in range(n_iteration):
            elapsed_time = now - start_time
            print(i, elapsed_time)
            or_image=reader.get_data(i)
            plt.imshow(or_image)
            plt.show()
            or_height, or_width, or_depth = or_image.shape

            fit_line, predict=lineofhorizont.horizont_line_pipeline(or_image, model, inp_w, inp_h, steps)

            predict = predict.reshape(or_height,or_width,1)
            predict1 = predict*255
            predict= np.uint8(np.concatenate((predict,predict,predict1),axis=2))
            imageOUT = cv2.bitwise_or(or_image,predict)
            imageOUT=cv2.line(imageOUT, (int(fit_line[2]-fit_line[0]*or_height), int(fit_line[3]-fit_line[1]*or_width)), 
                              (int(fit_line[2]+fit_line[0]*or_height), int(fit_line[3]+fit_line[1]*or_width)), 
                              (255, 0, 255), 5)
            now=time()
            plt.imshow(imageOUT)
            plt.show()
        reader.close()

    @staticmethod
    def test_from_folder(path, model, inp_w, inp_h, steps=1):
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

            fit_line, predict, img_inp_or, pred_inp_or=lineofhorizont.horizont_line_pipeline_verbose(or_image, model, inp_w, inp_h, steps)

            predict = predict.reshape(or_height,or_width,1)
            predict1 = predict*255
            predict= np.uint8(np.concatenate((predict,predict,predict1),axis=2))
            imageOUT = cv2.bitwise_or(or_image,predict)
            imageOUT=cv2.line(imageOUT, (int(fit_line[2]-fit_line[0]*or_height), int(fit_line[3]-fit_line[1]*or_width)), 
                              (int(fit_line[2]+fit_line[0]*or_height), int(fit_line[3]+fit_line[1]*or_width)), 
                              (255, 0, 255), 5)
            now=time()
        
            yield path_img, imageOUT, predict, img_inp_or, pred_inp_or
