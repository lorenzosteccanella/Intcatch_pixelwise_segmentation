import numpy as np
import random
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import os
import cv2
import itertools

class Dataset:
    
    def __init__(self, path, img_w, img_h, n_labels):
        self.path = path
        self.img_w = img_w
        self.img_h = img_h
        self.n_labels = n_labels
        self.data_shape = self.img_w * self.img_h
        
    def probability_augmentation(self, prob):
        random_prob=random.random()

        if(random_prob>prob):
            return False
        else:
            return True
        
    def invert_img(self, image, label):
        invert=iaa.Invert(1, per_channel=True)
        inverted_img=invert.augment_image(image)

        return inverted_img, label

    def sigmoid_cont_img(self, image, label, gain_max, gain_min, cutoff_max, cutoff_min):
        gain= random.uniform(gain_max, gain_min)
        cutoff= random.uniform(cutoff_max, cutoff_min)
        sigmoid_cont= iaa.SigmoidContrast(gain=gain, cutoff=cutoff)

        image = sigmoid_cont.augment_image(image)

        return image, label

    def shear(self, image, annotation, mins, maxs, step=1):
        angle=random.randrange(mins,maxs,step=step)
        shear=iaa.Affine(shear=angle)
        image_shear=shear.augment_image(image)
        annotation_shear=shear.augment_image(annotation)
        image_shear,annotation_shear=self.crop_from_center(image_shear,annotation_shear, 3, 3)
        return image_shear, annotation_shear

    def hue_saturation(self, image, annotation, minhs, maxhs):
        hue_sat=iaa.AddToHueAndSaturation((minhs, maxhs))
        image_hue_sat=hue_sat.augment_image(image)
        return image_hue_sat,annotation

    def brightness(self, image, annotation, min_b, max_b):
        brightness=iaa.Add(value=(min_b, max_b),per_channel=0)
        image_brightness=brightness.augment_image(image)
        return image_brightness,annotation 

    def motionblur(self, image, annotation, min_angle, max_angle, step=1):
        angle=random.randrange(min_angle,max_angle,step=step)
        motionblur=iaa.blur.MotionBlur(k=10, angle=angle)
        image_motion_blur=motionblur.augment_image(image)
        annotation_motion_blur=motionblur.augment_image(annotation)
        return image_motion_blur, annotation_motion_blur

    def resize_image(self, image, annotation, img_w, img_h):
        image = cv2.resize(image,(img_w,img_h))
        annotation = cv2.resize(annotation,(img_w,img_h))
        return image,annotation

    def crop_from_center(self, image, annotation, h_crop, w_crop):
        h,w,d=image.shape
        image_crop = image[int((h/2)-(h/h_crop)):int((h/2)+(h/h_crop)), int((w/2)-(w/w_crop)):int((w/2)+(w/w_crop))]
        annotation_crop = annotation[int((h/2)-(h/h_crop)):int((h/2)+(h/h_crop)), int((w/2)-(w/w_crop)):int((w/2)+(w/w_crop))]
        return image_crop, annotation_crop

    def zoom(self, image, annotation, min_zoom, max_zoom):
        zoom=random.uniform(min_zoom,max_zoom)
        rotation = iaa.Affine(scale=zoom,mode="symmetric")
        image_zoomed = rotation.augment_image(image)
        annotation_zoomed = rotation.augment_image(annotation)
        image_zoomed,annotation_zoomed=self.crop_from_center(image_zoomed,annotation_zoomed, 3, 3)
        return image_zoomed,annotation_zoomed

    def horizontal_flip(self, image, annotation):
        horizontal_flip = iaa.Fliplr(1.0)
        image = horizontal_flip.augment_image(image)
        annotation = horizontal_flip.augment_image(annotation)  
        return image,annotation

    def random_rotation(self, image, annotation, min_angle, max_angle, step=1):
        angle=random.randrange(min_angle,max_angle,step=step)
        rotation = iaa.Affine(scale=1,rotate=angle,mode="symmetric")
        image_rotated = rotation.augment_image(image)
        annotation_rotated = rotation.augment_image(annotation)
        image_rotated,annotation_rotated=self.crop_from_center(image_rotated,annotation_rotated, 3, 3)
        return image_rotated,annotation_rotated

    def binarylab(self, labels):
        x = np.zeros([img_h,img_w,n_labels])  

        for i in range(img_h):
            for j in range(img_w):
                x[i,j,labels[i][j]]=1
        return x

    def visualize_annotation(self, temp, plot=False):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0,5):
            r[temp==l]=label_colours[l,0]
            g[temp==l]=label_colours[l,1]
            b[temp==l]=label_colours[l,2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3), dtype=np.uint8)
        rgb[:,:,0] = (b)#[:,:,0]
        rgb[:,:,1] = (g)#[:,:,1]
        rgb[:,:,2] = (r)#[:,:,2]
        if plot:
            plt.imshow(rgb)
        else:
            return rgb

    def load_image(self, name):
        bgr=cv2.imread(name)
        rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def crop_center_image(self, image, annotation, img_w, img_h, augmentation=True):
        h,w,d=image.shape
        images=[]
        labels=[]
        number_of_crop=int(w/img_w)
        for j in range(number_of_crop):
            image_crop = image[int((h/2)-(img_h/2)):int((h/2)+(img_h/2)),int((j*img_w)):int((j+1)*img_w)]
            annotation_crop = annotation[int((h/2)-(img_h/2)):int((h/2)+(img_h/2)),int((j*img_w)):int((j+1)*img_w)]
            images.append(image_crop) 
            labels.append(annotation_crop)
        return images,labels
    
    @staticmethod
    def plot_image(image):
        image=image*255
        plt.imshow(image.astype(np.uint8))
        plt.show()
        
    @staticmethod
    def plot_binary_image(image):
        image=image*255
        binary_img = np.squeeze(image, axis=2)
        plt.imshow(binary_img)
        plt.show()
    
    def add_image(self, images, labels, image2add, label2add, img_w, img_h):
        image2add, label2add=self.resize_image(image2add, label2add, img_w, img_h)
        
        image2add = image2add/255.0
        
        images.append(image2add)
        labels.append(label2add)
        
        return images, labels
    
    def display_data_augmentation(self, imgs, w_n_imgs, h_n_imgs, margin_x, margin_y):
        w = w_n_imgs # Width of the matrix (nb of images)
        h = h_n_imgs # Height of the matrix (nb of images)
        n = w*h
        
        #Define the shape of the image to be replicated (all images should have the same shape)
        img_h, img_w, img_c = imgs[0].shape

        #Define the margins in x and y directions
        m_x = margin_x
        m_y = margin_y

        #Size of the full size image
        mat_x = img_w * w + m_x * (w - 1)
        mat_y = img_h * h + m_y * (h - 1)

        #Create a matrix of zeros of the right size and fill with 255 (so margins end up white)
        imgmatrix = np.zeros((mat_y, mat_x, img_c),np.uint8)
        imgmatrix.fill(255)

        #Prepare an iterable with the right dimensions
        positions = itertools.product(range(h), range(w))

        for (y_i, x_i), img in zip(positions, imgs):
            x = x_i * (img_w + m_x)
            y = y_i * (img_h + m_y)
            imgmatrix[y:y+img_h, x:x+img_w, :] = (img)
            
        return imgmatrix
    
    def single_image_augmentation(self, image_name, annotation_name):
        
        images=[]
        labels=[]
        
        print(os.getcwd() + "/" + self.path + image_name + "  -  ", end="")
        image = self.load_image(os.getcwd() + "/" + self.path + image_name)
        print(os.getcwd() + "/" + self.path + annotation_name)
        annotation = self.load_image(os.getcwd() + "/" + self.path + annotation_name)
        self.add_image(images, labels, image, annotation, self.img_w, self.img_h)

        images_crop_center, annotations_crop_center = self.crop_center_image(image,annotation, self.img_w, self.img_h)
        for image_crop_center, annotation_crop_center in zip(images_crop_center, annotations_crop_center):
            self.add_image(images, labels, image_crop_center, annotation_crop_center, self.img_w, self.img_h)
        image_horizontal_flip, annotation_horizontal_flip = self.horizontal_flip(image,annotation)
        self.add_image(images, labels, image_horizontal_flip, annotation_horizontal_flip, self.img_w, self.img_h)
        image_rotated,annotation_rotated = self.random_rotation(image,annotation, -10, 10)
        self.add_image(images, labels, image_rotated, annotation_rotated, self.img_w, self.img_h)
        image_zoomed, annotation_zoomed = self.zoom(image, annotation, 0.6, 1.2)
        self.add_image(images, labels, image_zoomed, annotation_zoomed, self.img_w, self.img_h)
        image_motion_blur, annotation_motion_blur = self.motionblur(image, annotation, 160, 360)
        self.add_image(images, labels, image_motion_blur, annotation_motion_blur, self.img_w, self.img_h)
        image_bright, annotation_bright = self.brightness(image, annotation,-50,+50)
        self.add_image(images, labels, image_bright, annotation_bright, self.img_w, self.img_h)
        image_huesat, annotation_huesat = self.hue_saturation(image, annotation, -30, +30)
        self.add_image(images, labels, image_huesat, annotation_huesat, self.img_w, self.img_h)
        image_shear, annotation_shear = self.shear(image, annotation, -50, +50)
        self.add_image(images, labels, image_shear, annotation_shear, self.img_w, self.img_h)
        inverted_img, inverted_label = self.invert_img(image, annotation)
        self.add_image(images, labels, inverted_img, inverted_label, self.img_w, self.img_h)
        sigmoid_img, sigmoid_label = self.sigmoid_cont_img(image, annotation, 9.0, 11.0, 0.0, 0.70)
        self.add_image(images, labels, sigmoid_img, sigmoid_label, self.img_w, self.img_h)
                    
        images = np.array(images)
        labels = np.array(labels)
        labels = labels[:,:,:,0:1]
            
        return images, labels
    
    def createDataset(self, augmentation=False, prob1=False):
        
        images=[]
        labels=[]
        
        with open(self.path+'data.txt') as f:
            txt = f.readlines()
            txt = [line.split(' ') for line in txt]
        for i in range(len(txt)):
            
            print(os.getcwd() + "/" + self.path + txt[i][0] + "  -  ", end="")
            image = self.load_image(os.getcwd() + "/" + self.path + txt[i][0])
            print(os.getcwd() + "/" + self.path + txt[i][1])
            annotation = self.load_image(os.getcwd() + "/" + self.path + txt[i][1])
            self.add_image(images, labels, image, annotation, self.img_w, self.img_h)
            if augmentation:
                
                images_crop_center, annotations_crop_center = self.crop_center_image(image,annotation, self.img_w, self.img_h)
                for image_crop_center, annotation_crop_center in zip(images_crop_center, annotations_crop_center):
                    self.add_image(images, labels, image_crop_center, annotation_crop_center, self.img_w, self.img_h)
                image_horizontal_flip, annotation_horizontal_flip = self.horizontal_flip(image,annotation)
                self.add_image(images, labels, image_horizontal_flip, annotation_horizontal_flip, self.img_w, self.img_h)
                if prob1 or self.probability_augmentation(0.8):
                    image_rotated,annotation_rotated = self.random_rotation(image,annotation, -10, 10)
                    self.add_image(images, labels, image_rotated, annotation_rotated, self.img_w, self.img_h)
                if prob1 or self.probability_augmentation(0.8):
                    image_zoomed, annotation_zoomed = self.zoom(image, annotation, 0.6, 1.2)
                    self.add_image(images, labels, image_zoomed, annotation_zoomed, self.img_w, self.img_h)
                if prob1 or self.probability_augmentation(0.6):
                    image_motion_blur, annotation_motion_blur = self.motionblur(image, annotation, 160, 360)
                    self.add_image(images, labels, image_motion_blur, annotation_motion_blur, self.img_w, self.img_h)
                if prob1 or self.probability_augmentation(0.6):
                    image_bright, annotation_bright = self.brightness(image, annotation,-50,+50)
                    self.add_image(images, labels, image_bright, annotation_bright, self.img_w, self.img_h)
                if prob1 or self.probability_augmentation(0.6):
                    image_huesat, annotation_huesat = self.hue_saturation(image, annotation, -20, +20)
                    self.add_image(images, labels, image_huesat, annotation_huesat, self.img_w, self.img_h)
                if prob1 or self.probability_augmentation(0.6):
                    image_shear, annotation_shear = self.shear(image, annotation, -20, +20)
                    self.add_image(images, labels, image_shear, annotation_shear, self.img_w, self.img_h)
                if prob1 or self.probability_augmentation(0.6):
                    inverted_img, inverted_label = self.invert_img(image, annotation)
                    self.add_image(images, labels, inverted_img, inverted_label, self.img_w, self.img_h)
                if prob1 or self.probability_augmentation(0.6):
                    sigmoid_img, sigmoid_label = self.sigmoid_cont_img(image, annotation, 9.0, 11.0, 0.0, 0.70)
                    self.add_image(images, labels, sigmoid_img, sigmoid_label, self.img_w, self.img_h)
                    
        images = np.array(images)
        labels = np.array(labels)
        labels = labels[:,:,:,0:1]
            
        return images, labels
        