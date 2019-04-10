import cv2
import numpy as np
import scipy
import glob
import matplotlib.pyplot as plt

class Line_of_horizont_fitting:

    line=[]

    def inputNormalized(self, image, img_w, img_h):
        image=self.resize_image(image, img_w, img_h)
        #image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image=image/255.0
        return image

    def resize_image(self, image, img_w, img_h):
        image = cv2.resize(image,(img_w,img_h))
        return image

    def plot_binary_image(self, image):
        image=image*255
        binary_img = np.squeeze(image, axis=2)
        plt.imshow(binary_img)
        plt.show()

    def get_binary_image(self, image, treshold):
        image = cv2.threshold(image,treshold,1,cv2.THRESH_BINARY)
        return image[1]

    def binary_edge_detection(self, image):
        edges = image - scipy.ndimage.morphology.binary_dilation(image)
        return edges

    def median_blur(self, img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.blur(img, (kernel_size, kernel_size))

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):  
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        return lines

    def binary2gray(self, image):
        image = np.uint8(255 * image)
        return image

    def Collect_points(self, lines):

        # interpolation & collecting points for RANSAC
        points=[]
        for line in lines:
            new_point = np.array([[int(line[0]),int(line[1])]])
            points.append(new_point)
            new_point = np.array([[int(line[2]),int(line[3])]])
            points.append(new_point)

        return points

    def smoothing(self, lines, pre_frame=10):
        # collect frames & print average line
        lines = np.squeeze(lines)
        avg_line = np.array([0.0,0.0,0.0,0.0])

        for ii,line in enumerate(reversed(lines)):
            if ii == pre_frame:
                break
            avg_line += line
        avg_line = avg_line / pre_frame

        return avg_line

    def getLineImage(self, image,label,fit_line, width, height):
        height,width,_=image.shape
        imageOUT = cv2.bitwise_or(image,label)
        cv2.line(imageOUT, (int(fit_line[2]-fit_line[0]*width), int(fit_line[3]-fit_line[1]*width)), (int(fit_line[2]+fit_line[0]*width), int(fit_line[3]+fit_line[1]*width)), (255, 0, 255), 12)

    def predict_segmentation(self, image, model):
        predict = model.predict(image[None,...])

        return predict[0]

    def horizont_line_pipeline(self, image, model, img_w, img_h, average_n_frame, kernel_median_blur=50, predict_treshold=0.5,
                               rho = 2, theta = np.pi/180, threshold = 20, min_line_length = 20, max_line_gap = 5):
        or_image=image

        or_height, or_width, or_depth = or_image.shape

        image=self.inputNormalized(or_image,img_w,img_h)
        predict = self.predict_segmentation(image,model)
        predict = self.median_blur(predict,kernel_median_blur)
        predict = self.get_binary_image(predict,predict_treshold)
        predict = self.resize_image(predict, or_width, or_height)
        output = self.binary_edge_detection(predict)
        output = self.binary2gray(output)

        rho = rho# distance resolution in pixels of the Hough grid
        theta = theta # angular resolution in radians of the Hough grid
        threshold = threshold     # minimum number of votes (intersections in Hough grid cell)
        min_line_length = min_line_length #minimum number f pixels making up a line
        max_line_gap = 5    # maximum gap in pixels between connectable line segments
        lines=self.hough_lines(output, rho, theta, threshold, min_line_length, max_line_gap)
        line_arr = np.squeeze(lines)
        points = self.Collect_points(line_arr)

        if(len(points)<2):
            points= line_arr.reshape(lines.shape[0]*2,2)

        if(len(points)>2):
            fit_line = cv2.fitLine(np.float32(points), cv2.DIST_HUBER, 1, 0.001, 0.001)
            self.line.append(fit_line)

            if len(self.line) > 10:
                fit_line = self.smoothing(self.line, average_n_frame)

        return fit_line, predict