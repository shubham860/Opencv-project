import cv2
import numpy as np


image = cv2.imread('test_image.jpg')
lane_image = np.copy(image) 

#cannny edge detector
def canny_detection(image):
   grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
   blur_image = cv2.GaussianBlur(grayscale_image, (5,5), 0)
   canny_image = cv2.Canny(blur_image, 50, 150)
   return canny_image

canny_image = canny_detection(lane_image)
cv2.imshow('result3',canny_image)

#find region of interest
def region_of_interest(image):
    height = image.shape[0]
    traingle = np.array([
            [(200,height), (1100,height), (550,250)]
            ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, traingle, 255)
    #now we apply the masking between mask image and canny image to get the region of interest and in mask image white traingle represens the 1111111 and black represents 0000 in binary soo we apply bitwise and operation b/w both image to get the region of interest
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

cv2.imshow('result4',region_of_interest(canny_image))

cropped_image = region_of_interest(canny_image)
cv2.imshow('Region of interest',cropped_image)

#Apply hough transform method for find straight line of lanes
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10)
    return line_image        
    
lines = cv2.HoughLinesP(cropped_image,2 , np.pi/180, 100,np.array([]), minLineLength=40, maxLineGap=5)
line_image = display_lines(lane_image, lines)
cv2.imshow('Lane lines',line_image)


#combine images

combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow('scatter_image',combo_image)

#find average lines on image
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope  = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else: 
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])        
            
            
#averaged_lines = average_slope_intercept(lane_image, lines)
#line_image = display_lines(lane_image, averaged_lines)
#combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#cv2.imshow('Real_image',combo_image)

#video capture object
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened):   # returns true if video is intialised
    _,frame = cap.read()   #decode every video frame
    canny_image = canny_detection(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image,2 , np.pi/180, 100,np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('LANE DETECTION',combo_image)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.distroyAllWindows()
    







