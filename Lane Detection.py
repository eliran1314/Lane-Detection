import cv2
import numpy as np #scientific computing
import os
import time
duration = 0.1  # seconds
freq = 2500  # Hz



global flag
flag=0
global count
count =0
global start_time
start_time = time.time()

def beep():
    print ("\a")

def make_coordinates (image, line_paremeters):
    #print (type (line_paremeters))
    global flag
    global count
    try:
        slope, intercept = line_paremeters
        #print (slope)
        #print (intercept)
        y1 = 700
        y2 = int (y1*(3/4)) #the image maximum height to draw the lines
        x1 = int ((y1 - intercept)/slope)
        x2 = int ((y2 - intercept)/slope)
        # print ("Lane ok")
        return np.array([x1,y1,x2,y2])
    except:
        #print ("not in lane")
            #os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
            #os.system('play --no-show-progress --null --channels 1 synth %s sine %f' %( 0.1, 400))
        #count = count+1
        return np.array([0,0,0,0])


def average_slope_intercept(image,lines):
    global flag
    global start_time
    left_fit =[]
    right_fit=[]
    if  lines is None:
        lines = []
        #print ("not in lane")
        if (flag>100):
            print ("You Moved lane !")
            flag=0
        else:
            flag=0
        #os.system('play --no-show-progress --null --channels 1 synth %s sine %f' %( 0.1, 400))
        #os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    for line in lines:
        #print ("Lane ok")
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append ((slope,intercept))
    flag = flag+1
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line,right_line])



def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #convert to gray
    blur = cv2.GaussianBlur(gray, (5, 5), 0) #blure the image, optional
    canny = cv2.Canny(blur, 50, 150) #gradiant the image what are the values ?
    return canny

def display_lines (image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            #print (x1,y1,x2,y2)
            try:
                cv2.line(line_image, (x1,y1), (x2,y2), (255, 0, 0), 10)
            except:
                pass
    return line_image


def region_of_intrest(image):
    height = image.shape[0]
    polygons = np.array([
    [(20, 700), (1500, 700), (700,400)] # the pixels to draw the lines (left line width(x) and height(y), right line, the end of the triangle )
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    maksed_image = cv2.bitwise_and(image, mask)
    return maksed_image

# image = cv2.imread('test_image.jpg') #read the image
# lane_image = np.copy(image) #copy the image
# canny_image = canny(lane_image)
# cropped_image = region_of_intrest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# average_lines = average_slope_intercept(lane_image,lines)
# line_image = display_lines(lane_image, average_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow('result',combo_image) #show image
# cv2.waitKey(0) # close on 0 click

cap = cv2.VideoCapture("21trim.mp4")
while (cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_intrest(canny_image)
     #detecting the lines (image, pixels, degree, threshhold, empty array, minimum line lenght in pixels, maximun pixels distance between line to be connected to single line)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 20, np.array([]), minLineLength=60, maxLineGap=2)
    average_lines = average_slope_intercept(frame,lines)
    line_image = display_lines(frame, average_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result',combo_image) #show image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
