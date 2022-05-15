import cv2
import numpy as np
from tkinter import *
from tkinter import ttk
from googletrans import Translator, LANGUAGES
from collections import Counter
from gtts import gTTS
from playsound import playsound
import os
import time

textUpdated = False
temp = ''

def nothing(x):
    pass

def speakText():
    tt = translateTextToSelectedLanguage()
    output = gTTS(text=(tt), lang=getSelectedLanguageCode(), slow=False)
    output.save("speak.mp3")
    playsound("speak.mp3")

def getSelectedLanguageCode():
    for key, value in LANGUAGES.items():
      if drop.get() == value:
        return key

def updateTextWithNewLanguage(e=None):
    global img_text, temp, textUpdated
    textUpdated = True
    print("TextUpdated:",textUpdated)
    temp = img_text
    img_text = translateTextToSelectedLanguage()

def translateTextToSelectedLanguage():
    translator = Translator()
    lang = getSelectedLanguageCode()
    t = translator.translate(img_text.lower(),dest=lang)
    return t.text

image_x, image_y = 64,64

from keras.models import load_model
classifier = load_model('Trained_model.h5')

def predictor():
       import numpy as np
       from keras.preprocessing import image
       test_image = image.load_img('1.png', target_size=(64, 64))
       test_image = image.img_to_array(test_image)
       """
       The expand_dims(arr,  axis) function is used to expand the shape of an array.
        Insert a new axis that will appear at the axis position in the expanded array shape.

        arr
        Input array

        axis
        Position where new axis to be inserted

        >>> a = np.array([2, 4])
        >>> b = np.expand_dims(a, axis=0)
        >>> b
        array([[2, 4]])
       """
       test_image = np.expand_dims(test_image, axis = 0)
       result = classifier.predict(test_image)
       
       if result[0][0] == 1:
              return 'A'
       elif result[0][1] == 1:
              return 'B'
       elif result[0][2] == 1:
              return 'BS'
       elif result[0][3] == 1:
              return 'C'
       elif result[0][4] == 1:
              return 'D'
       elif result[0][5] == 1:
              return 'E'
       elif result[0][6] == 1:
              return 'F'
       elif result[0][7] == 1:
              return 'G'
       elif result[0][8] == 1:
              return 'H'
       elif result[0][9] == 1:
              return 'I'
       elif result[0][10] == 1:
              return 'J'
       elif result[0][11] == 1:
              return 'K'
       elif result[0][12] == 1:
              return 'L'
       elif result[0][13] == 1:
              return 'M'
       elif result[0][14] == 1:
              return 'N'
       elif result[0][15] == 1:
              return None
       elif result[0][16] == 1:
              return 'O'
       elif result[0][17] == 1:
              return 'P'
       elif result[0][18] == 1:
              return 'Q'
       elif result[0][19] == 1:
              return 'R'
       elif result[0][20] == 1:
              return 'S'
       elif result[0][21] == 1:
              return ' '
       elif result[0][22] == 1:
              return 'T'
       elif result[0][23] == 1:
              return 'U'
       elif result[0][24] == 1:
              return 'V'
       elif result[0][25] == 1:
              return 'W'
       elif result[0][25] == 1:
              return 'X'
       elif result[0][25] == 1:
              return 'Y'
       elif result[0][25] == 1:
              return 'Z'




window=Tk()
#/Size in Pixels
window.minsize(408, 210)
btn = Button(window, text = 'Speak!', command = speakText)
btn.place(x=0, y=0);
clicked = StringVar()
clicked.set("english")

# Converting dictionary values into list
v = list(LANGUAGES.values())
names=[]
{names[i]:v[i] for i in range(len(names))}
v.pop(81)

drop = ttk.Combobox(window, textvariable=clicked, state="readonly", height=25, values=(v))
drop.bind("<<ComboboxSelected>>", updateTextWithNewLanguage)
drop.place(x=80, y=3)
txt=Text(window, height=10, width=50)
txt.place(x=0, y=25)
window.title('Converted Text')
window.geometry("300x200+10+10")

# If the stream failed to open with arg: 0(webcam) the the if condition will fail and VideoCapture(1) will be executed to open the stream with external camera

# This code will create the object of VideoCapture to perform image operations and will open camera stream.
cam = cv2.VideoCapture(0)
if cam.read()[0] == False:
    cam = cv2.VideoCapture(1)

# This method will create new windw with title "Trackbars"
cv2.namedWindow("Trackbars")

# createTrackbar will create and add the slider in following syntax
"""
Python: cv.CreateTrackbar(trackbarName, windowNameVideoCapture, value, count, onChange) → None
Parameters: 
trackbarname – Name of the created trackbar.
winname – Name of the window that will be used as a parent of the created trackbar.
value – Optional pointer to an integer variable whose value reflects the position of the slider. Upon creation, the slider position is defined by this variable.
count – Maximal position of the slider. The minimal position is always 0.
onChange – Pointer to the function to be called every time the slider changes position. This function should be prototyped as void Foo(int,void*); , where the first parameter is the trackbar position and the second parameter is the user data (see the next parameter). If the callback is the NULL pointer, no callbacks are called, but only value is updated.
userdata – User data that is passed as is to the callback. It can be used to handle trackbar events without using global variables.
"""



cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
cv2.createTrackbar("L - S", "Trackbars", 40, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 6, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

cv2.namedWindow("Recognize Frame")

img_counter = 0
img_text = ''

my_text = ''
predictDetails = []

while True:
    # cam.read() will return return multiple args(tuple) flag: indicates 'frame' is successfully captured and 'frame':MultiDimentional array (matrix) of image.
    flag, frame = cam.read()

    # flip will flip the whole image to identtify flipped image
    frame = cv2.flip(frame,1)

    """
    Parameters: 
    trackbarname – Name of the trackbar.
    winname – Name of the window that is the parent of the trackbar.
    --The function returns the current position of the specified trackbar.
    """
    l_h = cv2.getTrackbarPos("L - H", "Trackbars")
    l_s = cv2.getTrackbarPos("L - S", "Trackbars")
    l_v = cv2.getTrackbarPos("L - V", "Trackbars")
    u_h = cv2.getTrackbarPos("U - H", "Trackbars")
    u_s = cv2.getTrackbarPos("U - S", "Trackbars")
    u_v = cv2.getTrackbarPos("U - V", "Trackbars")

    # This method rectangle will create the rectangle in specified x1 and y1 co-ordinates with specified (R, G, B) 
    img = cv2.rectangle(frame, (425,100),(625,300), (0,255,0), thickness=2, lineType=8)

    lower_blue = np.array([l_h, l_s, l_v])
    upper_blue = np.array([u_h, u_s, u_v])

    # The Frame of area where your hand is...
    imcrop = img[102:298, 427:623]

    """
    The function converts an input image from one color space to another. In case of a transformation to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the bytes are reversed). So the first byte in a standard (24-bit) color image will be an 8-bit Blue component, the second byte will be Green, and the third byte will be Red. The fourth, fifth, and sixth bytes would then be the second pixel (Blue, then Green, then Red), and so on.
    """
    hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
    """
    The below line is used the set the threshhold values according to the light intensity in the backgroung:
    The parameters of the method are : 
    src: first input array.
    lowerb:  inclusive lower boundary array or a scalar.
    upperb:  inclusive upper boundary array or a scalar.
    """
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    """
    Syntax: cv2.putText(image, text, org, font, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])

    Parameters:
    image: It is the image on which text is to be drawn.
    text: Text string to be drawn.
    org: It is the coordinates of the bottom-left corner of the text string in the image. The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
    font: It denotes the font type. Some of font types are FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, , etc.
    fontScale: Font scale factor that is multiplied by the font-specific base size.
    color: It is the color of text string to be drawn. For BGR, we pass a tuple. eg: (255, 0, 0) for blue color.
    thickness: It is the thickness of the line in px.
    lineType: This is an optional parameter.It gives the type of the line to be used.
    bottomLeftOrigin: This is an optional parameter. When it is true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
    """
    # cv2.putText(frame, my_text, (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0))
    cv2.imshow("Recognize Frame", frame)
    cv2.imshow("mask", mask)
    
    #if cv2.waitKey(1) == ord('c'):
        
    img_name = "1.png"
    save_img = cv2.resize(mask, (image_x, image_y))
    cv2.imwrite(img_name, save_img)
    # print("{} written!".format(img_name))
    my_text = predictor()

    predictDetails.append(my_text);
    if(len(predictDetails) > 50):
      predictedElementToAppend = dict(Counter(predictDetails))
      print("Counter Dictionary", predictedElementToAppend)
      maxValKey = max(predictedElementToAppend, key=predictedElementToAppend.get)
      print("Max Val Key", predictedElementToAppend[maxValKey])
      if(maxValKey != None and int(predictedElementToAppend[maxValKey]) > 25):
        print("TU:",textUpdated)
        if(textUpdated):
          drop.set("english")
          img_text = temp.upper()
          textUpdated = False
        if(maxValKey == 'BS'):
          img_text = img_text[:-1]
        else:
          img_text += maxValKey
      
      predictDetails.clear()

    txt.delete('1.0', END)
    txt.insert(INSERT, img_text)

    if cv2.waitKey(1) == 27:
        break

    window.update()

cam.release()
cv2.destroyAllWindows()