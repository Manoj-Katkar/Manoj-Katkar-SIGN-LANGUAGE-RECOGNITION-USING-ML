import cv2
import time
import numpy as np
import os


def nothing(x):
    pass

#We defined the size of image ad 64*64
image_x, image_y = 64, 64

def create_folder(folder_name):
    if not os.path.exists('./mydata/training_set/' + folder_name):
        os.mkdir('./mydata/training_set/' + folder_name)
    if not os.path.exists('./mydata/test_set/' + folder_name):
        os.mkdir('./mydata/test_set/' + folder_name)
        
def capture_images(ges_name):
    create_folder(str(ges_name))
    
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Capture")
    # time.sleep(3)

    img_counter = 0
    t_counter = 1
    training_set_image_name = 1
    test_set_image_name = 1
    listImage = [1,2,3,4,5]

    cv2.namedWindow("Trackbars")
    # time.sleep(3)

    cv2.createTrackbar("L - H", "Trackbars", 0, 179, nothing)
    cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("L - V", "Trackbars", 0, 255, nothing)
    cv2.createTrackbar("U - H", "Trackbars", 179, 179, nothing)
    cv2.createTrackbar("U - S", "Trackbars", 255, 255, nothing)
    cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

    for loop in listImage:
        while True:

            ret, frame = cam.read()
            frame = cv2.flip(frame, 1)

            l_h = cv2.getTrackbarPos("L - H", "Trackbars")
            l_s = cv2.getTrackbarPos("L - S", "Trackbars")
            l_v = cv2.getTrackbarPos("L - V", "Trackbars")
            u_h = cv2.getTrackbarPos("U - H", "Trackbars")
            u_s = cv2.getTrackbarPos("U - S", "Trackbars")
            u_v = cv2.getTrackbarPos("U - V", "Trackbars")

            img = cv2.rectangle(frame, (425, 100), (625, 300), (0, 255, 0), thickness=2, lineType=8)

            lower_blue = np.array([l_h, l_s, l_v])
            upper_blue = np.array([u_h, u_s, u_v])
            imcrop = img[102:298, 427:623]
            hsv = cv2.cvtColor(imcrop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_blue, upper_blue)

            '''syntax:
            cv2.bitwise_and(frame,frame, mask= mask)'''
            result = cv2.bitwise_and(imcrop, imcrop, mask=mask)

            cv2.putText(frame, str(img_counter), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
            cv2.imshow("test", frame)
            cv2.imshow("mask", mask)
            cv2.imshow("result", result)

            waitKey = cv2.waitKey(1)
            if waitKey == ord('c'):

                if t_counter <= 350:
                    """
                    str.format() is one of the string formatting methods in Python3, which allows multiple substitutions and value formatting. This method lets us concatenate elements within a string through positional formatting.

                    Syntax : { } .format(value)
                    Parameters :
                    (value) : Can be an integer, floating point numeric constant, string, characters or even variables.
                    Returntype : Returns a formatted string with the value passed as parameter in the placeholder position.
                    """
                    img_name = "./mydata/training_set/" + str(ges_name) + "/{}.png".format(training_set_image_name)
                    save_img = cv2.resize(mask, (image_x, image_y))
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    training_set_image_name += 1


                if t_counter > 350 and t_counter <= 400:
                    img_name = "./mydata/test_set/" + str(ges_name) + "/{}.png".format(test_set_image_name)
                    save_img = cv2.resize(mask, (image_x, image_y))
                    cv2.imwrite(img_name, save_img)
                    print("{} written!".format(img_name))
                    test_set_image_name += 1
                    if test_set_image_name > 250:
                        break


                t_counter += 1
                if t_counter == 401:
                    t_counter = 1
                img_counter += 1


            elif waitKey==27:
                print("you pressed escape")
                break

        if test_set_image_name > 250:
            break


    cam.release()
    cv2.destroyAllWindows()

    
ges_name = input("Enter gesture name: ")
#creating dataset for specified gesture
capture_images(ges_name)