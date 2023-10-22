from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier


width = 200  # canvas width
height = 200  # canvas height
center = height // 2
white = (255, 255, 255)  # canvas back


def save():
    global draw
    # save image to hard drive
    filename = "a_" + e1.get() + ".png"
    output_image.save(filename)
    im = cv2.imread(filename)

    grayImage = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(
        grayImage, 250, 255, cv2.THRESH_BINARY
    )
    image = image_resize(blackAndWhiteImage, height=20)
    # Using cv2.imwrite() method
    # Saving the image
    cv2.imwrite(filename, image)
    # Use this to if you read png image
    # im = cv2.cvtColor(im,cv2.COLOR_BGRA2RGB)
    # Use this to if you read jpg image
    # im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    image[image < 254] = 0
    image[image > 254] = 1

    # Clean Canvas
    draw.rectangle((0, 0, width, height), fill="white")
    canvas.delete("all")

    # Image is a numpy array
    print(image)
    print("Faire un truc")
    np.savetxt("a_" + e1.get(), image)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=35)
    draw.line([x1, y1, x2, y2], fill="black", width=5)


master = Tk()

# create a tkinter canvas to draw on
canvas = Canvas(master, width=width, height=height, bg="white")
canvas.pack()

# create an empty PIL image and draw object to draw on
output_image = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(output_image)
canvas.pack(expand=YES, fill=BOTH)
canvas.bind("<B1-Motion>", paint)

# add a button to save the image
button = Button(text="save", command=save)
button.pack()
e1 = Entry(master)
e1.pack()

master.mainloop()
