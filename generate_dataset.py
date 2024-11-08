import cv2
import numpy as np
import random
import os
import sys
from alive_progress import alive_bar

min_size = 10
dim = 512
WHITE = (255,255,255)

nb_rectangle = 1000
nb_ellipse = 1000
nb_triangle = 1000

def show_img(img):
    cv2.imshow("Dataset", img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        return True
    else:
        cv2.destroyAllWindows()

def write_img(img,path):
    # https://stackoverflow.com/questions/7624765/converting-an-opencv-image-to-black-and-white
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, bw_img) = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(path, bw_img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def rectangle():
    img = np.zeros((dim,dim,3), np.uint8)
    pt1 = (random.randint(0, dim-min_size), random.randint(0, dim-min_size))
    pt2 = (random.randint(pt1[0] + min_size, dim), random.randint(pt1[1] + min_size, dim))
    cv2.rectangle(img,pt1,pt2,WHITE,-1)
    return img

def ellipse():
    img = np.zeros((dim,dim,3), np.uint8)
    pt1 = (random.randint(0, dim-min_size), random.randint(0, dim-min_size))
    pt2 = (random.randint(0 + min_size, int(dim/2)-min_size), random.randint(0 + min_size, int(dim/2)-min_size))
    cv2.ellipse(img,pt1,pt2,0,0,360,WHITE,-1)
    return img

def triangle():
    img = np.zeros((dim,dim,3), np.uint8)
    pt1 = (random.randint(0, dim-min_size), random.randint(0, dim-min_size))
    pt2 = (random.randint(0, dim-min_size), random.randint(0, dim-min_size))
    pt3 = (random.randint(0, dim-min_size), random.randint(0, dim-min_size))
    pts = np.array([pt1,pt2,pt3], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(img, np.array([pts]),WHITE)
    return img

def generate_rectangle():
    
    if not os.path.exists("dataset/rectangle"):
        os.mkdir("dataset/rectangle")
    
    print("Generating rectangles...")

    with alive_bar(nb_rectangle) as bar:
        for i in range(0,nb_rectangle):
            img = rectangle()
            # if show_img(img):
            #     break
            write_img(img, f"dataset/rectangle/rectangle_image{i}.jpg")
            bar()

def generate_ellipse():

    if not os.path.exists("dataset/ellipse"):
        os.mkdir("dataset/ellipse")

    print("Generating ellipses...")

    with alive_bar(nb_ellipse) as bar:
        for i in range(0,nb_ellipse):
            img = ellipse()
            write_img(img,f"dataset/ellipse/ellipse_image{i}.jpg")
            bar()

def generate_triangle():

    if not os.path.exists("dataset/triangle"):
        os.mkdir("dataset/triangle")

    print("Generating triangles...")

    with alive_bar(nb_triangle) as bar:
        for i in range(0,nb_triangle):
            img = triangle() 
            write_img(img,f"dataset/triangle/triangle_image{i}.jpg")
            bar()

def generate_random():

    functions = [(rectangle,"rectangle"),(ellipse,"ellipse"),(triangle,"triangle")]
    func = random.choice(functions)
    img = func[0]()
    label = func[1]
    return img,label    

def main():

    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    generate_rectangle()
    generate_ellipse()
    generate_triangle()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Ctrc + c pressed, exiting...")
        cv2.destroyAllWindows()
        sys.exit(1)