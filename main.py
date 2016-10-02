import os
import itertools
import cv2


SCALE_FACTOR = 1.11
MIN_NEIGHBORS = 5
MIN_SIZE = (30, 30)


image_path = (
    "img/src/01.jpg",
    "img/src/02.jpg",
    "img/src/03.jpg",
    "img/src/04.jpg",
    "img/src/05.jpg",
    "img/src/06.jpg",
    "img/src/07.jpg",
    "img/src/08.jpg",
    "img/src/09.jpg",
    "img/src/10.jpg",
    "img/src/11.jpg",
    "img/src/12.jpg",
)

cascade_path = (
    "opencv/data/haarcascades/haarcascade_frontalcatface.xml",
    "opencv/data/haarcascades/haarcascade_frontalcatface_extended.xml",
    "opencv/data/haarcascades/haarcascade_frontalface_alt.xml",
    "opencv/data/haarcascades/haarcascade_frontalface_alt_tree.xml",
    "opencv/data/haarcascades/haarcascade_frontalface_alt2.xml",
    "opencv/data/haarcascades/haarcascade_frontalface_default.xml",
    "opencv/data/haarcascades/haarcascade_profileface.xml",
)


def split_path(path):
    basename = os.path.basename(path)
    root, ext =  os.path.splitext(basename)
    return path, root, ext


def detect(cascade, image):
    return cascade.detectMultiScale(
        image,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS,
        minSize=MIN_SIZE
    )


def render(facerect, image, border_color=(255, 255, 255)):
    if 0 != len(facerect):
        for rect in facerect:
            cv2.rectangle(
                image,
                tuple(rect[0:2]),
                tuple(rect[0:2] + rect[2:4]),
                border_color,
                thickness=2
            )
    return image


if __name__ == "__main__":
    images = {root: cv2.imread(path) for path, root, ext in map(split_path, image_path)}
    grays = {name: (image, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) for name, image in images.items()}

    cascades = {root: cv2.CascadeClassifier(path) for path, root, ext in map(split_path, cascade_path)}

    dir_name = "img/dest/scaleFactor_{0}__minNeighbors_{1:0>2}__minSize_{2:0>2}x{3:0>2}".format(
        SCALE_FACTOR,
        MIN_NEIGHBORS,
        *MIN_SIZE
    )
    os.makedirs(dir_name)

    for (image_name, (image, gray)), (cascade_name, cascade) in itertools.product(grays.items(), cascades.items()):
        facerect = detect(cascade, gray)
        image = render(facerect, image)

        print(image_name, cascade_name)
        # print(facerect, end="\n\n")
        # print("{0}/{1}_{2}.jpg".format(dir_name, cascade_name, image_name))

        cv2.imwrite("{0}/{1}_{2}.jpg".format(dir_name, image_name, cascade_name), image)
