import matplotlib.pyplot as plt
import cv2


def visualize_ifopened(image, pred: int):

    x, y = image.shape[0], image.shape[1]
    cv2.putText(
        image,
        f"{pred}",
        (x - 20, y - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
    )
    return image


def visualize_pupil(input_img, output_img):

    plt.imshow(input_img)
    plt.show()

    plt.imshow(output_img)
    plt.show()


def draw_ellipse(model):
    pass


def draw_normal_vectors_2D(image, start_point, vector, color):
    x1, y1 = start_point
    x2, y2 = start_point + vector * 25
    shape = 192 / 2
    img = cv2.arrowedLine(
        image,
        (int(x1 + shape), int(y1 + shape)),
        (int(x2 + shape), int(y2 + shape)),
        color=color,
        thickness=1,
    )
    return img


def draw_ellipse(image, ellipse, color):
    img = cv2.ellipse(
        image,
        (int(ellipse[0][0]), int(ellipse[0][1])),
        (int(ellipse[1][0] // 2), int(ellipse[1][1] // 2)),
        int(ellipse[2]),
        0,
        360,
        color,
    )
    return img


def draw_point(image, point):
    x, y = point
    shape = 192 / 2
    img = cv2.circle(
        image,
        center=(int(x + shape), int(y + shape)),
        radius=3,
        color=(0, 0, 0),
        thickness=-1,
    )
    return img
