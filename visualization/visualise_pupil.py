import matplotlib.pyplot as plt
import cv2


def visualize_pupil(input_img, output_img):

    # outputs_soft = torch.softmax(output_img)
    # outputs_soft = np.transpose(outputs_sig.cpu().detach().numpy(), (1, 2, 0)).copy()
    plt.imshow(input_img)
    plt.show()

    plt.imshow(output_img)
    plt.show()


# def visualize_mask(image, mask):
#     with torch.no_grad():
#     pupil_trainer.model.eval()
#     pupil_trainer.model = pupil_trainer.model.to(device)
#     for i, (inputs, masks, opened) in enumerate(pupil_trainer.dataloaders['test']):

#         inputs = inputs.to(device)
#         outputs = pupil_trainer.model(inputs)


#         image = np.transpose(inputs[0].cpu().numpy(), (1, 2, 0)).copy()
#         outputs_sig = torch.sigmoid(outputs['out'][0])
#         outputs_sig = np.transpose(outputs_sig.cpu().numpy(), (1, 2, 0)).copy()

#         visualize_pupil(image, outputs_sig)
#         ellipse = utils.fit_ellipse(outputs_sig)
#         if ellipse:
#             image = utils.draw_ellipse(image, ellipse)
#             plt.imshow(image)
#             plt.show()

#         if i  == 5 :
#             break


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


def draw_point(image, point):
    x, y = point
    shape = 192 / 2
    img = cv2.circle(image, center=(
        int(x + shape), int(y + shape)), radius=2, color=(0, 255, 0))
    return img
