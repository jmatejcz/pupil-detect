import matplotlib.pyplot as plt


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
