import time
import copy
import torch
import matplotlib.pyplot as plt
import cv2


def train_first_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    dataloaders,
    dataset_sizes,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model = model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # tp = 0
            # tn = 0
            # fp = 0
            # fn = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # print(inputs, labels)
                inputs = inputs.to(device)
                print(labels)
                labels = [l.to(device) for l in labels]
                # labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    # print(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    outputs.to(device)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # if phase == 'test':
                #     tp += torch.sum(preds == labels.data and preds == 1)
                #     tn += torch.sum(preds == labels.data and preds == 0)
                #     fp += torch.sum(preds != labels.data and preds == 1)
                #     fn += torch.sum(preds != labels.data and preds == 0)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            # print(f'tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}')

            # deep copy the model
            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_second_model(
    model,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    dataloaders,
    dataset_sizes,
):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    model = model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # tp = 0
            # tn = 0
            # fp = 0
            # fn = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # print(inputs, labels)
                inputs = inputs.to(device)
                print(labels)
                labels = [l.to(device) for l in labels]
                # labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    print(outputs, labels)
                    outputs.to(device)

                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
                # if phase == 'test':
                #     tp += torch.sum(preds == labels.data and preds == 1)
                #     tn += torch.sum(preds == labels.data and preds == 0)
                #     fp += torch.sum(preds != labels.data and preds == 1)
                #     fn += torch.sum(preds != labels.data and preds == 0)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            # print(f'tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}')

            # deep copy the model
            # if phase == "test" and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:4f}")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def visualize_model(model, num_images=6, class_to_show=None):
    was_training = model.training
    model.eval()
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["test"]):
            inputs = inputs.to(device)
            labels = labels.to(device)
            label = int(labels[0])
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            image = np.transpose(inputs[0].cpu().numpy(), (1, 2, 0)).copy()
            x, y = image.shape[0], image.shape[1]
            digit = int(preds[0])
            # if class_to_show:
            #     cv2.putText(image, f"{digit}", (x-20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
            #     ax = plt.imshow(image)
            if label == class_to_show:
                cv2.putText(
                    image,
                    f"{digit}",
                    (x - 20, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                )
                ax = plt.imshow(image)

            if i >= num_images:
                model.train(mode=was_training)
                return

        model.train(mode=was_training)


def evaluate(model):
    model.eval()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders["val"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            if preds == 1:
                pass
