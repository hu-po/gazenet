import time
import copy
import torch

def train_gazenet(model, dataloader, criterion, optimizer, scheduler, **kwargs):
    """
    Trains a gazenet model
    """

    # Keep track of start time so we know how long training is taking
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    for epoch in range(1, kwargs['num_epochs'] + 1):
        print('Epoch {}/{}'.format(epoch, kwargs['num_epochs']))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for data in dataloader[phase].dataloader:
                # Convert input data into variables for training
                inputs = data['image']
                labels = torch.stack((data['gaze_x'], data['gaze_y']), 1).float()
                if kwargs['use_gpu']:
                    inputs, labels = torch.autograd.Variable(inputs.cuda()), \
                                     torch.autograd.Variable(labels.cuda())
                else:
                    inputs, labels = torch.autograd.Variable(inputs), \
                                     torch.autograd.Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)

            epoch_loss = running_loss / len(dataloader[phase].dataset)
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if kwargs.get('writer', False):
                # Add example input and output
                kwargs['writer'].add_image('image', inputs, epoch)
                kwargs['writer'].add_text('output', str(outputs), epoch)
                # Add loss to tensorboard run
                loss_name = 'loss/' + phase
                kwargs['writer'].add_scalar(loss_name, epoch_loss, epoch)

            # deep copy the model
            if phase == 'test' and epoch_loss > best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
