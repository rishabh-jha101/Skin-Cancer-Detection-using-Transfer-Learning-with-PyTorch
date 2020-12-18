import numpy as np
import torch
import torch.nn as nn
from prepare import test_loader
from model import vgg16

## track test loss (Classification)
batch_size = 20

    ## if GPU is_available
#test_on_gpu = torch.cuda.is_available()
test_on_gpu = False
# if not test_on_gpu:
#     print('CUDA is not available.  Testing on CPU ...')
# else:
#     print('CUDA is available.  Testing on GPU ...')

# Load model
vgg16.load_state_dict(torch.load('vgg16_skin_cancer.pt'))

    ##Model should be on GPU with dataset
# if test_on_gpu:
#     vgg16.cuda()

#same criterion as training
criterion = nn.CrossEntropyLoss()
number_of_classes = 2
class_correct = list(0. for i in range(number_of_classes))
class_total = list(0. for i in range(number_of_classes))

#Keeping track of test loss
test_loss = 0.0

vgg16.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    # if test_on_gpu:
    #     data, target = data.cuda(), target.cuda()
    # forward pass: compute predicted outputs by passing inputs to the model
    output = vgg16(data)
    # calculate the batch loss
    loss = criterion(output, target)
    # update test loss
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not test_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    print(correct)
    # calculate test accuracy for each object class
    for i in range(len(correct)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

## Test Error in each class
classes = ['Benign', 'Malignant']

for i in range(number_of_classes):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

#Overall Accuracy
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))
