import numpy as np
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.solver import Solver

# Load the (preprocessed) CIFAR10 data.

data = get_CIFAR10_data()
for k, v in list(data.items()):
  print(('%s: ' % k, v.shape))

best_model = None
################################################################################
# TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
# batch normalization and dropout useful. Store your best model in the         #
# best_model variable.                                                         #
################################################################################
layer_nums = [3, 5]
learning_rates = [1e-3, 1e-4, 1e-5]
weight_scales = [1, 1e-1, 1e-2]
regulation_strengths = [0.01, 0.1, 1]

best_val_acc = 0.0

for ln in layer_nums:
    for lr in learning_rates:
        for ws in weight_scales:
            for reg in regulation_strengths:
                model = FullyConnectedNet([100] * ln, input_dim=32*32*3,
                                         num_classes=10, reg=reg,
                                         weight_scale=ws, dtype=np.float64)
                solver = Solver(model, data,
                               update_rule='adam',
                               optim_config={
                                   'learning_rate' : lr,
                               },
                               lr_decay=0.95,
                               batch_size=100,
                               num_epochs=10)
                solver.train()
                if solver.val_acc_history[-1] > best_val_acc:
                    best_model = model
                    best_val_acc = solver.val_acc_history[-1]
                    print('better accuracy: %f with ln %d, lr %e, ws %e, reg %f' %
                          (best_val_acc, ln, lr, ws, reg))
y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())
