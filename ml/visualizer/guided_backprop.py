"""
Created on Thu Oct 26 11:23:47 2017
Original Author:
@author: Utku Ozbulak - github.com/utkuozbulak
Changes for ResNet Compatibility:
Moritz Freidank - github.com/MFreidank
"""
import os

import torch
from torch.nn import ReLU
from torch.optim import Adam
import numpy as np

from ml.visualizer.utils import save_gradient_images


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model):
        # Put model in evaluation mode
        self.model = model
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.children())[0]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """

        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        # Loop through layers, hook up ReLUs with relu_hook_function
        for module in self.model.modules():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def visualize(self, input_image, input_path, result_dir):
        input_image.requires_grad = True
        gradient = self.generate_gradients(input_image)
        saliency = self.get_saliency(gradient)

        filename = f'{input_path.split("/")[-1].split(".")[0]}_guided_grad'
        save_gradient_images(gradient, filename, result_dir)
        filename = f'{input_path.split("/")[-1].split(".")[0]}_guided_saliency'
        save_gradient_images(saliency, filename, result_dir)

    def generate_gradients(self, input_image):
        # Forward pass
        model_output = self.model(input_image)
        self.model.zero_grad()
        target_class = torch.argmax(model_output, -1)
        grad_input,  = torch.autograd.grad(model_output[0, target_class], input_image)
        return grad_input.data.cpu().numpy()[0]

    def get_saliency(self, gradient):
        """
            Generates the absolute saliency map based on the gradient
        Args:
            gradient (numpy arr): Gradient of the operation to visualize

        returns:
            saliency map
        """
        abs_gradient = np.abs(gradient)
        saliency = (np.maximum(0, abs_gradient) / abs_gradient.max())
        return saliency

    # def optimize(self, input_image, target_class):
    #     input_image = torch.tensor(np.random.uniform(-0.1, 0.1, [1, 3, 512, 512]).astype(dtype=np.float32),
    #                                device='cuda', requires_grad=True)
    #     self.optimizer = Adam([input_image], lr=0.01, weight_decay=0)
    #     model_output = self.model(input_image)
    #     target_class = torch.argmax(model_output, -1)
    #     for i in range(50):
    #         # Forward pass
    #         model_output = self.model(input_image)
    #         l2 = torch.mean(input_image**2)
    #         (model_output[0, 0] - 10000*l2).backward()  # whatever is the biggest activation value make it even bigger
    #         img_tensor_grad = input_image.grad.data
    #         smooth_grads = img_tensor_grad / torch.mean(img_tensor_grad)
    #         # smooth_grads[smooth_grads < 1e-3] = 0
    #         input_image.data += 0.1 * smooth_grads  # gradient ascent
    #
    #         # input_image.data = clip(input_image.data)
    #         input_image.grad.data.zero_()  # clear the gradients otherwise they would get accumulated
    #
    #         print(f'{i}')  #: {float(smooth_grads)}')
    #     return input_image.detach().cpu().numpy()[0]
