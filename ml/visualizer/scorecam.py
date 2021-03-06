"""
Created on Wed Apr 29 16:11:20 2020

@author: Haofan Wang - github.com/haofanwang
"""
import os

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from ml.visualizer.utils import get_example_params, save_class_activation_images, save_image


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer=0):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in enumerate(list(self.model.children())):
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
                break
        return conv_output

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output = self.forward_pass_on_convolutions(x)
        # Forward pass on the classifier
        x = self.model(x)
        return conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer=0):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def visualize(self, input_image, input_path, result_dir, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.cpu().numpy())
        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(input_image.size(-2), input_image.size(-1)),
                                         mode='bilinear', align_corners=False)
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            w = F.softmax(self.extractor.forward_pass(input_image*norm_saliency_map)[1],dim=1)[0][target_class]
            cam += w.data.cpu().numpy() * target[i, :, :].data.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[3],
                       input_image.shape[2]), Image.ANTIALIAS))/255

        original_image = input_image.cpu().detach().numpy()[0]
        original_image = np.transpose(original_image, [1, 2, 0])
        reverse_mean = np.array([-0.4432, -0.3938, -0.3764])
        reverse_std = np.array([1/0.1560, 1/0.1815, 1/0.1727])
        original_image /= reverse_std
        original_image -= reverse_mean
        original_image = np.uint8(original_image * 255)

        filename = f'{input_path.split("/")[-1].split(".")[0]}'
        save_class_activation_images(original_image, cam, filename, result_dir=result_dir)


if __name__ == '__main__':
    # Get params
    target_example = 0  # Snake
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(target_example)
    # Score cam
    score_cam = ScoreCam(pretrained_model, target_layer=11)
    # Generate cam mask
    cam = score_cam.generate_cam(prep_img, target_class)
    # Save mask
    save_class_activation_images(original_image, cam, file_name_to_export)
    print('Score cam completed')
