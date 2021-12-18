from ml.visualizer import get_visualizer
from utils.device import put_minibatch_to_device
from ml.visualizer.utils import save_gradient_images, get_positive_negative_saliency


class Visualizer(object):

    def __init__(self, model, config, result_dir):
        """
        Class to control the visualization of what an nn model learned.

        :param config: config namespace containing the experiment configuration
        :param args: arguments of the training
        """
        self.result_dir = result_dir
        self.config = config

        # initialize the required elements for the ml problem
        self.visualizers = [get_visualizer(key, model) for key in config.vizualized_images]

    def visualize(self, input_image, input_path):
        """
        Visualize the model for the image

        :param input_image:
        """
        # save the original image
        filename = f'{input_path.split("/")[-1].split(".")[0]}'
        save_gradient_images(input_image[0].cpu().numpy(), filename, self.result_dir)

        for visualizer in self.visualizers:
            visualizer.visualize(input_image, input_path, self.result_dir)
