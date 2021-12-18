# -*- coding: utf-8 -*-
# @Time    : 2021/11/11
# @Author  : Albert Gregus
# @Email   : g.albert95@gmail.com
import torchmetrics

from ml.visualizer.guided_backprop import GuidedBackprop
from ml.visualizer.occlusion_sensitivity import OcclusionSensitivity
from ml.visualizer.scorecam import ScoreCam


def get_visualizer(visualizer_name, model, params={}):
    """
    Get the visualizer function according to the config name and parameters.

    :param visualizer_name: the name of the visualizer
    :param model: the model to be used for the
    :return: the metric function
    """

    if visualizer_name == 'GuidedBackprop':
        return GuidedBackprop(model, **params)
    elif visualizer_name == 'ScoreCam':
        return ScoreCam(model, **params)
    elif visualizer_name == 'OcclusionSensitivity':
        return OcclusionSensitivity(model, **params)
    elif visualizer_name == 'ClassSpecificImageGeneration':
        return ClassSpecificImageGeneration(model, **params)
    elif visualizer_name == 'RegularizedClassSpecificImageGeneration':
        raise NotImplementedError
        return RegularizedClassSpecificImageGeneration(model, **params)
    else:
        raise ValueError(f'Wrong metric name: {visualizer_name}')
