# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from CAM.GradCAM import GradCAM, GradCAMpp, SmoothGradCAM
from CAM.GroupCAM import GroupCAM
from CAM.Guided_Backprop import GuidedBackProp
from CAM.Integrated_Gradidents import IntegratedGradients
from CAM.ScoreCAM import ScoreCAM
from CAM.Smooth_Integrated import SmoothIntGrad

CAM = {
    'GradCAM': GradCAM,
    'GradCAMpp': GradCAMpp,
    "SmoothGradCAM": SmoothGradCAM,
    "GuidedBackProp": GuidedBackProp, # 不能运行
    "IntegratedGradients": IntegratedGradients, # 不能运行
    'ScoreCAM': ScoreCAM,
    'SmoothIntGrad': SmoothIntGrad,
    'GroupCAM': GroupCAM,
}


def get_CAM(name, **kwargs):
    return CAM[name](**kwargs)
