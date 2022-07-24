# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from CAM.GradCAM import GradCAM, GradCAMpp, SmoothGradCAM
from CAM.GroupCAM import GroupCAM
from CAM.ScoreCAM import ScoreCAM

CAM = {
    'GradCAM': GradCAM,
    'GradCAMpp': GradCAMpp,
    "SmoothGradCAM": SmoothGradCAM,
    'ScoreCAM': ScoreCAM,
    'GroupCAM': GroupCAM,
}


def get_CAM(name, **kwargs):
    return CAM[name](**kwargs)
