#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#  Copyright © 2020 The DeePray Authors. All Rights Reserved.
#
#  Distributed under terms of the GNU license.
#  ==============================================================================


"""
Build model

Author:
    Hailin Fu, hailinfufu@outlook.com
"""
from deepray.model import model_lr


def BuildModel(flags):
    if flags.model == 'lr':
        model = model_lr.LogisitcRegression(flags)
    elif flags.model == 'fm':
        from deepray.model import model_fm
        model = model_fm.FactorizationMachine(flags)
    elif flags.model == 'ffm':
        from deepray.model import model_ffm
        model = model_ffm.FactorizationMachine(flags)
    elif flags.model == 'nfm':
        from deepray.model import model_nfm
        model = model_nfm.NeuralFactorizationMachine(flags)
    elif flags.model == 'afm':
        from deepray.model import model_afm
        model = model_afm.AttentionalFactorizationMachine(flags)
    elif flags.model == 'deepfm':
        from deepray.model import model_deepfm
        model = model_deepfm.DeepFM(flags)
    elif flags.model == 'xdeepfm':
        from deepray.model import model_xdeepfm
        model = model_xdeepfm.ExtremeDeepFMModel(flags)
    elif flags.model == 'wdl':
        from deepray.model import model_wdl
        model = model_wdl.WideAndDeepModel(flags)
    elif flags.model == 'dcn':
        from deepray.model import model_dcn
        model = model_dcn.DeepCrossModel(flags)
    elif flags.model == 'din':
        from deepray.model import model_din
        model = model_din.DeepInterestNetwork(flags)
    elif flags.model == 'dien':
        from deepray.model import model_dien
        model = model_dien.DeepInterestEvolutionNetwork(flags)
    elif flags.model == 'flen':
        from deepray.model import model_flen
        model = model_flen.FLENModel(flags)
    elif flags.model == 'autoint':
        from deepray.model import model_autoint
        model = model_autoint.AutoIntModel(flags)
    else:
        raise ValueError('--model {} was not found.'.format(flags.model))
    return model
