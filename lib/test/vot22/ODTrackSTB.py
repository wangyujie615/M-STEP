import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from lib.test.vot22.stb_tracker import run_vot_exp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
run_vot_exp('m_step', 'baseline_mste_tpe_full_16_18_20', vis=False)