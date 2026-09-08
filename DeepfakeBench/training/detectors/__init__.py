import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from utils.registry import DETECTOR
from .effort_detector import EffortDetector
from .effort_detector_aepa import EffortDetectorAEPA
from .effort_detector_maxev import EffortDetectorMaxEvidence
from .effort_detector_dualcomp import EffortDetectorDualComplement
from .effort_detector_lfeq import EffortDetectorLFEQ
from .effort_detector_lfeq_mean import EffortDetectorLFEQMean
from .effort_detector_lfeq_per_token import EffortDetectorLFEQPerToken
from .effort_detector_lfeq_concat import EffortDetectorLFEQConcat
