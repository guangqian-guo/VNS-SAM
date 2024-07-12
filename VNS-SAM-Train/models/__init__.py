# from train.segment_anything_training.modeling.mask_decoder import MaskDecoder
from .sam_enhancev4_23_6 import MaskDecoderEnhance
from .sam_enhance_MGD import MaskDecoderMGD
from .sam_enhance_MGDv3 import MaskDecoderMGDv3
from .sam_enhancev4_23_6_light import MaskDecoderEnhanceLight
from .sam_enhancev4_23_6_fusion_results import MaskDecoderEnhancefusion
from .sam_enhancev4_23_6_fusion_results_token_interactive import MaskDecoderEnhancefusionTI
from .sam_enhancev4_23_6_fusion_results_token_interactivev2 import MaskDecoderEnhancefusionTIv2
from .sam_enhancev4_23_6_fusion_results_token_interactivev3 import MaskDecoderEnhancefusionTIv3
from .sam_enhancev4_23_6_fusion_results_token_interactivev4 import MaskDecoderEnhancefusionTIv4
from .sam_enhancev4_23_6_fusion_results_token_interactivev5 import MaskDecoderEnhancefusionTIv5
from .sam_enhancev4_23_6_fusion_results_token_interactivev6 import MaskDecoderEnhancefusionTIv6

from .sam_enhance_dwd import MaskDecoderEnhanceDWD
from .sam_hq import MaskDecoderHQ
from .sam_en_s import MaskDecoderEnhanceS
from .sam import MaskDecoderSAM


model_registry = {
    "sam-en": MaskDecoderEnhance,
    "sam-mgd": MaskDecoderMGD,
    "sam-mgdv3": MaskDecoderMGDv3,
    "sam-hq": MaskDecoderHQ,
    "sam-en-s": MaskDecoderEnhanceS,
    "sam": MaskDecoderSAM,
    "sam-en-light": MaskDecoderEnhanceLight,
    "sam-en-fusion": MaskDecoderEnhancefusion,
    "sam-en-dwd": MaskDecoderEnhanceDWD,

    ##### my try #######################################
    "sam-en-ti": MaskDecoderEnhancefusionTI,
    "sam-en-tiv2": MaskDecoderEnhancefusionTIv2,
    "sam-en-tiv3": MaskDecoderEnhancefusionTIv3,
    "sam-en-tiv4": MaskDecoderEnhancefusionTIv4,
    "sam-en-tiv5": MaskDecoderEnhancefusionTIv5,
    "sam-en-tiv6": MaskDecoderEnhancefusionTIv6,
    #####################################################
}



