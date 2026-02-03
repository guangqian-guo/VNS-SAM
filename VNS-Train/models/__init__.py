# from train.segment_anything_training.modeling.mask_decoder import MaskDecoder
from .vns_sam import VNS_SAM_Decoder

model_registry = {
    "vns-sam": VNS_SAM_Decoder
}