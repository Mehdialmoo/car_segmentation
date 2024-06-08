from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything import SamPredictor


def load_sam():
    sam_checkpoint = "./saved_models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    return SamAutomaticMaskGenerator(sam)


def cifar10():
    pass
