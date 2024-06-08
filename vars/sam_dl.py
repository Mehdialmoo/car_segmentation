import os
import urllib


def maybe_download_sam():
    model_path = "saved_models/sam_vit_h_4b8939.pth"
    if not os.path.exists(model_path):
        print("Downloading SAM model...")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", model_path)
    else:
        print("SAM model already downloaded.")
