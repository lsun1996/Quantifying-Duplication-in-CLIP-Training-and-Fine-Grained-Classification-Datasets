"""load the laion400m dataset for image retrival"""
from lightning.data import StreamingDataset, StreamingDataLoader
from lightning.data.streaming.serializers import JPEGSerializer
import torchvision.transforms.v2 as T
import imagehash
import torchvision.transforms as T
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image
import io


class LAOINStreamingDataset(StreamingDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.serializer = JPEGSerializer()

    def __getitem__(self, index):
        id, image, text, _, _, _ = super().__getitem__(index)
        return Image.open(io.BytesIO(image)), text, str(id)