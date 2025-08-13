"""
SOPHIA: Sophia Epigraphic Artificial Intelligence

Multimodal deep learning framework for reading ancient inscriptions
from Saint Sophia Cathedral in Kyiv, Ukraine.
"""

__version__ = "0.1.0"
__author__ = "SOPHIA Research Team"

from .data import InscriptionDataset, ImageAnnotationProcessor
from .models import SophiaModel, MultimodalTransformer
from .training import SophiaTrainer
from .inference import InscriptionReader

__all__ = [
    "InscriptionDataset",
    "ImageAnnotationProcessor", 
    "SophiaModel",
    "MultimodalTransformer",
    "SophiaTrainer",
    "InscriptionReader"
]
