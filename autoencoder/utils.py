import os
import json
from glob import glob
from random import choices
from typing import Dict, Any
#
import cv2
import numpy as np


def read_config_file(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r") as infile:
        return json.load(infile)


def sample_reference_examples(data_dir: str):
    examples = []
    for i in range(10):
        j = sorted(glob(os.path.join(data_dir, f"{i}_*.jpg")))[0]
        examples.append(j)
    return examples


def indent_str(message, num_indent=1):
    return "\t"*num_indent + message
