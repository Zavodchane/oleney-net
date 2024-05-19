import os
from datetime import datetime

from PIL import Image
from PIL.ExifTags import TAGS

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_image_metadata(image_path) -> dict:

    image = Image.open(image_path)
    image_name = os.path.basename(image_path)

    metadata = {"name": image_name}

    info = image.getexif()

    if info:
        for tag, value in info.items():
            tag_name = TAGS.get(tag, tag)
            
            if tag_name == "Make":
                metadata["make"] = value

            if tag_name == "Model":
                metadata["model"] = value

            if tag_name == "DateTime":
                date_object = datetime.strptime(value.replace(":", "-", 2), DATE_FORMAT)
                metadata["datetime"] = date_object

    return metadata
