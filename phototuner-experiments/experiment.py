# rotation
# crop
# white balance (same as temperature)
# contrast
# saturation
# vibrance
# sharpness
# curves
# noise
# orton effect?
# vignette
# highlights
# shadows
# exposure
# tint
# clarity
import base64
import glob
import json
import logging
import os
import shutil
from io import BytesIO
import sys
import numpy as np
import cv2

import requests
from PIL import Image, ImageEnhance

methods = [
    "brightness",
    "contrast",
    "saturation",
    "sharpness",
    "crop",
    "noise",
    "rotation",
    "white_balance",
    "vibrance",
    "curves",
    "orton",
    "vignette",
    "highlights",
    "shadows",
    "exposure",
    "tint",
    "clarity",
]
model_url = "http://0.0.0.0:59101/v1/models/nima-mobilenet-aesthetic:predict"
image_dir = "./images/"
tmp_dir = "./tmp/"


def _setup_logger():
    logger = logging.getLogger("autotuner")
    logger.setLevel("DEBUG")
    f = logging.Formatter("%(asctime)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setFormatter(f)
    sh.setLevel("DEBUG")
    logger.addHandler(sh)
    return logger


logger = _setup_logger()


def _main():
    image_names = glob.glob(image_dir + "*")
    image_names = image_names[:1]
    data = _construct_data(image_names)
    data = _process_images(data)


def _construct_data(image_names):
    data = {}
    for image in image_names:
        data[image] = {"original_score": None, "methods": _get_data_blocks()}
    return data


def _get_data_blocks():
    methods_list = []

    for method in methods:
        methods_list.append({"method_name": method})

    return methods_list


def _process_images(data):
    image_names = data.keys()

    for image_name in image_names:
        logger.info("Starting to process <{}>".format(image_name))
        _empty_tmp_dir()

        original_image = _load_image(image_name)
        original_image.save(tmp_dir + "/original.png")

        resized_image = _resize_image(original_image, 224, 224)
        original_image.close()

        original_score = _get_scores_for_image(resized_image)

        method_blocks = data[image_name]["methods"]
        filled_blocks = []
        for block in method_blocks:
            updated_block = {
                "method_name": block["method_name"],
                "lower_val_score": None,
                "higher_val_score": None,
            }

            fn_to_call = _method_function_switcher(updated_block["method_name"])
            if fn_to_call is None:
                logger.info(
                    "\tMethod <{}> not implemented.".format(
                        updated_block["method_name"]
                    )
                )
                continue  # If we haven't implemented then skip this block

            # If we get to here then we can process the image
            logger.info("\t<{}>.".format(updated_block["method_name"]))

            resized_image.save(tmp_dir + "/resized.png")
            scores = fn_to_call()
            logger.info(
                "\t\toriginal <{}>, low <{}>, high <{}>.".format(
                    round(original_score, 2), round(scores[0], 2), round(scores[1], 2)
                )
            )

            filled_blocks.append(updated_block)
        # _empty_tmp_dir()
        resized_image.close()
        data[image_name]["methods"] = filled_blocks
    return data


def _get_scores_for_image(image):
    encoded_image = _encode_image(image).tolist()
    body = {"instances": [encoded_image]}
    body = json.dumps(body)

    headers = {"content-type": "application/json"}
    json_response = requests.post(model_url, data=body, headers=headers,).json()

    predictions_list = json_response["predictions"][0]

    total = 0
    for idx, pred_val in enumerate(predictions_list):
        total += pred_val * (idx + 1)

    return total


def _load_image(image_path):
    img = Image.open(image_path)
    return img


def _resize_image(image, width, height):
    img = image.resize((width, height))
    return img


def _encode_image(image):
    img_arr = np.asarray(image)
    return img_arr


def _empty_tmp_dir():
    folder = tmp_dir
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logger.info("Failed to delete %s. Reason: %s" % (file_path, e))


def _method_function_switcher(method):
    switcher = {
        "brightness": _do_brightness,
        "contrast": _do_contrast,
        "saturation": _do_saturation,
        "sharpness": _do_sharpness,
        "crop": _do_crop,
    }

    fn_to_call = switcher.get(method, None)
    return fn_to_call


def _do_saturation():
    orig = Image.open(tmp_dir + "/resized.png")
    converter = ImageEnhance.Color(orig)
    lower_img = converter.enhance(0.8)
    higher_img = converter.enhance(1.2)

    return _get_scores_for_image(lower_img), _get_scores_for_image(higher_img)


def _do_contrast():
    orig = Image.open(tmp_dir + "/resized.png")
    converter = ImageEnhance.Contrast(orig)
    lower_img = converter.enhance(0.9)
    higher_img = converter.enhance(1.1)

    return _get_scores_for_image(lower_img), _get_scores_for_image(higher_img)


def _do_brightness():
    orig = Image.open(tmp_dir + "/resized.png")
    converter = ImageEnhance.Brightness(orig)
    lower_img = converter.enhance(0.8)
    higher_img = converter.enhance(1.2)

    return _get_scores_for_image(lower_img), _get_scores_for_image(higher_img)


def _do_sharpness():
    orig = Image.open(tmp_dir + "/resized.png")
    converter = ImageEnhance.Sharpness(orig)
    lower_img = converter.enhance(0.7)
    higher_img = converter.enhance(1.5)

    return _get_scores_for_image(lower_img), _get_scores_for_image(higher_img)


def _do_crop():
    orig = Image.open(tmp_dir + "/original.png")

    inset_amount = 0.15

    x0 = 0
    xil = orig.size[0] * (0 + inset_amount)
    xir = orig.size[0] * (1 - inset_amount)
    x100 = orig.size[0]

    y0 = 0
    yit = orig.size[1] * (0 + inset_amount)
    yib = orig.size[1] * (1 - inset_amount)
    y100 = orig.size[1]

    cropped_images = {
        "tl": orig.crop((x0, y0, xir, yib)),
        "tm": orig.crop((xil, y0, xir, yib)),
        "tr": orig.crop((xil, y0, x100, yib)),
        "lm": orig.crop((x0, yit, xir, yib)),
        "mm": orig.crop((xil, yit, xir, yib)),
        "rm": orig.crop((xil, yit, x100, yib)),
        "bl": orig.crop((x0, yit, xir, y100)),
        "bm": orig.crop((xil, yit, xir, y100)),
        "br": orig.crop((xil, yit, x100, y100)),
    }

    cropped_scores = {
        "tl": None,
        "tm": None,
        "tr": None,
        "lm": None,
        "mm": None,
        "rm": None,
        "bl": None,
        "bm": None,
        "br": None,
    }

    for key in cropped_images.keys():
        resized = _resize_image(cropped_images[key], 224, 224)
        resized.save(tmp_dir + "/" + key + ".png")
        cropped_scores[key] = round(_get_scores_for_image(resized), 2 )

    print(cropped_scores)

    return -1, -1



def _save_test(lower, higher):
    lower.save(tmp_dir + "/lower.png")
    higher.save(tmp_dir + "/higher.png")
    sys.exit()


if __name__ == "__main__":
    _main()
