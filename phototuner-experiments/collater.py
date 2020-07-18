# Show all effects on multiple instances of single images
#
#    effect -    |    effect -    |    effect -    |    effect -    |    effect -    |
# original image | original image | original image | original image | original image |
#    effect +    |    effect +    |    effect +    |    effect +    |    effect +    |
#
# repeat for multiple images

#
#
#

# Show crop effect worst | original | best vertically for multiple images

# Show all effects worst | original | best vertically for multiple images
import glob
import json
import os
import sys
import numpy as np

import requests
from PIL import Image, ImageEnhance, ImageFont, ImageDraw


def _get_image_paths(images_dir):
    extensions = ["jpg", "jpeg", "png", "gif"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, "*." + ext)))
    return image_files


def _brightness(effect_direction, image):
    converter = ImageEnhance.Brightness(image)

    if effect_direction is "decrease":
        return converter.enhance(0.5)

    return converter.enhance(1.5)


def _contrast(effect_direction, image):
    converter = ImageEnhance.Contrast(image)

    if effect_direction is "decrease":
        return converter.enhance(0.5)

    return converter.enhance(1.5)


def _saturation(effect_direction, image):
    converter = ImageEnhance.Color(image)

    if effect_direction is "decrease":
        return converter.enhance(0.5)

    return converter.enhance(1.5)


def _sharpness(effect_direction, image):
    converter = ImageEnhance.Sharpness(image)

    if effect_direction is "decrease":
        return converter.enhance(0.01)

    return converter.enhance(1.8)


def _get_image_score(image, model_url):
    image.resize((224, 224))
    encoded_image = np.asarray(image).tolist()
    body = {"instances": [encoded_image]}
    body = json.dumps(body)

    headers = {"content-type": "application/json"}
    json_response = requests.post(model_url, data=body, headers=headers,).json()

    predictions_list = json_response["predictions"][0]

    total = 0
    for idx, pred_val in enumerate(predictions_list):
        total += pred_val * (idx + 1)

    return round(total, 1)


def _get_image_scores(image):
    print("Getting scores...")
    aesthetic_url = "http://0.0.0.0:59101/v1/models/nima-mobilenet-aesthetic:predict"
    technical_url = "http://0.0.0.0:59101/v1/models/nima-mobilenet-technical:predict"
    print("\tDone.")

    return (
        str(_get_image_score(image, aesthetic_url)),
        str(_get_image_score(image, technical_url)),
    )


def _create_effect_collages(images):
    image_resize = (500, 500)
    image_padding = (50, 100)
    transformations = {
        "brightness": _brightness,
        "contrast": _contrast,
        "saturation": _saturation,
        "sharpness": _sharpness,
    }
    transformation_keys = ["brightness", "contrast", "saturation", "sharpness"]
    effects = ["decrease", "increase"]
    effected_images = []
    for image_path in images:
        print("Processing <{}> for all effects.".format(image_path))
        original_image = Image.open(image_path)
        original_image.thumbnail(image_resize)
        image_transformed = []
        for transformation_key in transformation_keys:
            print("\t<{}>".format(transformation_key))
            transformed_images = [
                transformations[transformation_key](
                    effect_direction, original_image.copy()
                )
                for effect_direction in effects
            ]
            transformed_images.insert(1, original_image.copy())
            image_transformed.append(transformed_images)
        effected_images.append(image_transformed)

    image_collages = []
    print("Creating image collages")
    for effected_image_set in effected_images:
        original_image_size = effected_image_set[0][1].size
        new_collage_size = (
            (original_image_size[0] * (len(effected_image_set) + 1))
            + ((len(effected_image_set) + 1) * image_padding[0])
            + image_padding[0],
            (original_image_size[1] * (len(effected_image_set[0]) + 1))
            + ((len(effected_image_set[0]) + 1) * image_padding[1]),
        )
        collage = Image.new("RGB", new_collage_size, (255, 255, 255))

        for idx_x, effected_image in enumerate(effected_image_set):
            for idx_y, image in enumerate(effected_image):
                position = (
                    ((idx_x + 1) * image_padding[0])
                    + ((idx_x + 1) * original_image_size[0]),
                    ((idx_y + 1) * image_padding[1])
                    + ((idx_y + 1) * original_image_size[1]),
                )
                collage.paste(image, position)

                aesthetic_score, technical_score = _get_image_scores(image.copy())
                font = ImageFont.truetype(
                    os.path.join("fonts", "open-sans", "OpenSans-Regular.ttf"), 30
                )
                draw = ImageDraw.Draw(collage)
                draw.text(
                    (
                        ((idx_x + 1) * original_image_size[0])
                        + ((idx_x + 1) * image_padding[0]),
                        ((idx_y + 2) * original_image_size[1])
                        + ((idx_y + 1) * image_padding[1]),
                    ),
                    "Aesthetic: "
                    + aesthetic_score
                    + " , Technical: "
                    + technical_score,
                    (0, 0, 0),
                    font=font,
                )

        # Add the text
        font = ImageFont.truetype(
            os.path.join("fonts", "open-sans", "OpenSans-Regular.ttf"), 50
        )
        draw = ImageDraw.Draw(collage)

        for idx, effect_direction in enumerate(
            ["Effect decrease", "Original Image", "Effect increase"]
        ):
            draw.text(
                (
                    100,
                    (original_image_size[1] * (idx + 2))
                    - (original_image_size[1] / 2.0)
                    + (image_padding[1] * idx),
                ),
                effect_direction,
                (0, 0, 0),
                font=font,
            )

        for idx, effect_key in enumerate(transformation_keys):
            draw.text(
                (
                    (
                        image_padding[0] * (idx + 2)
                        + (original_image_size[0] * (idx + 1))
                    ),
                    original_image_size[1] - (image_padding[1] / 2.0),
                ),
                effect_key.capitalize(),
                (0, 0, 0),
                font=font,
            )

        collage.thumbnail((1000, 1000))
        image_collages.append(collage)
    return image_collages


def _create_crop_collages(images):
    image_resize = (500, 500)
    image_padding = (50, 150)
    location_mapper = {
        "tl": "Top-left",
        "tm": "Top-center",
        "tr": "Top-right",
        "lm": "Center-left",
        "mm": "Center-center",
        "rm": "Center-right",
        "bl": "Bottom-left",
        "bm": "Bottom-center",
        "br": "Bottom-right",
    }
    collages = []
    for image_path in images:
        print("Processing <{}> for cropping.".format(image_path))
        original_image = Image.open(image_path)
        original_image.thumbnail(image_resize)
        original_image_size = original_image.size
        crop_results = _do_crop(original_image.copy())

        new_collage_size = (
            (4 * original_image_size[0] + (3 * image_padding[0])),
            (3 * original_image_size[1]) + (2 * image_padding[1]),
        )
        collage = Image.new("RGB", new_collage_size, (255, 255, 255))

        aesthetic_scores = [
            ["Original Image", original_image, _get_image_scores(original_image)[0]],
            ["Best crop", crop_results[0]["best"]["image"], crop_results[0]["best"]["score"]],
            ["Worst crop", crop_results[0]["worst"]["image"], crop_results[0]["worst"]["score"]]
        ]
        aesthetic_scores = sorted(aesthetic_scores, key=lambda x: float(x[2]))

        technical_scores = [
            ["Original Image", original_image, _get_image_scores(original_image)[1]],
            ["Best crop", crop_results[1]["best"]["image"], crop_results[1]["best"]["score"]],
            ["Worst crop", crop_results[1]["worst"]["image"], crop_results[1]["worst"]["score"]]
        ]
        technical_scores = sorted(technical_scores, key=lambda x: float(x[2]))

        font = ImageFont.truetype(
            os.path.join("fonts", "open-sans", "OpenSans-Regular.ttf"), 30
        )
        draw = ImageDraw.Draw(collage)

        for idx, image in enumerate(aesthetic_scores):
            title, image, score = image
            collage.paste(image, (((idx + 1) * original_image_size[0]) + (idx * image_padding[0]),original_image_size[1]))
            draw.text((((idx + 1) * original_image_size[0]) + (idx * image_padding[0]) + 50,
                       (original_image_size[1] * 2) + (0)), "Score: " + str(score), (0, 0, 0), font=font)
            draw.text((((idx + 1) * original_image_size[0]) + (idx * image_padding[0]) + 50,
                       (original_image_size[1] * 1) - (image_padding[1]) + (image_padding[1] / 2.0)), title, (0, 0, 0), font=font)

        for idx, image in enumerate(technical_scores):
            title, image, score = image
            collage.paste(image, (((idx + 1) * original_image_size[0]) + (idx * image_padding[0]),(original_image_size[1] * 2) + image_padding[1]))
            draw.text((((idx + 1) * original_image_size[0]) + (idx * image_padding[0]) + 50,(original_image_size[1] * 3) + (image_padding[1])), "Score: " + str(score), (0,0,0), font=font)
            draw.text((((idx + 1) * original_image_size[0]) + (idx * image_padding[0]) + 50,(original_image_size[1] * 2) + (image_padding[1] / 2.0)), title, (0,0,0), font=font)

        font = ImageFont.truetype(
            os.path.join("fonts", "open-sans", "OpenSans-Regular.ttf"), 50
        )

        draw.text(
            (50, (2 * original_image_size[1]) - (original_image_size[1] / 2.0)),
            "Aesthetic scores",
            (0, 0, 0),
            font=font,
        )
        draw.text(
            (
                50,
                (3 * original_image_size[1])
                - (original_image_size[1] / 2.0)
                + image_padding[1],
            ),
            "Technical scores",
            (0, 0, 0),
            font=font,
        )

        collage.thumbnail((1000, 1000))
        collages.append(collage)
    return collages


def _do_crop(image):
    orig = image
    orig_size = orig.size

    inset_amount = 0.2

    x0 = 0
    xil = orig.size[0] * (0 + inset_amount)
    xir = orig.size[0] * (1 - inset_amount)
    x100 = orig.size[0]

    y0 = 0
    yit = orig.size[1] * (0 + inset_amount)
    yib = orig.size[1] * (1 - inset_amount)
    y100 = orig.size[1]

    crop_locations = {
        "tl": (x0, y0, xir, yib),
        "tm": (xil, y0, xir, yib),
        "tr": (xil, y0, x100, yib),
        "lm": (x0, yit, xir, yib),
        "mm": (xil, yit, xir, yib),
        "rm": (xil, yit, x100, yib),
        "bl": (x0, yit, xir, y100),
        "bm": (xil, yit, xir, y100),
        "br": (xil, yit, x100, y100),
    }

    cropped_images = {
        "tl": orig.crop(crop_locations["tl"]),
        "tm": orig.crop(crop_locations["tm"]),
        "tr": orig.crop(crop_locations["tr"]),
        "lm": orig.crop(crop_locations["tm"]),
        "mm": orig.crop(crop_locations["mm"]),
        "rm": orig.crop(crop_locations["rm"]),
        "bl": orig.crop(crop_locations["bl"]),
        "bm": orig.crop(crop_locations["bm"]),
        "br": orig.crop(crop_locations["br"]),
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

    best_aesthetic_key = "tl"
    worst_aesthetic_key = "tl"
    best_technical_key = "tl"
    worst_technical_key = "tl"
    for key in cropped_images.keys():
        scores = _get_image_scores(cropped_images[key])
        cropped_scores[key] = scores

        if scores[0] > cropped_scores[best_aesthetic_key][0]:
            best_aesthetic_key = key
        if scores[0] < cropped_scores[worst_aesthetic_key][0]:
            worst_aesthetic_key = key
        if scores[1] > cropped_scores[best_technical_key][1]:
            best_technical_key = key
        if scores[1] < cropped_scores[worst_technical_key][1]:
            worst_technical_key = key

    aesthetic = {
        "best": {
            "location": best_aesthetic_key,
            "image": _place_cropped_image(cropped_images[best_aesthetic_key], orig_size, crop_locations[best_aesthetic_key]),
            "score": cropped_scores[best_aesthetic_key][0],
        },
        "worst": {
            "location": worst_aesthetic_key,
            "image": _place_cropped_image(cropped_images[worst_aesthetic_key],orig_size, crop_locations[worst_aesthetic_key]),
            "score": cropped_scores[worst_aesthetic_key][0],
        },
    }
    technical = {
        "best": {
            "location": best_technical_key,
            "image": _place_cropped_image(cropped_images[best_technical_key],orig_size, crop_locations[best_technical_key]),
            "score": cropped_scores[best_technical_key][1],
        },
        "worst": {
            "location": worst_technical_key,
            "image": _place_cropped_image(cropped_images[worst_technical_key],orig_size, crop_locations[worst_technical_key]),
            "score": cropped_scores[worst_technical_key][1],
        },
    }

    return aesthetic, technical


def _place_cropped_image(cropped_image, size, location):
    base = Image.new("RGB", size, (200,200,200))
    base.paste(cropped_image, (int(location[0]), int(location[1])))
    return base


def _main():
    out_dir = "out"
    gallery_dir = "effect_gallery"
    crop_dir = "crop_effect"
    all_effect_dir = "all_effect"
    images_dir = "in"
    images_dir = "images"

    images = _get_image_paths(images_dir)

    all_effect_collages = _create_effect_collages(images)
    for idx, collage in enumerate(all_effect_collages):
        collage.save(os.path.join(out_dir, gallery_dir, str(idx) + ".png"))

    crop_collages = _create_crop_collages(images)
    for idx, collage in enumerate(crop_collages):
        collage.save(os.path.join(out_dir, crop_dir, str(idx) + ".png"))


if __name__ == "__main__":
    _main()
