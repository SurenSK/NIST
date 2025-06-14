"""
* Last Update: Jun-06-2025
* Description: This script validates T2I Generator's .xml submission file against t2i_GeneratorResult.dtd
* Author: Seungmin Seo
"""

import xml.etree.ElementTree as ET
import argparse
from typing import Tuple, Dict, List
import subprocess
import sys
import os
import re
from PIL import Image

# import piexif


def insert_dtd_path(xml_path: str, dtd_path: str):
    dtd_path = os.path.abspath(dtd_path)
    with open(xml_path, "r+") as file:
        content = file.read()
        content = re.sub(r'SYSTEM\s+".*?\.dtd"', f'SYSTEM "{dtd_path}"', content)

        file.seek(0)
        file.write(content)
        file.truncate()


def run_xmllint(xml: str):
    validation_result = subprocess.run(
        ["xmllint", "--noout", "--valid", xml],
        capture_output=True,
        text=True,
    )
    if validation_result.returncode != 0:
        print(f"Validation Failed: {validation_result.stderr}")
        return False

    return True


def validate_structure(args: argparse.Namespace) -> str:
    if not os.path.isdir(args.submission):
        print(f"Error: '{args.submission}' is not a valid directory.")
        return None

    items = os.listdir(args.submission)

    xml_files = [f for f in items if f.endswith(".xml")]
    if len(xml_files) != 1:
        print(f"Error: Expected 1 XML file, but found {len(xml_files)}.")
        return None

    required_folders = {"images_generated", "images_prompts"}
    existing_folders = {
        f for f in items if os.path.isdir(os.path.join(args.submission, f))
    }

    if required_folders != existing_folders:
        print(
            f"Error: Expected folders {required_folders}, but found {existing_folders}."
        )
        return None

    xml_file = os.path.join(args.submission, xml_files[0])
    insert_dtd_path(xml_file, args.dtdpath)

    return xml_file


def parse_topicxml(topic_path: str) -> Dict[str, Dict[str, str]]:
    tree = ET.parse(topic_path)
    root = tree.getroot()

    topic_dict = {}

    for topic in root.findall("topic"):
        num = topic.find("num").text.strip() if topic.find("num") is not None else None
        title = (
            topic.find("title").text.strip()
            if topic.find("title") is not None
            else None
        )
        prompt = (
            topic.find("prompt").text.strip()
            if topic.find("prompt") is not None
            else None
        )

        if num:
            topic_dict[num] = {"title": title, "prompt": prompt}

    return topic_dict


def validate_contents(
    submission: str, topic_dict: Dict[str, Dict[str, str]]
) -> Tuple[int, List[str], List[str]]:
    tree = ET.parse(submission)
    root = tree.getroot()

    valid_img_filename = []
    used_image_prompt_count = 0
    filename_pattern = re.compile(
        r"^topic\.(1[0-4][0-9]|150|[1-9][0-9]?|0)\.image\.(10|[1-9])\.webp$",
        re.IGNORECASE,
    )

    valid_prompt_filenames = []

    for topic_result in root.findall("GeneratorRunResult/GeneratorTopicResult"):
        topic_id = topic_result.get("topic")
        used_image_prompts = topic_result.get("usedImagePrompts")
        if used_image_prompts == "T":
            used_image_prompt_count += 1

            topic_num = topic_id.split("_")[1]
            valid_prompt_filenames.extend(
                [f"topic.{topic_num}.prompt.1.webp", f"topic.{topic_num}.prompt.2.webp"]
            )

        topic_valid = False
        images = topic_result.findall("Image")
        image_count = len(images)

        if image_count > 10:
            print(
                f"Validation failed: Too many images for topic {topic_id}. Found {image_count}, maximum allowed is 10."
            )
            exit(1)

        for image in images:
            filename = image.get("filename")
            prompt = image.get("prompt")
            nist_prompt = image.get("NIST-prompt")

            if filename_pattern.match(filename):
                valid_img_filename.append(filename)
            else:
                print(f"Validation failed: Invalid filename format: {filename}")
                exit(1)

            if (
                nist_prompt == "T"
                and topic_id in topic_dict
                and prompt == topic_dict[topic_id]["prompt"]
            ):
                topic_valid = True

        if topic_id in topic_dict and not topic_valid:
            print(
                f"Validation failed: Topic {topic_id} does not contain the expected prompt."
            )
            exit(1)

    return used_image_prompt_count, valid_img_filename, valid_prompt_filenames


def validate_images(
    submission_dir: str, img_files: List[str], count: int, prompt_valid: List[str]
):
    img_dir = os.path.join(submission_dir, "images_generated")
    prompt_dir = os.path.join(submission_dir, "images_prompts")

    existing_files = set(os.listdir(img_dir))
    missing_files = set(img_files) - existing_files

    if len(existing_files) != len(set(img_files)):
        print(
            f"Validation failed: Extra files found in {img_dir}: {existing_files - set(img_files)}"
        )
        exit(1)

    if missing_files:
        print(f"Validation failed: Missing files in {img_dir}: {missing_files}")
        exit(1)

    prompt_files = set(os.listdir(prompt_dir))
    if not (count <= len(prompt_files) <= count * 2):
        print(
            f"Validation failed: Invalid number of files in {prompt_dir}: {len(prompt_files)} (Expected between {count} and {count * 2})"
        )
        exit(1)

    invalid_prompt_files = prompt_files - set(prompt_valid)
    if invalid_prompt_files:
        print(
            f"Validation failed: Invalid prompt image filename in {prompt_dir}: {invalid_prompt_files}"
        )
        exit(1)

    allowed_resolutions = {
        (1080, 1080),
        (1138, 720),
        (1280, 534),
        (1280, 560),
        (1280, 570),
        (1280, 720),
        (1280, 960),
        (1920, 1080),
        (1920, 804),
        (1920, 816),
        (3840, 2160),
        (600, 480),
        (640, 360),
        (640, 480),
        (720, 540),
        (852, 480),
        (960, 720),
    }

    resolutions = {}

    for file in img_files:
        file = file.strip()
        file_path = os.path.abspath(os.path.join(img_dir, file))

        try:
            # exif_data = piexif.load(file_path)
            # if not exif_data or exif_data == {
            #     "0th": {},
            #     "Exif": {},
            #     "GPS": {},
            #     "Interop": {},
            #     "1st": {},
            #     "thumbnail": None,
            # }:
            #     print(f"Validation failed: EXIF metadata missing for {file}")
            #     exit(1)

            topic_id = file.split(".")[1]
            with Image.open(file_path) as img_pil:
                resolution = img_pil.size

                if resolution not in allowed_resolutions:
                    print(
                        f"Validation failed: {file} has an invalid resolution {resolution}"
                    )
                    exit(1)

                if topic_id not in resolutions:
                    resolutions[topic_id] = set()
                resolutions[topic_id].add(resolution)

        except Exception as e:
            print(f"Validation failed: Error processing {file_path} {e}")
            exit(1)

    for topic, res_set in resolutions.items():
        if len(res_set) != len(
            [f for f in img_files if f.startswith(f"topic.{topic}.image.")]
        ):
            print(
                f"Validation failed: Images under topic {topic} do not have unique resolutions."
            )
            exit(1)

    return True


def validate_t2i_generator(args):
    try:
        submission_xml = validate_structure(args)
        if not submission_xml:
            print(f"Validation failed: Submission directory structure is incorrect")
            exit(1)
        print("Pass validation 1/4: Submission directory structure is correct.")
        print("Passed 1/4".center(50, "="))

        if not run_xmllint(submission_xml):
            exit(1)
        print(f"Pass validation 2/4: XML syntax is validated against DTD.")
        print("Passed 2/4".center(50, "="))

        topic_dict = parse_topicxml(args.topic)
        prompt_count, img_valid, prompt_valid = validate_contents(
            submission_xml, topic_dict
        )
        print(f"Pass validation 3/4: XML contents validated")
        print("Passed 3/4".center(50, "="))

        validate_images(args.submission, img_valid, prompt_count, prompt_valid)
        print("Pass validation 4/4: All files validated")
        print("Passed all".center(50, "="))

    except Exception as e:
        print(e)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="validate contents of submission package which generator submitted"
    )
    parser.add_argument(
        "-s",
        "--submission",
        type=str,
        required=True,
        default="",
        help="path to the submission directory",
    )
    parser.add_argument(
        "-t",
        "--topic",
        type=str,
        required=True,
        default="",
        help="path to the topic file provided to participants",
    )
    parser.add_argument(
        "-d",
        "--dtdpath",
        type=str,
        required=True,
        default="",
        help="absolute path to .dtd file",
    )
    parser.set_defaults(func=validate_t2i_generator)

    args = parser.parse_args()
    if hasattr(args, "func") and args.func:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
