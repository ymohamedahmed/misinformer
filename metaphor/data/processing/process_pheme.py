import sys
import os
from typing import Dict
import json
from pathlib import Path, PosixPath
import csv

"""
Directory structure for each tweet: 
    - annotations.json
    - source-tweets/X.json
"""


def process_annotation(annotation: Dict[str, str], string=True):
    # taken from https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078?file=11767946
    if "misinformation" in annotation.keys() and "true" in annotation.keys():
        if int(annotation["misinformation"]) == 0 and int(annotation["true"]) == 0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation["misinformation"]) == 0 and int(annotation["true"]) == 1:
            if string:
                label = "true"
            else:
                label = 1
        elif int(annotation["misinformation"]) == 1 and int(annotation["true"]) == 0:
            if string:
                label = "false"
            else:
                label = 0
        elif int(annotation["misinformation"]) == 1 and int(annotation["true"]) == 1:
            print("OMG! They both are 1!")
            print(annotation["misinformation"])
            print(annotation["true"])
            label = None

    elif "misinformation" in annotation.keys() and "true" not in annotation.keys():
        # all instances have misinfo label but don't have true label
        if int(annotation["misinformation"]) == 0:
            if string:
                label = "unverified"
            else:
                label = 2
        elif int(annotation["misinformation"]) == 1:
            if string:
                label = "false"
            else:
                label = 0

    elif "true" in annotation.keys() and "misinformation" not in annotation.keys():
        print("Has true not misinformation")
        label = None
    else:
        print("No annotations")
        label = None

    return label
    """
    if "misinformation" in annotation.keys() and "true" in annotation.keys():
        if int(annotation["misinformation"]) == 0 and int(annotation["true"]) == 0:
            label = "unverified"
        elif int(annotation["misinformation"]) == 0 and int(annotation["true"]) == 1:
            label = "true"
        elif int(annotation["misinformation"]) == 1 and int(annotation["true"]) == 0:
            label = "false"
        elif int(annotation["misinformation"]) == 1 and int(annotation["true"]) == 1:
            label = None

    elif "misinformation" in annotation.keys() and "true" not in annotation.keys():
        # all instances have misinfo label but don't have true label
        if int(annotation["misinformation"]) == 0:
            label = "unverified"
        elif int(annotation["misinformation"]) == 1:
            label = "false"

    elif "true" in annotation.keys() and "misinformation" not in annotation.keys():
        print("Has true not misinformation")
        label = None
    else:
        print("No annotations")
        label = None

    return label
    """


def get_tweet_information(annotation_path: PosixPath):
    # example path: all-rnr-annotated-threads/gurlitt-all-rnr-threads/non-rumours/536805993334571008/annotation.json
    # associated source-tweet is: all-rnr-annotated-threads/gurlitt-all-rnr-threads/non-rumours/536805993334571008/source-tweets/536805993334571008.json
    split_path = str(annotation_path).split("/")
    tweet_id = split_path[-2]
    source_tweet_path = os.path.join(
        "/".join(split_path[:-1]), "source-tweets", f"{tweet_id}.json"
    )
    topic = split_path[-4].replace("-all-rnr-threads", "")
    text, timestamp, veracity = None, None, None
    with open(source_tweet_path) as source_tweet:
        source_tweet = json.load(source_tweet)
        text = source_tweet["text"]
        timestamp = source_tweet["created_at"]
    text = text.replace("\n", " ")

    with open(annotation_path) as annotation:
        annotation = json.load(annotation)
        veracity = process_annotation(annotation)

    return [tweet_id, timestamp, topic, text, veracity]


def get_all_annotation_paths(root: str):
    return [p for p in Path(root).rglob("annotation.json")]


def main(root, output_path):
    output_file = open(output_path, "w")
    output_writer = csv.writer(output_file)
    output_writer.writerow(["tweet_id", "timestamp", "topic", "text", "veracity"])
    count = 0
    none_count = 0
    for ann_path in get_all_annotation_paths(root):
        info = get_tweet_information(ann_path)
        if info[4] is None:
            none_count += 1
        count += 1
        output_writer.writerow(info)
    output_file.close()

    print(f"None label percentage: {100*none_count/count}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
