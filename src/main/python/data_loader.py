from typing import List
from subprocess import Popen
import random
import json
from shlex import split

from src_project.main.python import BASE, Dataset


def generate_synthetic(depth: int, width: int = 2, iteration: int = 10, m_path: str = BASE + "/data/",
                       rem_labels: int = 1, add_labels: int = 1, alter_labels: int = 1, prob: float = 0.2):
    command = "java -jar " + BASE + "/lib/synthetic_data_generator.jar -p '" + m_path + "' -d " + str(depth) + " -w " \
              + str(width) + " -i " + str(iteration) + " -n " + str(rem_labels) + " " + str(add_labels) + " " + \
              str(alter_labels) + " -pr " + str(prob)
    args = split(command)
    Popen(args).wait()


def sample_file(file, n_samples):
    sample = []
    for i, line in enumerate(file):
        if i < n_samples:
            sample.append(json.loads(line))
        elif i >= n_samples and random.random() < n_samples / float(i + 1):
            replace = random.randint(0, len(sample) - 1)
            sample[replace] = json.loads(line)
    return sample


def sample_yelp(n_samples: int) -> List[List[str]]:
    with open(Dataset.YELP.value[0], "r") as read_file:
        sample = sample_file(read_file, n_samples)
    labels = []
    for entry in sample:
        if entry['categories'] is not None:
            labels.append(entry['categories'])
        else:
            labels.append('none')

    return labels


def open_synthetic(noisy: int):
    m_path = Dataset.SYNTHETIC.value[0] if noisy is 0 else Dataset.NOISY_SYNTHETIC.value[0]
    with open(m_path, "r") as read_file:
        data = json.load(read_file)

    labels = [entry['labels'] for entry in data]
    return labels


def load(n_samples: int, dataset: Dataset, noise: int, width: int = 3, depth: int = 2) -> List[List[str]]:
    if dataset == Dataset.SYNTHETIC or dataset == Dataset.NOISY_SYNTHETIC:
        iteration = int(n_samples / (width ** depth))
        iteration = iteration if iteration > 0 else 1
        generate_synthetic(width=width, depth=depth, iteration=iteration, prob=noise)
        return open_synthetic(noise)
    else:
        return sample_yelp(n_samples)


def preprocess_trestle_yelp(n_samples):
    with open(Dataset.YELP.value[0], "r") as read_file:
        sample = sample_file(read_file, n_samples)

        data = []
        for entry in sample:
            data_entry = {}
            for i, label in enumerate(entry["categories"].split(',')):
                data_entry["?label_" + str(i)] = label

            data.append(data_entry)
            del entry

    return data


def preprocess_trestle_synthetic(dataset):
    with open(dataset.value[0], "r") as read_file:
        data = json.load(read_file)

    for entry in data:
        del entry["id"]
        for i, label in enumerate(entry["labels"].split(',')):
            entry["?label" + str(i)] = label

        del entry["labels"]

    return data
