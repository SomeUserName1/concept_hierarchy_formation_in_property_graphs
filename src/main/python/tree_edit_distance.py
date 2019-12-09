from shlex import split
from subprocess import Popen, PIPE
import logging
import numpy as np

from src.main.python import Dataset, BASE, result_summary, logger


# FIXME got error here recently, something broken maybe?
def compute_ted(children_or_tree, name, noise, seconds, samples, dataset: Dataset, trestle: bool = False, n_clusters=0):
    with open(dataset.value[1], "r") as read_file:
        ground_truth = read_file.read()

    if trestle:
        result = create_bracket_tree_trestle(children_or_tree)
    else:
        result = create_bracket_tree_rsl(children_or_tree, n_clusters)

    command = "java -jar " + BASE + "/lib/apted.jar -t " + ground_truth + " " + result
    args = split(command)
    with Popen(args, stdout=PIPE) as apted:
        apted = apted.stdout.read().decode("utf-8").rstrip()
        logger.info("Tree Edit Distance: " + apted)
        result_summary.write(str(dataset) + ", " + name + ", " + str(noise) + ", " + str(apted) + ", " + str(samples)
                             + ", " + str(seconds) + "\n")


def find_matching_brack(res_string, fix_point_idx):
    i = fix_point_idx
    br_count = 1
    while True:
        if res_string[i] == "{":
            br_count += 1
        elif res_string[i] == "}":
            br_count -= 1

        if br_count == 0:
            return i
        else:
            i += 1


def find_fix(result, elem):
    fix = result.find("{" + str(elem)) + 1

    while result[fix + len(str(elem))].isdigit():
        fix = result.find("{" + str(elem)) + 1

    return fix


def create_bracket_tree_rsl(children: np.array, n_clusters):
    logger.info(children)
    children = children.astype(int)
    result = ""
    i = n_clusters - 1
    prev_dif = 0
    for merge in children:
        if merge[0] > n_clusters - 1 and merge[1] > n_clusters - 1:
            fix_1 = find_fix(result, merge[0])
            matching_bracket_1 = find_matching_brack(result, fix_1)
            fix_2 = find_fix(result, merge[1])
            matching_bracket_2 = find_matching_brack(result, fix_2)

            temp = result
            i += 1

            if (merge[0] == i - 1 or merge[1] == i - 1) and prev_dif == merge[2]:
                if fix_1 < fix_2:
                    result = result[:fix_1] + str(i) + result[fix_1 + len(str(i - 1)): matching_bracket_1] \
                             + result[fix_2 - 1: matching_bracket_2 + 1] + temp[matching_bracket_1: fix_2 - 1] \
                             + temp[matching_bracket_2 + 1:]

                else:
                    result = result[:fix_2] + str(i) + result[fix_2 + len(str(i - 1)): matching_bracket_2] \
                             + result[fix_1 - 1: matching_bracket_1 + 1] + temp[matching_bracket_2: fix_1 - 1] \
                             + temp[matching_bracket_1 + 1:]

            else:
                if fix_1 < fix_2:
                    result = result[0: fix_1 - 1] + "{" + str(i) + result[fix_1 - 1: matching_bracket_1 + 1] \
                             + result[fix_2 - 1: matching_bracket_2 + 1] + "}" \
                             + temp[matching_bracket_1 + 1: fix_2 - 1] + temp[matching_bracket_2 + 1:]

                else:
                    result = result[0: fix_2 - 1] + "{" + str(i) + result[fix_2 - 1: matching_bracket_2 + 1] \
                             + result[fix_1 - 1: matching_bracket_1 + 1] + "}" \
                             + temp[matching_bracket_2 + 1: fix_1 - 1] + temp[matching_bracket_1 + 1:]
            prev_dif = merge[2]

        elif merge[0] > n_clusters - 1 or merge[1] > n_clusters - 1:

            fix_point = find_fix(result, merge[0]) if merge[0] > n_clusters - 1 \
                else find_fix(result, merge[1])

            matching_bracket = find_matching_brack(result, fix_point)
            base_clust = merge[0] if merge[0] < n_clusters else merge[1]

            i += 1

            if (merge[0] == i - 1 or merge[1] == i - 1) and prev_dif == merge[2]:
                result = result[:fix_point] + str(i) + result[fix_point + len(str(i - 1)): matching_bracket] + "{" + \
                         str(base_clust) + "}" + result[matching_bracket:]
            else:
                result = result[0: fix_point - 1] + "{" + str(i) + result[fix_point - 1: matching_bracket + 1] + "{" + \
                         str(base_clust) + "}}" + result[matching_bracket + 1:]
            prev_dif = merge[2]
        else:
            # Two of the base clusters are merged, declare a "new" intermediate one for the merged
            i += 1
            result += "{" + str(i) + "{" + str(merge[0]) + "}{" + str(merge[1]) + "}}"
            prev_dif = merge[2]

    logging.info("Cluster result based bracket_tree ")
    logger.info(result)
    return result


def create_bracket_tree_trestle(tree):
    result = ""
    i = 0
    prev_tab_count = -1
    tab_stack = 0

    for elem in tree.__str__().split('\n'):
        cur_tab_count = elem.count('\t')
        if prev_tab_count == cur_tab_count:
            result += "}"
        elif prev_tab_count < cur_tab_count:
            tab_stack += 1
        else:
            delta = prev_tab_count - cur_tab_count
            for _ in range(0, delta + 1):
                result += "}"
            tab_stack -= delta

        result += elem[:elem.find("{")].replace("|-", "{" + str(i)).replace("\t", "")

        prev_tab_count = cur_tab_count

        i += 1
    logger.info("trestle result based bracket tree:")
    logger.info(result)
    return result