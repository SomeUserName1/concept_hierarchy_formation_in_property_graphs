from shlex import split
from subprocess import Popen, PIPE
import logging
import numpy as np

from constants import Dataset, BASE, result_summary, logger


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

    count = 0
    while result[fix + len(str(elem))].isdigit():
        count += 1
        fix = result.find("{" + str(elem)) + 1

    return fix


def create_bracket_tree_rsl(children: np.array, n_clusters):
    logger.info(children)
    children = children.astype(int)
    result = ""
    i = n_clusters - 1
    prev_dif = 0
    for merge in children:
        temp = result
        # if both clusters are not base clusters
        if merge[0] > n_clusters - 1 and merge[1] > n_clusters - 1:
            # finds the place in the array where {merge[0] is
            fix_1 = find_fix(result, merge[0])
            # finds the corresponding closing bracket
            matching_bracket_1 = find_matching_brack(result, fix_1)
            # finds the place in the array where {merge[1] is
            fix_2 = find_fix(result, merge[1])
            # finds the corresponding closing bracket
            matching_bracket_2 = find_matching_brack(result, fix_2)
            temp = result
            i += 1

            # merge was with same distance and consecutive => combine last with this merge
            if (merge[0] == i - 1 or merge[1] == i - 1) and prev_dif == merge[2]:
                smaller = fix_1 if fix_1 < fix_2 else fix_2
                smaller_matching_brack = matching_bracket_1 if smaller == fix_1 else matching_bracket_2
                larger = fix_2 if smaller == fix_1 else fix_1
                larger_matching_brack = matching_bracket_2 if smaller == fix_1 else matching_bracket_1

                result = result[:smaller] + str(i) + result[smaller + length(smaller, temp): smaller_matching_brack] \
                         + result[larger - 1: larger_matching_brack + 1] + temp[smaller_matching_brack: larger - 1] \
                         + temp[larger_matching_brack + 1:]

            else:
                smaller = fix_1 if fix_1 < fix_2 else fix_2
                smaller_matching_brack = matching_bracket_1 if smaller == fix_1 else matching_bracket_2
                larger = fix_2 if smaller == fix_1 else fix_1
                larger_matching_brack = matching_bracket_2 if smaller == fix_1 else matching_bracket_1

                result = result[: smaller - 1] + "{" + str(i) + result[smaller - 1: smaller_matching_brack + 1] \
                         + result[larger - 1: larger_matching_brack + 1] + "}" \
                         + temp[smaller_matching_brack + 1: larger - 1] + temp[larger_matching_brack + 1:]

            prev_dif = merge[2]

        # if one of the clusters merged is not a base cluster
        elif merge[0] > n_clusters - 1 or merge[1] > n_clusters - 1:
            fix_point = find_fix(result, merge[0]) if merge[0] > n_clusters - 1 \
                else find_fix(result, merge[1])

            matching_bracket = find_matching_brack(result, fix_point)
            base_clust = merge[0] if merge[0] < n_clusters else merge[1]

            i += 1

            # merge was with same distance and consecutive => combine last with this merge
            if (merge[0] == i - 1 or merge[1] == i - 1) and prev_dif == merge[2]:
                result = result[:fix_point] + str(i) + result[fix_point + len(str(i - 1)): matching_bracket] + "{" + \
                         str(base_clust) + "}" + result[matching_bracket:]
            # merge was not consecutive, insert at right position
            else:
                result = result[:fix_point - 1] + "{" + str(i) + result[fix_point - 1: matching_bracket + 1] + "{" + \
                         str(base_clust) + "}}" + result[matching_bracket + 1:]
            prev_dif = merge[2]
        # Two of the base clusters are merged, declare a "new" intermediate one for the merge
        else:
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


def length(place, string):
    count = 0
    while string[place + count].isdigit():
        count += 1
    return count
