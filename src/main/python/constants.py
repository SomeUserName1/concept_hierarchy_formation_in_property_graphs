from enum import Enum
from os import path
import os
import logging


BASE = os.getcwd()
IMG_BASE = BASE + "/doc/img/"
CACHE_PATH = "/tmp/"
RESULT_SUMMARY_PATH = path.join(BASE, "doc", "thesis", 'clustering_survey_results.log')
result_summary = open(RESULT_SUMMARY_PATH, 'w+')


class Dataset(Enum):
    SYNTHETIC = (BASE + "/data/synthetic.json", BASE + "/data/synthetic.tree")
    NOISY_SYNTHETIC = (BASE + "/data/synthetic_noisy.json", BASE + "/data/synthetic.tree")
    YELP = (BASE + "/data/business.json", BASE + "/data/business.tree")


results = {'single': [27, 28, 26, 25], 'rsl': [27, 28, 29, 35], 'hdbscan': [30, 31, 29, 27], 'dbscan': [9, 2, 8, 5],
           'optics': [9, 2, 3, 8], 'affinity_prop': [10, 9, 4, 5], 'spectral': [1, 2, 8, 5], 'kmeans': [5, 5, 8, 5],
           'kmedians': [5, 5, 6, 3], 'kmedoid': [7, 7, 10, 6], 'rock': [0, 2, 7, 5], 'bsas': [1, 5, 7, 6],
           'mbsas': [1, 5, 7, 6], 'ttsas': [0, 2, 7, 5], 'som': [5, 1, 6, 6], 'trestle': [0, 1, 8, 7]}

times = {'single': [0.000783, 0.029643], 'rsl': [0.00427, 0.058494], 'hdbscan': [0.027989, 0.00873947],
         'dbscan': [0.1537, 0.227691], 'optics': [0.17996, 4.80953], 'affinity_prop': [0.146596, 22.96246],
         'spectral': [0.25918, 1.89046], 'kmeans': [0.0664, 43.374066], 'kmedians': [0.0631, 55.23311],
         'kmedoid': [0.0711, 0.141862], 'rock': [0.08715, 132.5041620], 'bsas': [0.06154, 0.04385],
         'mbsas': [0.06174, 0.043392], 'ttsas': [0.06538, 0.02016234], 'som': [0.05452, 0.49852],
         'trestle': [0.098921, 0.58683]}

logger = logging.getLogger("clustering_survey")
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler(path.join(BASE, "doc", "clustering_survey.log"), mode='w+')
fh.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)

# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s: %(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add the handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)
