from django.test import TestCase

from apps.ml.income_classifier.random_forest import RandomForestClassifier
import inspect
from apps.ml.registry import MLRegistry
from apps.ml.income_classifier.extra_trees import ExtraTreesClassifier
from apps.ml.income_classifier.deepgalaxy import DeepGalaxyClassifier

import glob
import cv2
import numpy as np
import os
from apps.ml.income_classifier.cvae import *


class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data = {
            "age": 37,
            "workclass": "Private",
            "fnlwgt": 34146,
            "education": "HS-grad",
            "education-num": 9,
            "marital-status": "Married-civ-spouse",
            "occupation": "Craft-repair",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 68,
            "native-country": "United-States"
        }
        my_alg = RandomForestClassifier()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertEqual('<=50K', response['label'])

    def test_et_algorithm(self):
        # img_list = glob.glob('media/images/*.jpg')
        # print(img_list)
        # img_size = 512

        # obs_imgs = []
        # obs_imgs_title = []
        # img_name = img_list[0]

        # im = cv2.imread(img_name)
        # im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # im_resized = cv2.resize(im_grey, (img_size,img_size))
        # print(img_name, im_resized.shape)

        # obs_imgs.append((im_resized.reshape(img_size,img_size,1)/255).astype(np.float32))
        # obs_imgs_title.append(os.path.basename(img_name))
        # print(obs_imgs_title)
        # input_data = np.array(obs_imgs)
        
        my_alg = DeepGalaxyClassifier()
        response = my_alg.compute_prediction('/Users/pennyqxr/Code/deepgalaxydemo/backend/server/media/images/*.jpg')
        print("------------response---------", response)
        self.assertTrue('OK' in response)
        # self.assertTrue('label' in response)
        # self.assertEqual('<=50K', response['label'])

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "income_classifier"
        algorithm_object = RandomForestClassifier()
        algorithm_name = "random forest"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Piotr"
        algorithm_description = "Random Forest with simple pre- and post-processing"
        algorithm_code = inspect.getsource(RandomForestClassifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)
