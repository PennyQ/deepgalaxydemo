from django.test import TestCase

from apps.ml.classifier.deepgalaxy import DeepGalaxyClassifier
# from apps.ml.classifier.cvae import *


class MLTests(TestCase):
    def test_et_algorithm(self):      
        my_alg = DeepGalaxyClassifier()
        response = my_alg.compute_prediction('/Users/pennyqxr/Code/deepgalaxydemo/backend/server/media/images/*.jpg')
        print("------------response---------", response)
        self.assertTrue('OK' in response)
        # self.assertTrue('label' in response)
        # self.assertEqual('<=50K', response['label'])