#!/usr/bin/env python
"""
model tests
"""

import sys, os
import unittest
sys.path.insert(1, os.path.join('..', os.getcwd()))

## import model specific functions and variables
from model import *




class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model_train(test=True)
        self.assertTrue(os.path.exists(os.path.join("models", "test.joblib")))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## train the model
        model = model_load(test=True)
        
        self.assertTrue('predict' in dir(model))
        self.assertTrue('forecast' in dir(model))

       
    def test_03_predict(self):
        """
        test the predict function input
        """

        ## load model first
        model = model_load(test=True)
    
        ## ensure that a list can be passed
        result = model_predict(date='2019-07-01', country='united_states', df=None, model=None, test=True)
        y_pred = result['predicted']
        self.assertGreater(len(y_pred),0)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
