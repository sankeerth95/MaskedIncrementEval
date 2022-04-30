import torch
import torch.nn.functional as F
import unittest
from benchmark.input_handlers import DatasetInputHandler, IncrDatasetInputHandler_2args

from benchmark.model_getter import ModelGetter
from benchmark.model_handlers import IncrModelHandler, ModelHandler

class TestNetwork(unittest.TestCase):


    def test_incr(self):

        input_h = IncrDatasetInputHandler_2args(20)

        model = ModelGetter.get_e2vid_incr_model(None)
        model_h = IncrModelHandler(model, input_h.prev_x)

        c,h = model_h.run_once(input_h.get_single_sample(0))
        c,h = torch.zeros_like(c), torch.zeros_like(h)
        for i in range(10):
            c_incr, h_incr += model_h.run_once(input_h.get_single_sample(i))
            c += c_incr
            h += h_incr
        

        model_baseline = ModelGetter.get_e2vid_model(None)
        model_baseline_h = ModelHandler(model_baseline, input_h.prev_x)
        c_base,h_base = model_baseline_h.run_once(input_h.prev_x)


        diff_count = lambda x: float( x.count_nonzero().cpu().numpy() )        
        self.assertEqual(diff_count(c- c_base)    , 0., "check c")
        self.assertEqual(diff_count(h - h_base), 0., "check h")


