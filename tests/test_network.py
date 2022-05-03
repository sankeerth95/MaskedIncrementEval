import torch
import unittest
from benchmark.input_handlers import IncrDatasetInputHandler_2args
from benchmark.model_getter import ModelGetter
from benchmark.model_handlers import BaselineModelhandler, IncrModelHandler



class TestNetwork(unittest.TestCase):


    def test_incr(self):

        device = 'cuda'
        input_h = IncrDatasetInputHandler_2args(20, device=device)

        model = ModelGetter.get_e2vid_incr_model(None).to(device)
        model_h = IncrModelHandler(model, input_h.prev_x)

        c,h = model_h.run_oncel(input_h.get_single_sample(0))

        c = torch.zeros_like(c)#, torch.zeros_like(h)
        c_incr = torch.zeros_like(c) #, torch.zeros_like(h)

        for i in range(10):
            c_incr, h_incr = model_h.run_once(input_h.get_single_sample(i))
            c += c_incr
            # h += h_incr

        model_baseline = ModelGetter.get_e2vid_model(None).to(device)
        model_baseline_h = BaselineModelhandler(model_baseline)
        c_base,h_base = model_baseline_h.run_once(input_h.prev_x)

        diff_count = lambda x: float( torch.abs(x).count_nonzero().detach().cpu().numpy() )
        max_diff   = lambda x: float( torch.abs(x).max().detach().cpu().numpy() )

        print(max_diff(c - c_base))
        print(diff_count(c - c_base))


