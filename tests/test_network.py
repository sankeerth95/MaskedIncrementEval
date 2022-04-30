import torch
import torch.nn.functional as F
import unittest

class TestNetwork(unittest.TestCase):


    def test_incr(self):



        





        X = torch.randn(shape).cuda()
        X1 = X.detach().clone()
        incr = torch.randn(shape).cuda()
        zz = torch.relu(X+incr) - torch.relu(X)

        diff_count = lambda x: float( x.count_nonzero().cpu().numpy() )        
        self.assertEqual(diff_count(out- zz)    , 0., "output check")
        self.assertEqual(diff_count(X1+incr - X), 0., "incremented_input")


