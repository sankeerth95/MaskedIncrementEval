import torch
import torch.nn.functional as F
import ext.pointops.pointops as po
import unittest

class TestPointOps(unittest.TestCase):


    def test_accumulate_incr(self):
        shape = (1, 40, 80)
        
        X = torch.randn(shape).cuda()
        X1 = X.detach().clone()
        incr = torch.randn(shape).cuda()
        zz = torch.relu(X+incr) - torch.relu(X)
        out = po.activation_incr(X, incr)

        diff_count = lambda x: float( x.count_nonzero().cpu().numpy() )        
        self.assertEqual(diff_count(out- zz)    , 0., "output check")
        self.assertEqual(diff_count(X1+incr - X), 0., "incremented_input")



    def test_conv3x3incr(self):
        x = torch.randn(1, 32, 246, 360).cuda()
        w = torch.randn(32, 32, 3, 3).cuda()
        out = po.conv3x3_incr(input, w)

        zz = F.conv2d(x, w)
        diff_count = lambda var: float( var.count_nonzero().cpu().numpy() )
        self.assertEqual(diff_count(out - zz[0])    , 0., "output check")


