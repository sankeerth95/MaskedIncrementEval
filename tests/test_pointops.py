import torch
import torch.nn.functional as F
import ext.pointops.pointops_modules as po
import ext.pointops.pointops_functional as pf
import unittest

from incr_modules.mask_incr_modules import nnConvIncr

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


    def test_convkxkincr_functional(self):

        in_shape = [1, 3, 25, 25]
        c = nnConvIncr(3, 5, 3, 1, 1, bias=False).to(device='cuda').eval()

        x = torch.randn(in_shape, device='cuda').to(memory_format=torch.channels_last)
        out_base = c.forward_refresh_reservoirs(x.contiguous())
        out = c((x, torch.ones_like(x, dtype=bool)))
        diff_count = lambda var: float( var.count_nonzero().cpu().numpy() )
        max_val = lambda var: float( torch.abs(var).max().cpu().numpy() )
        self.assertEqual(diff_count(out_base - out[0])    , 0., "output check")



