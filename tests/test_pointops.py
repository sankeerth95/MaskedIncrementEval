import torch
import torch.nn.functional as F
import ext.pointops.pointops_modules as po
import ext.pointops.pointops_functional as pf
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


    def test_convkxkincr_functional(self):

        in_shape = (1, 128, 132, 176)
        x = torch.randn(in_shape, device='cuda').to(memory_format=torch.channels_last)
        w = torch.randn((256, 128, 3, 3), device='cuda')
        w1 = pf.convert_filter_out_channels_last(w)
        out = pf.functional_conv_module(x, w1, mask=None)

        out_baseline = F.conv2d(x, w)
        diff_count = lambda var: float( var.count_nonzero().cpu().numpy() )
        max_val = lambda var: float( torch.abs(var).max().cpu().numpy() )

        self.assertEqual(diff_count(out - out_baseline)    , 0., "output check")





        # my_conv = po.Conv5x5Incr(in_shape, 1, 1, w)
        # out = my_conv(x)
        # out_fr = my_conv.forward_refresh_reservoir(x)

        for i in range(10):
        # out_func = pf.functional_conv_module(x, w1, mask=None)



