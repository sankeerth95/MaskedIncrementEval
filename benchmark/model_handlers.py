from abc import abstractmethod
import torch.nn as nn
from incr_modules.fenced_module_masked import IncrementMaskModule

# solid design principles:
#
# single responsibility: 
# Open to extension, close to modification
# liskov proinciple: subclass b.s
# interfaces: should not have redundant methods;
# dependencies should be passed to it, should not initialize deps in current class 

class ModelHandler:
    def __init__(self, op: nn.Module):
        self.op = op

    @abstractmethod
    def run_once(self, input_sample):
        raise NotImplementedError

class BaselineModelhandler(ModelHandler):
    def __init__(self, op: nn.Module):
        super().__init__(op)
        self.op.eval()

    def run_once(self, input_sample):
        return self.op(input_sample, None)


class IncrModelHandler(ModelHandler):
    def __init__(self, incr_op: IncrementMaskModule, prev_x_input):
        super().__init__(incr_op)
        self.op.eval()
        self.op.forward_refresh_reservoirs(prev_x_input, None)

    def run_once(self, input_sample):
        return self.op(input_sample, None)


