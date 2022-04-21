import torch.nn as nn

class StoreIntermediateTensors:

    def __init__(self, modules: nn.Module):
        self.modules = modules
        self.store_tensors = {}
        for module in self.modules:
            self.store_tensors[module] = []

        self.hooks = {}

    def __del__(self):
        self.deregister_hooks()
        self.clear_tensors()

    def register_hooks(self):
        for module in self.modules:
            self.hooks[module] = module.register_forward_hook(self.load_tensors)
    
    def view_hooks(self):
        print(self.hooks)

    def deregister_hooks(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}

    def load_tensors(self, module, input, output):
        self.store_tensors[module].append(input)

    def clear_tensors(self):
        for module in self.modules:
            self.store_tensors[module] = []




def report_difference_histograms():
    return NotImplementedError



if __name__ == "__main__":
    import torch
    evflownet = torch.load('../pretrained_models/evflownet_model')
    modules = [
        evflownet.resnet_block[0].res_block[0],
        evflownet.resnet_block[0].res_block[1]
    ]
    su = StoreIntermediateTensors(modules)





