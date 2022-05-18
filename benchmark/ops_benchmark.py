import subprocess
import torch
import torch.nn as nn
from torch.autograd.profiler_util import FunctionEvent
from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm
from benchmark.model_handlers import ModelHandler
from benchmark.input_handlers import InputHandler

class ProfileHandler:
    def __init__(self, device='cuda'):
        self.prof = None
        self.device = device

    def profiler(self):
        activities = [ProfilerActivity.CPU]
        if self.device == 'cuda':
            activities.append(ProfilerActivity.CUDA)
        self.prof = profile(activities=activities, with_stack=True)
        return self.prof

    def save_prof_data(self):
        # raii guard reuqired?
        self.prof.export_chrome_trace("trace_{}.json".format(self.device))
        self.prof.export_stacks("/tmp/profiler_stacks.txt", "self_{}_time_total".format(self.device))
        with open('stack_flame_{}.svg'.format(self.device), 'w') as fp:
            subprocess.run(
                [ '/home/sankeerth/FlameGraph/flamegraph.pl', 
                '--title', 
                "{} time_total".format(self.device), 
                '--countname',
                "us.", 
                '--reverse', 
                '/tmp/profiler_stacks.txt'],
                stdout=fp
            )

    def get_event_inference(self, function_attribute: str) -> 'list[FunctionEvent]':
        ret = []
        for ev in self.prof.events():
            if ev.name == function_attribute:
                ret.append(ev)
        return ret

    def get_total_cuda_time(self, function_attribute: str) -> float:
        evs = self.get_event_inference(function_attribute=function_attribute)
        return sum([item.cuda_time for item in evs])

    def print_profiler_data(self) -> None:
        print(self.prof.key_averages().table(sort_by="{}_time_total".format(self.device), row_limit=30))

class BenchmarkNetwork:
    def __init__(self, input_h: InputHandler, model_h: ModelHandler):
        self.input_h = input_h
        self.model_h = model_h
        self.profiler = ProfileHandler()

    # TODO: only implemented for startindex = 0 for now, check later
    def benchmark(self,  maxiter = 10, save_profiler_data=False, print_profiler_data=True) -> float :
        with torch.no_grad():
            with self.profiler.profiler() as prof:
                for i in range(maxiter):
                    sample = self.input_h.get_single_sample(i)
                    with record_function("model_inference"):
                        self.model_h.run_once(sample)
            
            if save_profiler_data:
                self.profiler.save_prof_data()
            if print_profiler_data:
                self.profiler.print_profiler_data()

        return self.profiler.get_total_cuda_time('model_inference')


class BenchmarkComparer:
    def __init__(self, ih: InputHandler, op_slow: BenchmarkNetwork, op_fast: BenchmarkNetwork):
        self.ih = ih
        self.op_slow = op_slow
        self.op_fast = op_fast

    def compare_runtimes(self) -> float:
        time1 = self.op_slow.benchmark(self.ih.get_single_sample())
        time2 = self.op_fast.benchmark(self.ih.get_single_sample())
        return time2/time1



