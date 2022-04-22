from benchmark.model_getter import ModelGetter
from benchmark.model_handlers import IncrModelHandler, BaselineModelhandler
from benchmark.input_handlers import IncrDatasetInputHandler, RandomInputHandler, SparseRandomInputHandler, StructurallySparseRandomInputHandlerNCHW
from benchmark.operations import Conv2dBaseline
from benchmark.ops_benchmark import BenchmarkNetwork
from metrics.structural_sparsity import field_channel_sparsity

def benchmark_e2vid_incr(pth=None):
    device = 'cuda'
    input_h = IncrDatasetInputHandler(start_index=10, device=device)
    op = ModelGetter.get_e2vid_incr_model(pth).to(device)    
    model_h = IncrModelHandler(op, prev_x_input=input_h.prev_x)
    benchmark = BenchmarkNetwork(input_h, model_h)
    return benchmark.benchmark(maxiter=40, save_profiler_data=True, print_profiler_data=True)

def benchmark_e2vid(pth=None):
    device = 'cuda'
    input_h = IncrDatasetInputHandler(start_index=10, device=device)
    op = ModelGetter.get_e2vid_model(pth).to(device)    
    model_h = BaselineModelhandler(op)
    benchmark = BenchmarkNetwork(input_h, model_h)
    return benchmark.benchmark(maxiter=40, save_profiler_data=False, print_profiler_data=True)


def benchmark_conv(in_shape, shape=(32, 64), k=3, stride=1):
    device = 'cuda'
    input_h = RandomInputHandler(in_shape, device=device)
    model_h = Conv2dBaseline(shape, kernel=k, device=device)
    benchmark = BenchmarkNetwork(input_h, model_h)
    return benchmark.benchmark(maxiter=40, save_profiler_data=False, print_profiler_data=False)

def benchmark_spconv(in_shape, shape=(32, 64), k=3, stride=1, sparsity=0.99):
    device ='cuda'
    input_h = SparseRandomInputHandler(in_shape, sparsity=sparsity, device=device)
    model_h = SpconvBaseline(shape, kernel=k, device=device)
    benchmark = BenchmarkNetwork(input_h, model_h)
    return benchmark.benchmark(maxiter=40, save_profiler_data=False, print_profiler_data=False)


def benchmark_deltaconv(in_shape, shape=(32, 64), k=3, stride=1, sparsity=0.9):
    device='cuda'
    input_h = SparseRandomInputHandler(in_shape, sparsity=sparsity, device=device)
    model_h = DeltaConvBaseline(shape, kernel=k, device=device)
    benchmark = BenchmarkNetwork(input_h, model_h)
    return benchmark.benchmark(maxiter=20, save_profiler_data=False, print_profiler_data=False)


def benchmark_deltaconv_strsparse(in_shape, shape=(32, 64), k=3, stride=1, strsparsity=0.4):
    device='cuda'
    input_h = StructurallySparseRandomInputHandlerNCHW(in_shape, field_size=3, strsparsity=strsparsity, device=device)
    model_h = DeltaConvBaseline(shape, kernel=k, device=device)
    benchmark = BenchmarkNetwork(input_h, model_h)
    return benchmark.benchmark(maxiter=40, save_profiler_data=False, print_profiler_data=False)


