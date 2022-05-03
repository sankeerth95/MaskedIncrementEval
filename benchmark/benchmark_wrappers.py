from benchmark.model_getter import ModelGetter
from benchmark.model_handlers import IncrModelHandler, BaselineModelhandler
from benchmark.input_handlers import IncrDatasetInputHandler, IncrDatasetInputHandler_2args, RandomInputHandler, SparseRandomInputHandler, StructurallySparseRandomInputHandlerNCHW
from benchmark.operations import ActivationHandler, ActivationIncrHandler, Conv2dBaseline, Conv3x3IncrBaseline
from benchmark.ops_benchmark import BenchmarkNetwork

def benchmark_e2vid_incr(pth=None):
    device = 'cuda'
    input_h = IncrDatasetInputHandler_2args(start_index=10, device=device)
    op = ModelGetter.get_e2vid_incr_model(pth).to(device)    
    model_h = IncrModelHandler(op)
    model_h.refresh(input_h.prev_x)
    benchmark = BenchmarkNetwork(input_h, model_h)
    return benchmark.benchmark(maxiter=10, save_profiler_data=False, print_profiler_data=True)

def benchmark_e2vid(pth=None):
    device = 'cuda'
    input_h = IncrDatasetInputHandler(start_index=10, device=device)
    op = ModelGetter.get_e2vid_model(pth).to(device)    
    model_h = BaselineModelhandler(op)
    benchmark = BenchmarkNetwork(input_h, model_h)
    return benchmark.benchmark(maxiter=40, save_profiler_data=False, print_profiler_data=False)


def benchmark_conv(in_shape, shape=(32, 64), k=3, stride=1):
    device = 'cuda'
    input_h = RandomInputHandler(in_shape, device=device)
    model_h = Conv2dBaseline(shape, kernel=k, device=device)
    benchmark = BenchmarkNetwork(input_h, model_h)
    return benchmark.benchmark(maxiter=40, save_profiler_data=True, print_profiler_data=False)



def benchmark_incrconv(in_shape, shape=(32, 64), k=3, stride=1):
    device = 'cuda'
    input_h = RandomInputHandler(in_shape, device=device)
    model_h = Conv3x3IncrBaseline(in_shape, shape, kernel_size=k, device=device)
    benchmark = BenchmarkNetwork(input_h, model_h)
    return benchmark.benchmark(maxiter=40, save_profiler_data=False, print_profiler_data=False)



def benchmark_deltaconv(in_shape, shape=(32, 64), k=3, stride=1, sparsity=0.9):
    device='cuda'
    input_h = SparseRandomInputHandler(in_shape, sparsity=sparsity, device=device)
    model_h = DeltaConvBaseline(shape, kernel=k, device=device)
    benchmark = BenchmarkNetwork(input_h, model_h)
    return benchmark.benchmark(maxiter=40, save_profiler_data=False, print_profiler_data=False)



# CHW tensor only;
def benchmark_incrRelu(in_shape):
    device='cuda'
    input_h = SparseRandomInputHandler(in_shape, sparsity=0.5, device=device)
    model_h = ActivationIncrHandler(in_shape, device=device)
    benchmark = BenchmarkNetwork(input_h, model_h)
    return benchmark.benchmark(maxiter=40, save_profiler_data=False, print_profiler_data=False)




# CHW tensor only;
def benchmark_Relu(in_shape):
    device='cuda'
    input_h = SparseRandomInputHandler(in_shape, sparsity=0.5, device=device)
    model_h = ActivationHandler(device=device)
    benchmark = BenchmarkNetwork(input_h, model_h)
    return benchmark.benchmark(maxiter=40, save_profiler_data=False, print_profiler_data=False)



