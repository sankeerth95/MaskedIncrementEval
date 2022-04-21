from benchmark.benchmark_wrappers import *

def run_sequence_of_deltaconv_strsparse_benchmarks():
    in_shape = [1, 256, 346, 260]
    shape = [256,512]
    k=3
    for strsparsity in [0.0, 0.7, 0.9, 0.95, 1.]:
        ret = benchmark_deltaconv_strsparse(in_shape=in_shape, shape=shape, k=k, strsparsity=strsparsity)
        print(ret)

def run_sequence_of_deltaconv_sparse_benchmarks():
    in_shape = [1, 256, 346, 260]
    shape = [256,512]
    k=3
    for sparsity in [0.0, 0.7, 0.9, 0.95, 0.98, 0.99]:
        ret = benchmark_deltaconv(in_shape=in_shape, shape=shape, k=k, sparsity=sparsity)
        print(ret)


def compare_deltaconv_with_torchconv_(ksize=3):
    
    def compare_benchmarks(in_shape, shape=(32, 64), k=3, stride=1):
        t1 = benchmark_conv(in_shape=in_shape, shape=shape, k=k)
        t2 = benchmark_deltaconv(in_shape=in_shape, shape=shape, k=k)
        return t1/t2

    print(compare_benchmarks(in_shape=(1, 32, 346, 260), shape=(32, 64), k=ksize))
    print(compare_benchmarks(in_shape=(1, 3, 346, 260), shape=(3, 32), k=ksize))
    print(compare_benchmarks(in_shape=(1, 32*2, 346, 260), shape=(32*2, 64*2), k=ksize)) # fastest as of now
    print(compare_benchmarks(in_shape=(1, 32*4, 346, 260), shape=(32*4, 64*4), k=ksize))
    print(compare_benchmarks(in_shape=(1, 32, 346//2, 260//2), shape=(32, 64), k=ksize))
    print(compare_benchmarks(in_shape=(1, 32, 346//4, 260//4), shape=(32, 64), k=ksize))




def compare_deltaconvstrsparse_with_baseline(ksize=3, strsparsity=0.3):
    
    def compare_benchmarks(in_shape, shape=(32, 64), k=3, stride=1):
        t1 = benchmark_conv(in_shape=in_shape, shape=shape, k=k)
        t2 = benchmark_deltaconv_strsparse(in_shape=in_shape, shape=shape, k=k, strsparsity=strsparsity)
        return t1/t2

    print(compare_benchmarks(in_shape=(1, 32, 346, 260), shape=(32, 64), k=ksize))
    print(compare_benchmarks(in_shape=(1, 3, 346, 260), shape=(3, 32), k=ksize))
    print(compare_benchmarks(in_shape=(1, 32*2, 346, 260), shape=(32*2, 64*2), k=ksize))
    print(compare_benchmarks(in_shape=(1, 32*4, 346, 260), shape=(32*4, 64*4), k=ksize))
    print(compare_benchmarks(in_shape=(1, 32, 346//2, 260//2), shape=(32, 64), k=ksize))
    print(compare_benchmarks(in_shape=(1, 32, 346//4, 260//4), shape=(32, 64), k=ksize))

def compare_deltaconvstrsparse_varying_sparsity(ksize=3, sp1=0.3, sp2=0.7):
    
    def compare_benchmarks(in_shape, shape=(32, 64), k=3, stride=1):
        t1 = benchmark_deltaconv_strsparse(in_shape=in_shape, shape=shape, k=k, strsparsity=sp1)
        t2 = benchmark_deltaconv_strsparse(in_shape=in_shape, shape=shape, k=k, strsparsity=sp2)
        return t1/t2

    print(compare_benchmarks(in_shape=(1, 32, 346, 260), shape=(32, 64), k=ksize))
    print(compare_benchmarks(in_shape=(1, 3, 346, 260), shape=(3, 32), k=ksize))
    print(compare_benchmarks(in_shape=(1, 32*2, 346, 260), shape=(32*2, 64*2), k=ksize)) 
    print(compare_benchmarks(in_shape=(1, 32*4, 346, 260), shape=(32*4, 64*4), k=ksize))
    print(compare_benchmarks(in_shape=(1, 32, 346//2, 260//2), shape=(32, 64), k=ksize))
    print(compare_benchmarks(in_shape=(1, 32, 346//4, 260//4), shape=(32, 64), k=ksize))


if __name__ == '__main__':

    # compare_deltaconvstrsparse_varying_sparsity(ksize=3, sp1=0.3, sp2=0.7)

    # run_sequence_of_deltaconv_sparse_benchmarks()
    pth = None
    # pth = '/home/sankeerth/tmp/e2depth_train_test_1/checkpoint-epoch001-loss-0.0000.pth.tar'
    print(benchmark_e2vid_incr(pth))

    # ret = benchmark_conv(in_shape = (1, 256, 346, 260), shape=[256, 512])
    # print(ret)
    


