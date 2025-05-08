import triton
import torch
import triton.language as tl


a = torch.rand(4,4)
b = torch.rand(4,4)

c_torch = torch.add(a,b)

print(a)
print(b)
print(c_torch)


@triton.jit
def mat_add_kernel(a_ptr, b_ptr, c_ptr, N0, N1, B0: tl.constexpr, B1: tl.constexpr):
    block_id_x = tl.program_id(axis=0)
    block_id_y = tl.program_id(axis=1)
    
    
    off_x =  block_id_x * B0 + tl.arange(0, B0)
    off_y =  block_id_y * B1 + tl.arange(0, B1)
    off_z = off_x[None, :] + off_y[:, None] * N0
    
    maskx = off_x < N0 
    masky = off_y < N1
    mask = maskx[None, :] & masky[:, None]
    a = tl.load(a_ptr + off_z , mask)
    b = tl.load(b_ptr + off_z , mask)
    
    z = a + b
    
    
    tl.store(c_ptr + off_z, z, mask)
    return 
    
def mat_add(a,b):
    N0,N1 = a.shape
    B0 = 2
    B1 = 2
    c = torch.empty_like(a).to("cuda") 
    grid = lambda meta:(triton.cdiv(N0, B0), triton.cdiv(N1, B1))
    mat_add_kernel[grid](a,b,c,N0, N1, B0, B1)
    
    return c
    
triton_c = mat_add(a.to("cuda") ,b.to("cuda") )
print(triton_c)

c_ = triton_c.to("cpu")
match = torch.allclose(c_torch, c_,rtol=1e-05, atol=1e-08, equal_nan=False)
print(match)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot 用作图表 x 轴的参数名
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name` `x_name` 的不同可能值
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot 参数名，其值对应于图表中不同线条
        line_vals=['triton', 'torch'],  # possible values for `line_arg`` `line_arg` 的可能值
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines 线条的标签名称
        styles=[('blue', '-'), ('green', '-')],  # line styles 线条的样式
        ylabel="GB/s",  # label name for the y-axis y 轴的标签名称
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot. 图表的名称，也用作保存图表的文件名
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name` `x_names` 和 `y_name` 中未包含的函数参数的值
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    y = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.add(x, y))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: mat_add(x,y))
    gbps = lambda ms: 3 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)
    
benchmark.run(show_plots=True, print_data=True)