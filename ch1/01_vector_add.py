import triton
import torch
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE:tl.constexpr,):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask = mask)
    y = tl.load(y_ptr + offsets, mask = mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask = mask)

@triton.jit
def add_kernel_2tile(x_ptr, y_ptr, output_ptr, n_elements, num_tiles_per_CTA ,BLOCK_SIZE:tl.constexpr,):
    pid = tl.program_id(axis = 0)
    block_start = pid * BLOCK_SIZE * num_tiles_per_CTA
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    for i in range(num_tiles_per_CTA):
      mask = offsets < n_elements
      x = tl.load(x_ptr + offsets, mask = mask)
      y = tl.load(y_ptr + offsets, mask = mask)
      output = x + y
      tl.store(output_ptr + offsets, output, mask = mask)
      offsets += BLOCK_SIZE

def add_2tile(x:torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    
    n_elements = output.numel()
    TILE_SIZE = 128
    num_tiles_per_CTA = 2
    grid = (triton.cdiv(n_elements, num_tiles_per_CTA * TILE_SIZE),1,1)
    add_kernel_2tile[grid](x, y, output, n_elements,num_tiles_per_CTA ,BLOCK_SIZE=TILE_SIZE)
    return output


def add(x:torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    #assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

def add1(x:torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    
    n_elements = output.numel()
    TILE_SIZE = 128
    grid = (triton.cdiv(n_elements, TILE_SIZE),1,1)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=TILE_SIZE, num_warps = 4)
    return output
    
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=0)
y = torch.rand(size, device=0)
output_torch = x + y
output_triton = add_2tile(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device=0, dtype=torch.float32)
    y = torch.rand(size, device=0, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add_2tile(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)