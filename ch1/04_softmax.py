import torch


import triton
import triton.language as tl
from triton.runtime import driver




def naive_softmax(x):
    """Compute row-wise softmax of X using native pytorch
    ???? PyTorch ?? X ??? softmax


    We subtract the maximum element in order to avoid overflows. Softmax is invariant to
    this shift.
    ??????????????Softmax ???????????
    """
    # read  MN elements ; write M  elements
    # ?? MN ???;?? M ???
    x_max = x.max(dim=1)[0]
    # read MN + M elements ; write MN elements
    # ?? MN + M ???;?? MN ???
    print("x_max",x_max)
    z = x - x_max[:, None]
    # read  MN elements ; write MN elements
    # ?? MN ???;?? MN ???
    print("z",z)
    numerator = torch.exp(z)
    # read  MN elements ; write M  elements
    # ?? MN ???;?? M ???
    print("numerator",numerator)
    denominator = numerator.sum(dim=1)
    print("denominator",denominator[:, None])
    # read MN + M elements ; write MN elements
    # ?? MN + M ???;?? MN ???
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements ; wrote 3MN + 2M elements
    # ??:?? 5MN + 2M ???;?? 3MN + 2M ???
    return ret
 
 
device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        # ?????????????????? 1 ?
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # ?????? n_cols ???????,????????
        # row in a single block
        # ??????
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        # ????? SRAM ?,????,?? BLOCK_SIZE ???? n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        # ?????????????
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        # ???,Triton ??????????,??????(??,??? CUDA ?? __expf)?
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        # ????? DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape


    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    # ????????????? `x` ????????
    BLOCK_SIZE = triton.next_power_of_2(n_cols)


    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # ???????????????????????????????? (`num_warps`)
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    # ?????????????????????????,??????????????
    num_warps = 8


    # Number of software piepling stages.
    # ??????????
    num_stages = 4 if SIZE_SMEM > 200000 else 2


    # Allocate output
    # ??????
    y = torch.empty_like(x)


    # pre-compile kernel to get register usage and compute thread occupancy.
    # ?????????????????????????
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    #print("num_programs",num_programs)
    if kernel is None:
        kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        #occupancy = min(occupancy, SIZE_SMEM // size_smem)
        occupancy = occupancy
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)


    num_programs = min(num_programs, n_rows)
    #print("num_programs",num_programs)
    
    # Create a number of persistent programs.
    # ??????????
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y




print("device:",device)
print("NUM_SM:",NUM_SM)
print("NUM_REGS:",NUM_REGS)
print("SIZE_SMEM:",SIZE_SMEM)
print("WARP_SIZE:",WARP_SIZE)


torch.manual_seed(0)
x = torch.randn(4, 4, device='cuda')
print("x", x)
y_naive = naive_softmax(x)
y_torch = torch.softmax(x, axis=1)
y_triton = softmax(x)
print(y_naive)
print(y_torch)
print( torch.allclose(y_triton, y_torch))


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot ???? x ?????
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name` `x_name` ??????
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot ???,????????????
        line_vals=['triton', 'torch'],  # possible values for `line_arg`` `line_arg` ????
        line_names=[
            "Triton",
            "Torch",
        ],  # label name for the lines ???????
        styles=[('blue', '-'), ('green', '-')],  # line styles ?????
        ylabel="GB/s",  # label name for the y-axis y ??????
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot. ?????,???????????
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name` `x_names` ? `y_name` ???????????
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)
    
benchmark.run(show_plots=True, print_data=True)
