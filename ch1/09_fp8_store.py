import triton
import triton.language as tl
import torch
from task import input_t, output_t
block_shape = (128, 128)

def generate_input(m: int, n: int, k: int, seed: int) -> input_t:
    """
    Generate random input and weights for Blockwise W8A8 Matmul scaled to FP32.
    
    Returns:
        Tuple of (
            a: torch.Tensor[float8_e4m3fnuz] of shape [m, k], 
            b: torch.Tensor[float8_e4m3fnuz] of shape [n, k], 
            a_scale: torch.Tensor[float32] of shape [m, k // 128], 
            b_scale: torch.Tensor[float32] of shape [n // 128, k // 128], 
            c: torch.Tensor[bfloat16] of shape [m, n]
        )
    """
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    block_shape_n, block_shape_k = block_shape
    scale_n =  (n + block_shape_n - 1) // block_shape_n
    scale_k =  (k + block_shape_k - 1) // block_shape_k
    print(scale_n,scale_k )
    # Generate random inputs with FP8 quantization
    a = (torch.randn((k, m), dtype=torch.bfloat16, device="cuda", generator=gen)).to(torch.float8_e4m3fnuz)
    b = (torch.randn((k, n), dtype=torch.bfloat16, device="cuda", generator=gen)).to(torch.float8_e4m3fnuz)

    # Generate scaling factors with FP32
    a_scale = torch.randn([scale_k, m], dtype=torch.float32, device="cuda", generator=gen)
    b_scale = torch.randn([scale_k, scale_n], dtype=torch.float32, device="cuda", generator=gen)


    c = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    return (a.T, b.T, a_scale.T, b_scale.T, c)
    
    
data = generate_input(2,3,4,0)
#a, b, a_scale, b_scale, c = data
#print(a, b, a_scale, b_scale, c)

@triton.jit
def fp8_gemm_kernel(
    # ??????
    a_ptr,
    stride_am, stride_ak,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # ??????????
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

def test(data):
    a, b, a_scale, b_scale, c = data
    
    m, k = a.shape
    n, _ = b.shape
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # ??????
    print(a)
    grid = (triton.cdiv(m, BLOCK_SIZE_M), triton.cdiv(n, BLOCK_SIZE_N))
    fp8_gemm_kernel[grid](
        a,a.stride(0), a.stride(1),BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K)

test(data)



