import torch
from task import input_t, output_t
#from utils import make_match_reference

import triton
import triton.language as tl

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

    # Generate random inputs with FP8 quantization
    a = (torch.randn((k, m), dtype=torch.bfloat16, device="cuda", generator=gen)).to(torch.float8_e4m3fnuz)
    b = (torch.randn((k, n), dtype=torch.bfloat16, device="cuda", generator=gen)).to(torch.float8_e4m3fnuz)

    # Generate scaling factors with FP32
    a_scale = torch.randn([scale_k, m], dtype=torch.float32, device="cuda", generator=gen)
    b_scale = torch.randn([scale_k, scale_n], dtype=torch.float32, device="cuda", generator=gen)


    c = torch.zeros((m, n), dtype=torch.bfloat16, device="cuda")
    return (a.T, b.T, a_scale.T, b_scale.T, c)
    
    
a,b,a_s,b_s,c = generate_input(4,4,4, 0)
'''
print("A",a)
print("B",b)

print("a scale", a_s)
print("b scale", b_s)

print(c)
'''
def custom_kernel(data: input_t) -> output_t:
    """
    Reference implementation of block-scale fp8 gemm 
    Args:
        data: Tuple that expands to:
            a: torch.Tensor[float8_e4m3fnuz] of shape [m, k], 
            b: torch.Tensor[float8_e4m3fnuz] of shape [n, k], 
            a_scale: torch.Tensor[float32] of shape [m, k // 128], 
            b_scale: torch.Tensor[float32] of shape [n // 128, k // 128], 
            c: torch.Tensor[bfloat16] of shape [m, n]
    Returns:
        Tensor containing output in bf16
    """
    # c: [m, n] is pre-allocated memory to avoid timing allocation overhead.
    a, b, a_scale, b_scale, c = data
    
    # a is M x K in column-major order, we convert here for simplicity.
    a = a.contiguous()
    
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()
    
    # constants
    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]
    block_shape_n = 128
    block_shape_k = 128
    scale_n = b_scale.shape[0]
    scale_k = b_scale.shape[1]

    # Apply scaling to input 'a'
    a_scale = a_scale.unsqueeze(-1).repeat(1, 1, block_shape_k)  # Shape: [m, scale_k, block_shape_k]
    a_scale = a_scale.reshape(m, scale_k * block_shape_k) 
    a_scale = a_scale[:, :k]

    # Dequantize 'a', in your implementation you should do this at the end.
    print(a)
    print(a_scale)
    a = a.to(a_scale.dtype) * a_scale 
    print(a)
    print(a.dtype)
    # Apply scaling to input 'b'
    b_scale = (
        b_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k)
        .permute(0, 2, 1, 3)  # Reorder dimensions: [scale_n, blk_n, scale_k, blk_k]
        .reshape(scale_n * block_shape_n, scale_k * block_shape_k)
    )
    b_scale = b_scale[:n, :k]
    print(b)
    print(b_scale)
    # Dequantize 'b', in your implementation you should do this at the end.
    b = b.to(b_scale.dtype) * b_scale 
    print(b)
    c[...] = (a @ b.T).to(torch.bfloat16)
    return c

@triton.jit
def fp8_gemm_kernel(
    # ??????
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    # ????
    m, n, k,
    # ??(????)
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_a_scale_m, stride_a_scale_k,
    stride_b_scale_n, stride_b_scale_k,
    # ????(????)
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # ??????????
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # ???????/???
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # ??????(BF16)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.bfloat16)
    
    # ???? GEMM
    for k_block in range(0, k, BLOCK_SIZE_K):
        # ?? A ???(FP8 -> FP16)
        a_offs = offs_m[:, None] * stride_am + (offs_k + k_block)[None, :] * stride_ak
        a_mask = (offs_m[:, None] < m) & ((offs_k + k_block)[None, :] < k)
        a_fp8 = tl.load(a_ptr + a_offs, mask=a_mask, other=0)
        
        # ?? A ?????
        a_scale_offs = (offs_m[:, None] // 128) * stride_a_scale_m + \
                       ((offs_k + k_block)[None, :] // 128) * stride_a_scale_k
        a_scale = tl.load(a_scale_ptr + a_scale_offs, mask=a_mask, other=1.0)
        
        # ?? A ? FP16:FP8 * scale
        a_fp16 = (a_fp8.to(tl.float16) * a_scale).to(tl.bfloat16)
        
        # ?? B ???(FP8 -> FP16)
        b_offs = (offs_k + k_block)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = ((offs_k + k_block)[:, None] < k) & (offs_n[None, :] < n)
        b_fp8 = tl.load(b_ptr + b_offs, mask=b_mask, other=0)
        
        # ?? B ?????
        b_scale_offs = ((offs_k + k_block)[:, None] // 128) * stride_b_scale_k + \
                       (offs_n[None, :] // 128) * stride_b_scale_n
        b_scale = tl.load(b_scale_ptr + b_scale_offs, mask=b_mask, other=1.0)
        
        # ?? B ? FP16:FP8 * scale
        b_fp16 = (b_fp8.to(tl.float16) * b_scale).to(tl.bfloat16)
        
        # ????????
        accumulator += tl.dot(a_fp16, b_fp16)
    
    # ????? C(BF16)
    c_offs = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    tl.store(c_ptr + c_offs, accumulator, mask=c_mask)

def fp8_gemm(
    #a: torch.Tensor,  # [m, k], float8_e4m3fnuz
    #b: torch.Tensor,  # [n, k], float8_e4m3fnuz
    #a_scale: torch.Tensor,  # [m, k // 128], float32
    #b_scale: torch.Tensor,  # [n // 128, k // 128], float32
    #c: torch.Tensor,  # [m, n], bfloat16
    data: input) -> output_t: 

    a, b, a_scale, b_scale, c = data
    a = a.contiguous()
    
    a_scale = a_scale.contiguous()
    b_scale = b_scale.contiguous()
    
    assert a.is_contiguous() and b.is_contiguous()
    assert a_scale.is_contiguous() and b_scale.is_contiguous()
    assert c.is_contiguous()
    
    m, k = a.shape
    n, _ = b.shape
    
    # ??????(?????)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # ??????
    grid = (triton.cdiv(m, BLOCK_SIZE_M), triton.cdiv(n, BLOCK_SIZE_N))
    
    # ????
    fp8_gemm_kernel[grid](
        a, b, c,
        a_scale, b_scale,
        m, n, k,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        a_scale.stride(0), a_scale.stride(1),
        b_scale.stride(0), b_scale.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return c
    
    
    
data = generate_input(2,3,4, 0)
torch_output = custom_kernel(data)
print(torch_output)

#triton_output = fp8_gemm(data)
#print(triton_output)
'''
rtol = 1e-2 
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("YES Triton and Torch match")
else:
    print("NO Triton and Torch differ")  
'''