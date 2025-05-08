import torch


import triton
import triton.language as tl
from triton.runtime import driver

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
print("device:",device)
print("NUM_SM:",NUM_SM)
print("NUM_REGS:",NUM_REGS)
print("SIZE_SMEM:",SIZE_SMEM)
print("WARP_SIZE:",WARP_SIZE)
print(target.backend)
print(target.arch)