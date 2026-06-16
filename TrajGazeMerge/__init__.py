import torch
if hasattr(torch, "compiler") and not hasattr(torch.compiler, "is_compiling"):
    torch.compiler.is_compiling = lambda: False
