import torch

# Set to False to disable cuDNN and test if the issue is cuDNN specific
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

# Check your current CUDA and cuDNN version
print(torch.version.cuda)
print(torch.backends.cudnn.version())

try:
    # Your problematic code snippet
    data = torch.randn([1, 64, 130, 130], dtype=torch.float, device='cuda', requires_grad=True)
    net = torch.nn.Conv2d(64, 64, kernel_size=[3, 3], padding=[1, 1], stride=[1, 1], dilation=[1, 1], groups=1)
    net = net.cuda().float()
    out = net(data)
    out.backward(torch.randn_like(out))
    torch.cuda.synchronize()
except RuntimeError as e:
    print(e)
    # Add any additional information here that might help to diagnose the problem
