try:
    import torch
    from torch.autograd import Variable
except:
    print('PyTorch not working. MMD measurement won\'t be available')
