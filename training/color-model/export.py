# 학습한 CNN(state_dict)을 ONNX로. 입력 image [N,3,64,64] RGB 0~1, 출력 prob [N,20] (softmax 포함)
import sys, torch, torch.nn as nn, torch.nn.functional as F
class Net(nn.Module):
    def __init__(s, n=20, w=1):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        c = [24*w, 48*w, 96*w, 160*w]
        s.f = nn.Sequential(nn.Conv2d(3, c[0], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True),
                            blk(c[0], c[1]), nn.MaxPool2d(2), blk(c[1], c[2]), nn.MaxPool2d(2), blk(c[2], c[3]),
                            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(c[3], n))
        s.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
    def forward(s, x):
        return s.f((x - s.mean) / 0.25)
class WithSoftmax(nn.Module):
    def __init__(s, net): super().__init__(); s.net = net
    def forward(s, image): return F.softmax(s.net(image), 1)
net = Net(w=int(sys.argv[3]) if len(sys.argv) > 3 else 1); net.load_state_dict(torch.load(sys.argv[1])); net.eval()
m = WithSoftmax(net).eval()
torch.onnx.export(m, (torch.rand(1, 3, 64, 64),), sys.argv[2], input_names=['image'], output_names=['prob'],
                  dynamic_axes={'image': {0: 'n'}, 'prob': {0: 'n'}}, opset_version=17, dynamo=False)
print('saved', sys.argv[2])
