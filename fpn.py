import torch
import config as cfg
import torch.nn.functional as F

class PyramidFeatures(torch.nn.Module):
    """fpn网络"""
    def __init__(self):
        super(PyramidFeatures, self).__init__()

        self.c5_channels = cfg.c5_channels  #2048
        self.c4_channels = cfg.c4_channels # 1024
        self.c3_channels = cfg.c3_channels #512
        self.fpn_channels = cfg.fpn_channels #256

        self.P5_1 = torch.nn.Conv2d(self.c5_channels, self.fpn_channels, kernel_size=1, stride=1, padding=0)
        self.P5_2 = torch.nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=1, padding=1)

        self.P4_1 = torch.nn.Conv2d(self.c4_channels, self.fpn_channels, kernel_size=1, stride=1, padding=0)
        self.P4_2 = torch.nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=1, padding=1)

        self.P3_1 = torch.nn.Conv2d(self.c3_channels, self.fpn_channels, kernel_size=1, stride=1, padding=0)
        self.P3_2 = torch.nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=1, padding=1)

        self.P6 = torch.nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=2, padding=1)

        self.P7 = torch.nn.Conv2d(self.fpn_channels, self.fpn_channels, kernel_size=3, stride=2, padding=1)

        for m in self.modules():
            if(isinstance(m,torch.nn.Conv2d)):
                torch.nn.init.xavier_uniform_(m.weight,gain=1)
                torch.nn.init.constant_(m.bias,0)

    def forward(self, x):
        c3, c4, c5 = x  #[2, 512, 152, 100])  ([2, 1024, 76, 50])t  ([2, 2048, 38, 25])

        c3_shape=c3.shape[2:]#toize([152, 100]))
        c4_shape=c4.shape[2:]#torch([76, 50])

        p5=self.P5_1(c5)   # ([2, 256, 38, 25])  ([2, 2048, 38, 25])   tConv2d(2048, 256, (1, 1), (1, 1))
        p5_upsampled=F.interpolate(p5,size=c4_shape,mode="nearest") #torch.Size([2, 256, 76, 50])
        p5=self.P5_2(p5)   #([2, 256, 38, 25])     Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        p4=self.P4_1(c4)  #([2, 1024, 76, 50])       ([2, 1024, 76, 50])      Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        p4=p4+p5_upsampled #torch.Size([2, 256, 76, 50])
        p4_upsampled=F.interpolate(p4,size=c3_shape,mode="nearest") #t([2, 256, 152, 100])
        p4=self.P4_2(p4) #([2, 256, 76, 50])       Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        p3=self.P3_1(c3) #[2, 256, 152, 100])   ([2, 512, 152, 100])   #Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        p3=p3+p4_upsampled #([2, 256, 152, 100])
        p3=self.P3_2(p3) #t([2, 256, 152, 100])   Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        p6=self.P6(p5) #([2, 256, 19, 13]) ([2, 256, 38, 25]) Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        p7=F.relu(p6)
        p7=self.P7(p7) #([2, 256, 10, 7])   [2, 256, 19, 13])   Conv2d(256, 256, kere=(3, 3), stre=(2, 2), padg=(1, 1))

        return [p3,p4,p5,p6,p7]
if __name__ == '__main__':
    x3=  torch.rand(2, 512, 152, 100)
    x4 = torch.rand(2, 1024, 76, 50)
    x5 = torch.rand(2, 2048, 38, 25)
    x=[x3,x4,x5]
    model=PyramidFeatures()
    out=model(x)
