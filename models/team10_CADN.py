import torch
import torch.nn as nn
from torch import einsum

from utils.config import size,batch_size,patchSize,cudaNum,featureDepth,attentionDepth

import torch.nn.functional as F

#torch.cuda.set_device(cudaNum)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block(featureDepth,attentionDepth))
    return nn.Sequential(*layers)


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock, self).__init__()
        #nf: number of feature maps
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

    
class GRDB(nn.Module):
    def __init__(self, featureNum=64,growthNum=32, bias=True):
        super(GRDB, self).__init__()
        
        self.RDB_1 = ResidualDenseBlock(nf=featureNum, gc=growthNum, bias=True)
        self.RDB_2 = ResidualDenseBlock(nf=featureNum, gc=growthNum, bias=True)
        self.RDB_3 = ResidualDenseBlock(nf=featureNum, gc=growthNum, bias=True)
        self.conv = nn.Conv2d(3*featureNum,featureNum,kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        
    def forward(self, x):
                              
        x1 = self.relu(self.RDB_1(x))
        x2 = self.relu(self.RDB_2(x1))
        x3 = self.relu(self.RDB_3(x2))
        x4 = self.conv(torch.cat((x1, x2, x3), 1))
        
        return x4 * 0.2 + x

class SElayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class PAM_Module(nn.Module):
    # 空间注意力模块
    def __init__(self , in_dim):
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim //8 , kernel_size =1 )
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim // 8 , kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim = -1)
    def forward(self , x):
        x = x.squeeze(-1)
        #print((x.shape))
        m_batchsize , C,height , width = x.size()
        #print("m_batchsize :",m_batchsize , "  C:",C , " heigth:",height , " width:" , width)
        # permute:维度换位
        # proj_query: (1,60,9,9) -> (1,7,9,9) -> (1,7,81) -> (1,81,7)
        proj_query = self.query_conv(x).view(m_batchsize , -1 , width*height).permute(0,2,1)
        #print("proj_equery : " , proj_query.shape)
        # proj_key: (1,60,9,9) -> (1,7,9,9) -> (1,7,81)
        proj_key = self.key_conv(x).view(m_batchsize , -1 , width*height)
        #print("proj_key:" , proj_key.shape)
        # energy : (1 , 81 , 81) 空间位置上每个位置相对与其他位置的注意力energy
        energy = torch.bmm(proj_query , proj_key)
        attention = self.softmax(energy) #对第三个维度求softmax，某一维度的所有行求softmax
        proj_value = self.value_conv(x).view(m_batchsize , -1 , width*height)
        #print("proj_value : " , proj_value.shape)
        #proj_value : （1,60,81） attetnion:(1,81,81) -> (1,60,81)
        out = torch.bmm(proj_value , attention.permute(0,2,1)) #60行81列，每一行81个元素都是每个元素对其他位置的注意力权重乘以value后的值
        out = out.view(m_batchsize , C , height , width)
        out = (self.gamma*out + x).unsqueeze(-1)
        return out.view(m_batchsize , C , height , width)

class CAM_Module(nn.Module):
    # 通道注意力模块
    def __init__(self , in_dim) :
        super(CAM_Module, self).__init__()
        self.channel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1)) #可学习参数
        self.softmax = torch.nn.Softmax(dim = -1)
    def forward(self , x):
        m_batchsize , C , height , width = x.size()

        proj_query = x.view(m_batchsize , C , -1)
        #print("proj_query:" , proj_query.shape)
        proj_key = x.view(m_batchsize , C , -1).permute(0 , 2 , 1)
        #print("proj_key : " , proj_key.shape)
        energy = torch.bmm(proj_query , proj_key)
        # print("energy:" , energy)
        # expand_as(energy) 把tensor的形状扩展为energy一样的size
        energy_new = torch.max(energy , -1 , keepdim = True)[0].expand_as(energy) - energy
        # print(energy_new)
        attention = self.softmax(energy_new)
        #print(attention.shape)
        proj_value = x.view(m_batchsize , C , -1)
        #print(proj_value.shape)
        out = torch.bmm(attention , proj_value)
        #print(out.shape)
        out = out.view(m_batchsize , C , height , width)
        out = self.gamma*out + x


        return out


# 定义判别器  #####Discriminator######使用多层网络来作为判别器

# 最后接sigmoid激活函数得到一个0到1之间的概率进行二分类。
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1)
        self.bn7 = nn.BatchNorm2d(512)

        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=2)
        self.bn8 = nn.BatchNorm2d(512)

        #self.mhsa = MHSA(dim=512, fmap_size=(feature_size, feature_size))

        self.ave = nn.AdaptiveAvgPool2d(1)

        self.maxPool = nn.AdaptiveMaxPool2d(1)

        self.conv9 = nn.Conv2d(1024, 256, kernel_size=1)

        self.conv10 = nn.Conv2d(256, 1, kernel_size=1)

        self.relu = nn.LeakyReLU(0.5)

        #self.CAM = CAM_Module(512)
        self.PAM = PAM_Module(512)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.relu(self.bn7(self.conv7(x)))
        x = self.relu(self.bn8(self.conv8(x)))
       
        x_PAM = self.PAM(x)
        #x_CAM = self.CAM(x)
        
        x_ave = self.ave(x)
        x_max = self.maxPool(x)

        x = torch.cat((x_ave,x_max),1)
        
        x = self.relu(self.conv9(x))
        
        x = self.conv10(x)

        x = torch.sigmoid(x.view(batch_size, 1))

        return x


# ###### 定义生成器 Generator #####

# 能够在-1～1之间。
class generator(nn.Module):
    def __init__(self,num=featureDepth,kernel=1):
        super(generator, self).__init__()
        self.conv1 = nn.Conv2d(3, int(num), kernel_size=kernel, stride=1, padding=0)
        self.GRDB = make_layer(GRDB,3)
        
        self.tconv = nn.ConvTranspose2d(3*num,3, kernel_size=kernel, stride=1, padding=0)
        self.relu = nn.ReLU(inplace= False)
        self.tanh = nn.Tanh()
        self.se = SElayer(int(num))

    def forward(self, x):
        
        residual_1 = x
        x_1 = self.relu(self.conv1(x))
        
        x_1_se = self.se(x_1)
        
        x_GRDB = self.relu(self.GRDB(x_1))

        out = self.tconv(self.relu(torch.cat((x_GRDB,x_1_se,x_1),1)))
        out = residual_1 + out

        x = self.tanh(out)
        
        return x
