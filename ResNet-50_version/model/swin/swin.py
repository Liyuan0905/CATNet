import torch
from .decoder import DepthDecoder, DepthDecoderUp
from .swin_encoder import SwinTransformer
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch.nn.utils.spectral_norm as sn


from model.resnet.ResNet import B2_ResNet
import torchvision.models as models

class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    # paper: Image Super-Resolution Using Very DeepResidual Channel Attention Networks
    # input: B*C*H*W
    # output: B*C*H*W
    def __init__(
        self, n_feat, kernel_size=3, reduction=16,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

class Edge_Module(nn.Module):

    def __init__(self, in_fea=[256, 256, 256], mid_fea=32):
        super(Edge_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5 = nn.Conv2d(in_fea[2], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)

        self.classifer = nn.Conv2d(mid_fea * 3, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 3)

    def forward(self, x2, x4, x5):
        _, _, h, w = x2.size()
        edge2_fea = self.relu(self.conv2(x2))
        edge2 = self.relu(self.conv5_2(edge2_fea))
        edge4_fea = self.relu(self.conv4(x4))
        edge4 = self.relu(self.conv5_4(edge4_fea))
        edge5_fea = self.relu(self.conv5(x5))
        edge5 = self.relu(self.conv5_5(edge5_fea))

        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=True)
        edge5 = F.interpolate(edge5, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge2, edge4, edge5], dim=1)
        edge = self.rcab(edge)
        edge = self.classifer(edge)
        return edge

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    '''
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    '''

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=[6, 12, 18]):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        # Check if we are using distributed BN and use the nn from encoding.nn
        # library rather than using standard pytorch.nn

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3, dilation=r, padding=r, bias=False),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x, edge):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = F.interpolate(img_features, x_size[2:], mode='bilinear', align_corners=True)
        out = img_features

        edge_features = F.interpolate(edge, x_size[2:],  mode='bilinear', align_corners=True)
        edge_features = self.edge_conv(edge_features)
        out = torch.cat((out, edge_features), 1)

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x

class FCDiscriminator(nn.Module):
    def __init__(self, ndf):
        super(FCDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(4, ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Conv2d(ndf, 1, kernel_size=3, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bn1 = nn.BatchNorm2d(ndf)
        self.bn2 = nn.BatchNorm2d(ndf)
        self.bn3 = nn.BatchNorm2d(ndf)
        self.bn4 = nn.BatchNorm2d(ndf)
        #self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # #self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x

class EBM_Prior(torch.nn.Module):
    def __init__(self, ebm_out_dim, ebm_middle_dim, latent_dim):
        super().__init__()

        e_sn = False

        apply_sn = sn if e_sn else lambda x: x

        f = torch.nn.GELU()

        self.ebm = nn.Sequential(
            apply_sn(nn.Linear(latent_dim, ebm_middle_dim)), f,
            apply_sn(nn.Linear(ebm_middle_dim, ebm_middle_dim)), f,
            apply_sn(nn.Linear(ebm_middle_dim, ebm_out_dim))
        )
        self.ebm_out_dim = ebm_out_dim

    def forward(self, z):
        return self.ebm(z.squeeze()).view(-1, self.ebm_out_dim, 1, 1)


class Swin(torch.nn.Module):
    def __init__(self, img_size):
        super(Swin, self).__init__()

        self.encoder = SwinTransformer(img_size=img_size, embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32], window_size=12)
        pretrained_dict = torch.load('D:/Code/Data/swin_base_patch4_window12_384.pth')["model"]
        # pretrained_dict = torch.load('/home/zzheng/cly/swin_base_patch4_window12_384.pth')["model"]
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(pretrained_dict)
        channel_size = 32
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = BasicConv2d(in_planes=128, out_planes=channel_size, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(in_planes=256, out_planes=channel_size, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(in_planes=512, out_planes=channel_size, kernel_size=3, padding=1)
        self.conv4 = BasicConv2d(in_planes=1024, out_planes=channel_size, kernel_size=3, padding=1)
        self.conv5 = BasicConv2d(in_planes=1024, out_planes=channel_size, kernel_size=3, padding=1)
        
        self.conv_depth = nn.Conv2d(6, 3, kernel_size=1, padding=0)
        
        
    def forward(self, x, depth=None):
        # print(z.size())
        if depth != None:
            x = torch.cat((x,depth),1)
            x = self.conv_depth(x)
        features = self.encoder(x)

        x1 = features[-5]
        x2 = features[-4]
        x3 = features[-3]
        x4 = features[-2]
        x5 = features[-1]

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x5 = self.conv5(x5)

        return x1, x2, x3, x4, x5


class ResNet_endecoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, img_size):
        super(ResNet_endecoder, self).__init__()
        self.resnet = B2_ResNet()
        res50 = models.resnet50(pretrained=True)
        pretrained_dict = res50.state_dict()
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)
        
        channel = 32
        self.conv1 = BasicConv2d(in_planes=256, out_planes=channel, kernel_size=3, padding=1)
        self.conv2 = BasicConv2d(in_planes=512, out_planes=channel, kernel_size=3, padding=1)
        self.conv3 = BasicConv2d(in_planes=1024, out_planes=channel, kernel_size=3, padding=1)
        self.conv4 = BasicConv2d(in_planes=2048, out_planes=channel, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)  # 256 x 64 x 64
        x2 = self.resnet.layer2(x1)  # 512 x 32 x 32
        x3 = self.resnet.layer3_1(x2)  # 1024 x 16 x 16
        x4 = self.resnet.layer4_1(x3)  # 2048 x 8 x 8

        x1_feat = self.conv1(x1)
        x2_feat = self.conv2(x2)
        x3_feat = self.conv3(x3)
        x4_feat = self.conv4(x4)
        
        return x1_feat, x2_feat, x3_feat, x4_feat


class Saliency_feat_decoder(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel):
        super(Saliency_feat_decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.clc_layer = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel*5)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.racb43 = RCAB(channel * 2)
        self.racb432 = RCAB(channel * 3)
        self.racb4321 = RCAB(channel * 4)
        
        self.conv43 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 2 * channel)
        self.conv432 = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 3 * channel)
        
        self.conv = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], channel, 4 * channel)
        
        
    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x1, x2, x3, x4):
        # x5:        (_, 32, 12, 12)
        # x54:       (_, 32, 12, 12)
        # x543:      (_, 32, 24, 24)
        # x5432:     (_, 32, 48, 48)
        # x54321:    (_, 160, 96, 96)
        # sal_init:  (_, 1, 384, 384)
        
        # resx5:        (_, 32, 12, 12)
        # resx4:        (_, 32, 12, 12)
        # resx3:        (_, 32, 24, 24)
        # resx2:        (_, 32, 48, 48)
        # resx1:        (_, 160, 96, 96)
        

        
        x43 = torch.cat((self.upsample2(x4), x3), 1)
        x43 = self.racb43(x43)
        x43 = self.relu(self.conv43(x43))
        
        x432 = torch.cat((self.upsample4(x4),self.upsample2(x43), x2), 1)
        x432 = self.racb432(x432)
        x432 = self.relu(self.conv432(x432))
        
        x4321 = torch.cat((self.upsample8(x4),self.upsample4(x43),self.upsample2(x432), x1), 1)
        x4321 = self.relu(self.racb4321(x4321))
        
        output = self.conv(x4321)
        # sal_init = self.upsample4(self.clc_layer(x54321))
        
        return output



class PureDecoder(nn.Module):
    def __init__(self, img_size):
        super(PureDecoder, self).__init__()
        channel = 32
        self.resnet = ResNet_endecoder(img_size)
        self.noise_conv = BasicConv2d(in_planes=channel, out_planes=channel, kernel_size=3, padding=1)
        self.sal_dec = Saliency_feat_decoder(channel)

        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.clc_layer = self._make_pred_layer(Classifier_Module, [6, 12, 18, 24], [6, 12, 18, 24], 1, channel)
        
    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)
        
    
    
    def forward(self, x):
        x1, x2, x3, x4 = self.resnet(x)
        x4 = self.noise_conv(x4)        

        sal_init = self.sal_dec(x1, x2, x3, x4)  # (_, channel, _, _)
        # sal_4x = self.clc_layer_4x(sal_init)
        sal_pred = self.clc_layer(self.upsample4(sal_init))

        return sal_pred

