import torch
import torch.nn as nn


Tensor = torch.Tensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeqToANNContainer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)


class Layer(nn.Module):  # baseline
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )

    def forward(self, x):
        x = self.fwd(x)
        return x


class TEBN(nn.Module):

    def __init__(self, out_plane, eps=1e-5, momentum=0.1, T=10):
        super(TEBN, self).__init__()
        self.bn = SeqToANNContainer(nn.BatchNorm2d(out_plane))
        self.p = nn.Parameter(torch.ones(T, 1, 1, 1, 1, device=device))  # Default T=10

    def forward(self, input):
        y = self.bn(input)
        y = y.transpose(0, 1).contiguous()

        actual_T = y.shape[0]
        if actual_T != self.p.shape[0]:
            if actual_T < self.p.shape[0]:
                y = y * self.p[:actual_T]
            else:
                print(
                    f"Warning: TEBN initialized for T={self.p.shape[0]}, but received T={actual_T}. Parameter `p` might not be optimal.")
                y = y * self.p
        else:
            y = y * self.p

        y = y.contiguous().transpose(0, 1)  # [N, T, C, H, W]
        return y


class TEBNLayer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding, T=10, groups=1):
        super(TEBNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding, groups=groups, bias=False),
        )
        self.bn = TEBN(out_plane, T=T)

    def forward(self, x):
        y = self.fwd(x)
        y = self.bn(y)
        return y


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama=1.0):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super().__init__()
        self.v_th = thresh
        self.tau = tau
        self.gamma = gamma
        self.mem = None  # Will be initialized on first forward pass
        self.heaviside = ZIF.apply

    def reset_state(self):
        self.mem = None

    def forward(self, x: Tensor) -> Tensor:
        if self.mem is None or self.mem.shape[0] != x.shape[0]:
            self.mem = torch.zeros_like(x[:, 0, ...])

        batch_size, time_steps, *spatial_dims = x.shape

        spikes = torch.zeros_like(x)
        current_mem = self.mem.clone()

        for t in range(time_steps):
            current_mem = self.tau * current_mem + x[:, t, ...]
            spike = self.heaviside(current_mem - self.v_th, self.gamma)
            current_mem = current_mem * (1 - spike)
            spikes[:, t, ...] = spike

        self.mem = current_mem.detach()
        return spikes


class TriangularSurrogate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, alpha=1.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (1 / ctx.alpha) * (1 / ctx.alpha) * (
            (ctx.alpha - input.abs()).clamp(min=0)
        )
        return grad_input * temp, None



def DSConv7_VGG(in_planes, out_planes, T, tau):
    return nn.Sequential(
        TEBNLayer(in_planes, in_planes, kernel_size=7, stride=2, padding=3, groups=in_planes, T=T),
        TEBNLayer(in_planes, out_planes, kernel_size=1, stride=1, padding=0, T=T),
        LIFSpike(tau=tau)
    )


def DSConv5_VGG(in_planes, out_planes, T, tau):
    return nn.Sequential(
        TEBNLayer(in_planes, in_planes, kernel_size=5, stride=2, padding=2, groups=in_planes, T=T),
        TEBNLayer(in_planes, out_planes, kernel_size=1, stride=1, padding=0, T=T),
        LIFSpike(tau=tau)
    )


def DSConv3_VGG(in_planes, out_planes, T, tau):
    return nn.Sequential(
        TEBNLayer(in_planes, in_planes, kernel_size=3, stride=1, padding=1, groups=in_planes, T=T),
        TEBNLayer(in_planes, out_planes, kernel_size=1, stride=1, padding=0, T=T),
        LIFSpike(tau=tau)
    )


def SConv1_VGG(in_planes, out_planes, T, tau):
    return nn.Sequential(
        TEBNLayer(in_planes, out_planes, kernel_size=1, stride=1, padding=0, T=T),
        LIFSpike(tau=tau)
    )


class TACA_Block(nn.Module):

    def __init__(self, channel, k_size=3):
        super(TACA_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, t, c, h, w = x.size()
        y = self.avg_pool(x).view(b, t, c)  # Shape: [B, T, C]

        y_mean = y.mean(dim=1, keepdim=True)  # Shape: [B, 1, C]
        y_var = y.var(dim=1, keepdim=True, unbiased=False)  # Shape: [B, 1, C]

        y_fused = y_mean + y_var  # Shape: [B, 1, C]

        y_conv = self.conv(y_fused)  # Shape: [B, 1, C]

        y_out = self.sigmoid(y_conv).view(b, 1, c, 1, 1)  # Shape: [B, 1, C, 1, 1]

        return x * y_out.expand_as(x)


class PML_SurrogateBlock_VGG(nn.Module):

    def __init__(self, kernels, in_channel, out_channel, num_classes, T=10, tau=0.25):
        super(PML_SurrogateBlock_VGG, self).__init__()
        self.T = T
        convs = []

        if in_channel > out_channel:
            convs.append(SConv1_VGG(in_channel, out_channel, T, tau))
            for kernel in kernels:
                if kernel == 7:
                    convs.append(DSConv7_VGG(out_channel, out_channel, T, tau))
                elif kernel == 5:
                    convs.append(DSConv5_VGG(out_channel, out_channel, T, tau))
                elif kernel == 3:
                    convs.append(DSConv3_VGG(out_channel, out_channel, T, tau))
        else:
            first_kernel = kernels[0]
            if first_kernel == 7:
                convs.append(DSConv7_VGG(in_channel, out_channel, T, tau))
            elif first_kernel == 5:
                convs.append(DSConv5_VGG(in_channel, out_channel, T, tau))
            elif first_kernel == 3:
                convs.append(DSConv3_VGG(in_channel, out_channel, T, tau))
            for kernel in kernels[1:]:
                if kernel == 7:
                    convs.append(DSConv7_VGG(out_channel, out_channel, T, tau))
                elif kernel == 5:
                    convs.append(DSConv5_VGG(out_channel, out_channel, T, tau))
                elif kernel == 3:
                    convs.append(DSConv3_VGG(out_channel, out_channel, T, tau))

        self.convs = nn.Sequential(*convs)
        self.se_block = TACA_Block(out_channel)

        self.prediction = nn.Sequential(
            SeqToANNContainer(nn.AdaptiveAvgPool2d((2, 2))),
            SeqToANNContainer(nn.Flatten()),
            SeqToANNContainer(nn.Linear(out_channel * 4, 512)),
            SeqToANNContainer(nn.BatchNorm1d(512)),
            SeqToANNContainer(nn.LeakyReLU(0.01, inplace=True)),
            SeqToANNContainer(nn.Linear(512, num_classes)),
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.se_block(x)
        x = self.prediction(x)
        return x


class VGGSNN(nn.Module):
    def __init__(self, tau=0.25, T=10, num_class=10, input_size=48):
        super(VGGSNN, self).__init__()
        self.tau = tau
        self.T = T
        self.num_class = num_class
        self.input_size = input_size
        pool = SeqToANNContainer(nn.AvgPool2d(2))

        self.features = nn.Sequential(
            TEBNLayer(2, 64, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            TEBNLayer(64, 128, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            pool,
            TEBNLayer(128, 256, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            TEBNLayer(256, 256, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            pool,
            TEBNLayer(256, 512, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            pool,
            TEBNLayer(512, 512, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            pool,
        )
        W = int(self.input_size / 2 / 2 / 2 / 2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            SeqToANNContainer(nn.Linear(512 * W * W, self.num_class))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class VGGSNN_PML_TACA(nn.Module):

    def __init__(self, tau=0.25, T=10, num_class=10, input_size=48,
                 pml_places=[1, 2, 3], pml_kernels=[[7, 5, 3], [7, 5, 3], [7, 5, 3]], pml_pads=128):
        super(VGGSNN_PML_TACA, self).__init__()
        self.tau = tau
        self.T = T
        self.num_class = num_class
        self.input_size = input_size
        pool = SeqToANNContainer(nn.AvgPool2d(2))


        self.layer1 = nn.Sequential(
            TEBNLayer(2, 64, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            TEBNLayer(64, 128, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            pool,
        )  

        self.layer2 = nn.Sequential(
            TEBNLayer(128, 256, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            TEBNLayer(256, 256, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            pool,
        )  

        self.layer3 = nn.Sequential(
            TEBNLayer(256, 512, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            pool,
        )  

        self.layer4 = nn.Sequential(
            TEBNLayer(512, 512, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1, T=self.T),
            LIFSpike(tau=self.tau),
            pool,
        )  

        self.layers = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])
        channels = [128, 256, 512, 512]

        self.pml_places = pml_places
        self.pml_layers = nn.ModuleList()
        for i in range(len(self.pml_places)):
            idx = self.pml_places[i] - 1
            kerneli = pml_kernels[i]
            self.pml_layers.append(PML_SurrogateBlock_VGG(
                kernels=kerneli,
                in_channel=channels[idx],
                out_channel=pml_pads,
                num_classes=num_class,
                T=self.T,
                tau=self.tau
            ))

        W = int(self.input_size / 2 / 2 / 2 / 2)
        self.classifier = nn.Sequential(
            nn.Dropout(0.25),
            SeqToANNContainer(nn.Linear(512 * W * W, self.num_class))
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        outs = []
        x = input
        pml_i = 0
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if (i + 1) in self.pml_places:
                pml_out = self.pml_layers[pml_i](x)
                outs.append(pml_out)
                pml_i += 1

        x = torch.flatten(x, 2)
        final_out = self.classifier(x)
        outs.insert(0, final_out)
        return outs 


if __name__ == "__main__":
    print("--- Testing VGGSNN & VGGSNN_PML_TACA ---")


    model_base = VGGSNN(T=10, tau=0.25, input_size=48, num_class=10).to(device)
    test_input = torch.randn(1, 10, 2, 48, 48).to(device)
    output_base = model_base(test_input)
    print(f"VGGSNN Output shape: {output_base.shape}")


    model_pml = VGGSNN_PML_TACA(T=10, tau=0.25, input_size=48, num_class=10).to(device)
    outputs_pml = model_pml(test_input)
    print(f"\nVGGSNN_PML_TACA Output count: {len(outputs_pml)} (1 Final + {len(outputs_pml) - 1} Surrogates)")
    for i, out in enumerate(outputs_pml):
        print(f"  Output [{i}] shape: {out.shape}")