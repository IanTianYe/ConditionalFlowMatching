import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """为时间步生成正弦/余弦位置嵌入。"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """带组归一化的基础卷积模块。"""

    def __init__(self, in_ch, out_ch, time_emb_dim, cond_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        # 时间嵌入投影
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        # 条件（类别）嵌入投影
        self.cond_mlp = nn.Linear(cond_emb_dim, out_ch)

        # 残差连接（通道不一致时用 1x1 卷积匹配）
        if in_ch != out_ch:
            self.residual_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t_emb, c_emb):
        h = self.conv1(x)
        h = self.norm1(h)

        # 注入时间与条件嵌入（通过广播到空间维度）
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = h + self.cond_mlp(c_emb)[:, :, None, None]

        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.residual_conv(x)


class UNet(nn.Module):
    """用于条件 Flow Matching 的 U-Net 架构。"""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_classes=10,
        base_channels=64,
        time_emb_dim=128,
        cond_emb_dim=64,
    ):
        super().__init__()

        # 时间嵌入模块
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # 条件（类别）嵌入
        self.cond_embedding = nn.Embedding(num_classes, cond_emb_dim)

        # 输入投影
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # 编码器（下采样）
        self.down1 = Block(base_channels, base_channels, time_emb_dim, cond_emb_dim)
        self.down2 = Block(base_channels, base_channels * 2, time_emb_dim, cond_emb_dim)
        self.down3 = Block(base_channels * 2, base_channels * 2, time_emb_dim, cond_emb_dim)

        self.pool = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = Block(base_channels * 2, base_channels * 2, time_emb_dim, cond_emb_dim)

        # 解码器（上采样）
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 2, stride=2)
        self.up_block1 = Block(base_channels * 4, base_channels * 2, time_emb_dim, cond_emb_dim)

        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.up_block2 = Block(base_channels * 3, base_channels, time_emb_dim, cond_emb_dim)

        self.up3 = nn.ConvTranspose2d(base_channels, base_channels, 2, stride=2)
        self.up_block3 = Block(base_channels * 2, base_channels, time_emb_dim, cond_emb_dim)

        # 输出投影
        self.conv_out = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t, c):
        """
        参数:
            x: 输入张量，形状 [B, C, H, W]
            t: 时间步，形状 [B]
            c: 条件标签（类别），形状 [B]

        返回:
            输出张量，形状 [B, C, H, W]
        """
        # 计算时间与条件嵌入
        t_emb = self.time_mlp(t)
        c_emb = self.cond_embedding(c)

        # 初始卷积
        x = self.conv_in(x)

        # 编码器
        d1 = self.down1(x, t_emb, c_emb)
        x = self.pool(d1)

        d2 = self.down2(x, t_emb, c_emb)
        x = self.pool(d2)

        d3 = self.down3(x, t_emb, c_emb)
        x = self.pool(d3)

        # 瓶颈
        x = self.bottleneck(x, t_emb, c_emb)

        # 解码器 + 跳跃连接
        x = self.up1(x)
        # 确保尺寸匹配以便拼接
        if x.shape != d3.shape:
            x = F.interpolate(x, size=d3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, d3], dim=1)
        x = self.up_block1(x, t_emb, c_emb)

        x = self.up2(x)
        if x.shape != d2.shape:
            x = F.interpolate(x, size=d2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, d2], dim=1)
        x = self.up_block2(x, t_emb, c_emb)

        x = self.up3(x)
        if x.shape != d1.shape:
            x = F.interpolate(x, size=d1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, d1], dim=1)
        x = self.up_block3(x, t_emb, c_emb)

        # 输出层
        x = self.conv_out(x)

        return x


if __name__ == '__main__':
    # 测试模型
    model = UNet()
    x = torch.randn(4, 1, 28, 28)
    t = torch.rand(4)
    c = torch.randint(0, 10, (4,))

    out = model(x, t, c)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
