import torch
import torch.nn as nn


class ConditionalFlowMatching:
    """
    条件式 Flow Matching（流匹配）训练器。

    Flow Matching 的目标是学习一个“速度场”（向量场）v_theta(x, t, c)，
    使其沿着时间 t∈[0, 1] 将一个易采样的基分布（通常是高斯噪声 x_0 ~ N(0, I)）
    连续“推送”到真实数据分布（样本 x_1 ~ pdata）。这里采用最常见的“直线桥接/Rectified Flow”路径：

        条件概率路径：x_t = t * x_1 + (1 - t) * x_0

    在这条路径下，轨迹的真实条件速度（对 t 求导）为：
        v_t(x_t | x_1, x_0) = d/dt x_t = x_1 - x_0

    训练时，我们对 (x_t, t, c) 回归这个目标速度（MSE）。由于
    E[v_t(x_t | x_1, x_0) | x_t] = 真实的边缘速度 v*(x_t, t)，
    因而在无偏采样下，用条件速度作监督可以一致地学到所需的边缘速度场。
    """

    def __init__(self, model, device='cuda'):
        """
        参数:
            model: 预测速度场的神经网络（例如一个条件 U-Net），
                   其前向接口应为 model(x, t, c) -> 形状与 x 相同的张量
            device: 训练/采样所用设备（'cuda' 或 'cpu'）

        初始化:
            - 将模型移动到指定设备，便于后续张量运算和反向传播。
        """
        self.model = model.to(device)
        self.device = device

    def compute_loss(self, x1, c):
        """
        计算一次 Flow Matching 的训练损失（MSE）。

        参数:
            x1: 真实数据样本，形状 [B, C, H, W]
                （B 为 batch 大小，C/H/W 为通道/高/宽）
            c:  条件标签（如类别 id），形状 [B]

        返回:
            loss: 标量张量，预测速度与目标速度的均方误差（MSE）

        训练流程（一次前向）:
        1) 从均匀分布采样时间步 t ~ U[0, 1]，形状 [B]。
        2) 从标准高斯采样噪声 x0 ~ N(0, I)，形状与 x1 相同。
        3) 按直线桥接构造中间样本：
               x_t = t * x1 + (1 - t) * x0。
           注意 t 的形状需通过广播扩展到 [B, 1, 1, 1] 才能与 [B, C, H, W] 相乘。
        4) 目标速度 v_target = x1 - x0（与 t 无关；Rectified Flow 的显著优点）。
        5) 用模型预测速度 v_pred = model(x_t, t, c)。
        6) 计算 MSE(v_pred, v_target) 作为损失，用于反向传播更新参数。
        """
        batch_size = x1.shape[0]

        # 采样随机时间步 t ~ U[0, 1]，维度 [B]
        t = torch.rand(batch_size, device=self.device)

        # 采样噪声 x0 ~ N(0, I)，形状与 x1 完全一致
        x0 = torch.randn_like(x1)

        # 沿“条件概率路径”构造中间状态 x_t：
        # x_t = t * x_1 + (1 - t) * x_0
        # 为了与 [B, C, H, W] 广播匹配，把 t 扩展为 [B, 1, 1, 1]
        t_expanded = t[:, None, None, None]  # [B, 1, 1, 1]
        xt = t_expanded * x1 + (1 - t_expanded) * x0

        # 目标速度（真实条件速度）：v_t = x_1 - x_0
        # 这是 Rectified Flow 的关键：目标与 t 无关，训练更稳定
        target_v = x1 - x0

        # 模型在 (x_t, t, c) 处预测速度场
        predicted_v = self.model(xt, t, c)

        # 用均方误差作为监督损失
        loss = nn.functional.mse_loss(predicted_v, target_v)

        return loss

    @torch.no_grad()
    def sample(self, c, num_steps=100, img_size=(1, 28, 28)):
        """
        用显式欧拉法（Euler）积分 ODE 进行采样/生成。

        参数:
            c:         条件标签（类别 id 等），形状 [B]
            num_steps: 时间离散步数。越大越精细，但耗时越长。
                       欧拉法是一阶方法，可按需换成 Heun/RK4 提升质量和稳定性。
            img_size:  生成图像的形状 (C, H, W)。需与模型的输入/输出通道一致。

        返回:
            生成的样本张量，形状 [B, C, H, W]

        采样思路:
            1) 从基分布（标准高斯）采样初态 x(0) = x0。
            2) 沿时间 0 → 1 积分 ODE：dx/dt = v_theta(x, t, c)。
               欧拉一步：x_{t+Δt} = x_t + Δt * v_theta(x_t, t, c)。
            3) 积分结束得到 x(1)，即模型的生成结果。
        """
        batch_size = c.shape[0]

        # 初始状态：从标准高斯噪声开始（对应 x_0）
        x = torch.randn(batch_size, *img_size, device=self.device)

        # 在 [0, 1] 做均匀的时间步长
        dt = 1.0 / num_steps

        # 沿 ODE 前向积分。此处使用欧拉法（简单但偏差相对大）。
        # 若出现不稳定/欠拟合，可增大 num_steps 或换更高阶积分器。
        for i in range(num_steps):
            # 当前离散时间 t_i = i * Δt；形状 [B]
            # 这里每个样本的 t 相同，但也可以在 batch 内使用不同比例的 t
            t = torch.ones(batch_size, device=self.device) * i * dt

            # 在当前 (x, t, c) 处预测速度
            v = self.model(x, t, c)

            # 欧拉更新：x_{t+Δt} = x_t + v * Δt
            x = x + v * dt

        return x

    @torch.no_grad()
    def sample_trajectory(self, c, num_steps=100, img_size=(1, 28, 28)):
        """
        生成样本的同时，返回整个积分轨迹（便于可视化 or 调试）。

        参数:
            c:         条件标签，形状 [B]
            num_steps: 时间离散步数
            img_size:  生成图像的形状 (C, H, W)

        返回:
            trajectory: Python 列表，长度为 num_steps+1。
                        trajectory[k] 是时间 t=k/num_steps 时刻的样本，形状 [B, C, H, W]。
                        为节省显存、便于后处理，这里将每个状态移动到 CPU（.cpu()）。
        """
        batch_size = c.shape[0]

        # 以标准高斯噪声为初态
        x = torch.randn(batch_size, *img_size, device=self.device)
        trajectory = [x.cpu()]  # 保存 t=0 的状态

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.ones(batch_size, device=self.device) * i * dt
            v = self.model(x, t, c)
            x = x + v * dt
            # 为方便可视化/GIF/日志记录，存一份到 CPU
            trajectory.append(x.cpu())

        return trajectory


if __name__ == '__main__':
    from model import UNet

    # 简单自检：构建模型与训练器，在随机数据上跑通一次前向和采样
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet()
    cfm = ConditionalFlowMatching(model, device=device)

    # 构造假的真实数据（x1）与条件标签（c）
    x1 = torch.randn(4, 1, 28, 28).to(device)  # 假设数据分布的一个 batch
    c = torch.randint(0, 10, (4,)).to(device)  # MNIST 类别标签示例

    # 计算一次训练损失（会走前向图，但此处未做反向与优化，仅用于验证代码路径）
    loss = cfm.compute_loss(x1, c)
    print(f"Loss: {loss.item():.4f}")

    # 进行一次快速采样测试（步数较小，仅验证接口）
    c_test = torch.tensor([0, 1, 2, 3]).to(device)
    samples = cfm.sample(c_test, num_steps=10)
    print(f"Sample shape: {samples.shape}")
