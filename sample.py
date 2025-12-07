import torch
import matplotlib.pyplot as plt
import os
import numpy as np

from model import UNet
from flow_matching import ConditionalFlowMatching

"""
============================ 采样原理（Flow Matching / Rectified Flow）============================
我们训练得到的是一个时间依赖的“速度场” v_θ(x, t, c)，它给出在时刻 t、位置 x（带条件 c）时，
样本应当朝哪个方向、以多大“速度”移动。采样时，我们把这个速度场当作常微分方程（ODE）的
右端项来积分，具体为：

    dx/dt = v_θ(x(t), t, c),        t ∈ [0, 1]

初态 x(0) 来自一个易采样的基分布（本例使用标准高斯噪声 N(0, I)）。
当把 ODE 从 t=0 数值积分到 t=1 后，x(1) 就是生成的样本。

本项目采用最常见的“直线桥接（Rectified Flow）”进行训练：
    x_t = t * x_1 + (1 - t) * x_0,   其中 x_0 ~ N(0, I), x_1 ~ p_data
其真实条件速度为 d/dt x_t = x_1 - x_0。训练时我们用 MSE 回归 v_θ(x_t, t, c) 到 (x_1 - x_0)。
可以证明在期望意义下，这样学到的 v_θ 逼近“边缘速度”（无需显式估计 score）。

数值积分方法：
- 本脚本默认使用显式欧拉（Euler）法：
    x_{k+1} = x_k + Δt * v_θ(x_k, t_k, c)
  该方法简单、快速，但会有一定的截断误差。你可以将采样器替换为 Heun（改进欧拉）或 RK4
  以获得更稳定/更高质量的生成（代价是更多的模型前向调用）。
- “步数 num_steps”越大，时间离散越细，偏差越小，但速度越慢；通常可在质量与速度之间权衡。

归一化与反归一化（MNIST）：
- 我们在数据加载时把像素线性归一化到 [-1, 1]（Normalize((0.5,), (0.5,))）。
- 为了可视化，需要把模型输出样本从 [-1, 1] 反映射回 [0, 1]，再用 matplotlib 显示或保存。

条件生成：
- 条件标签 c（如类别 0~9）被送入网络（通过嵌入/调制等方式），使得 v_θ 对应类别敏感。
- 采样时传入不同的 c，即可生成对应类别的数字图片。
==============================================================================================
"""


def load_model(checkpoint_path, device='cuda'):
    """加载已训练的模型（U-Net）并切到推理模式。

    参数:
        checkpoint_path: checkpoint 文件路径（包含 model_state_dict 等）
        device:          加载到的设备（'cuda' 或 'cpu'）

    返回:
        model: 已加载好权重并处于 eval() 模式的 U-Net 模型
    """
    model = UNet(
        in_channels=1,
        out_channels=1,
        num_classes=10,
        base_channels=64,
        time_emb_dim=128,
        cond_emb_dim=64,
    )

    # 使用 map_location 将权重加载到指定设备（避免跨设备错误）
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # 推理模式（冻结 BN/Dropout 的训练行为）

    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Trained for {checkpoint['epoch']} epochs")

    return model


def generate_samples(checkpoint_path, num_samples=100, num_steps=100, device='cuda', save_dir='samples'):
    """
    生成随机样本（各类别均衡）并保存为网格图。

    采样原理（简述）：
    1) 构建 ConditionalFlowMatching 包装器（其内部持有 v_θ 模型）。
    2) 为每个类别 c ∈ {0,...,9} 生成相同数量的初始噪声样本 x(0) ~ N(0, I)。
    3) 沿 ODE：dx/dt = v_θ(x, t, c)，用欧拉法把 t 从 0 积分到 1，得到 x(1)。
    4) 把结果从 [-1, 1] 反归一化到 [0, 1]，按类别排成网格保存。

    参数:
        checkpoint_path: checkpoint 文件路径
        num_samples:     生成的总样本数（建议为 10 的倍数，方便各类均衡）
        num_steps:       ODE 时间离散步数（越大越精细，代价是更多前向调用）
        device:          采样设备
        save_dir:        生成图片的保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1) 加载权重并包装成 CFM（CFM 会在其 __init__ 中将模型迁移到 device）
    model = load_model(checkpoint_path, device)
    cfm = ConditionalFlowMatching(model, device=device)

    # 2) 构造均衡的条件标签序列：每个类别各 samples_per_class 张
    samples_per_class = num_samples // 10
    c = torch.arange(10).repeat(samples_per_class).to(device)  # 形状 [num_samples]

    # 3) ODE 采样（欧拉法）：x_{k+1} = x_k + Δt * v_θ(x_k, t_k, c)
    print(f"Generating {num_samples} samples with {num_steps} steps...")
    samples = cfm.sample(c, num_steps=num_steps)  # [N, 1, H, W]，N=num_samples

    # 4) 从 [-1, 1] 反归一化回 [0, 1]，并裁剪到合法范围以防数值溢出
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    samples = samples.cpu().numpy()

    # 5) 按类别列、样本行为布局成网格
    rows = samples_per_class  # 行数 = 每类样本的数量
    cols = 10                 # 列数 = 10 个类别

    fig, axes = plt.subplots(rows, cols, figsize=(20, 2 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]  # 统一成二维索引

    # 网格索引：第 j 列是类别 j；第 i 行是该类别的第 i 张
    # 在一维数组中的下标 = j * samples_per_class + i
    for i in range(rows):
        for j in range(cols):
            idx = j * samples_per_class + i
            axes[i, j].imshow(samples[idx, 0], cmap='gray')
            if i == 0:
                axes[i, j].set_title(f'Class {j}', fontsize=12)
            axes[i, j].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'generated_samples.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved samples to {save_path}")
    plt.close()


def visualize_generation_process(checkpoint_path, class_label=5, num_steps=100, device='cuda', save_dir='samples'):
    """
    可视化“从噪声到图像”的生成过程（轨迹可视化）。

    做法：
    1) 只生成某一类别（class_label）的一张图片，调用 cfm.sample_trajectory 得到
       整条 ODE 积分轨迹 {x(0), x(Δt), ..., x(1)}。
    2) 从轨迹中抽取若干帧（等间隔），将每一帧反归一化后并排显示。
    3) 从可视化可以直观看到：图像由纯噪声，逐步“在速度场的推动下”变得清晰。

    参数:
        checkpoint_path: checkpoint 路径
        class_label:     指定要生成的数字类别（0~9）
        num_steps:       ODE 时间离散步数
        device:          设备
        save_dir:        保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 加载模型并包装 CFM
    model = load_model(checkpoint_path, device)
    cfm = ConditionalFlowMatching(model, device=device)

    # 生成该类别的整条轨迹（列表，长度 num_steps+1）
    c = torch.tensor([class_label]).to(device)
    print(f"Generating class {class_label} with trajectory...")
    trajectory = cfm.sample_trajectory(c, num_steps=num_steps)

    # 选取可视化帧数（例如取 10 帧，等间隔抽取）
    num_frames = 10
    frame_indices = np.linspace(0, len(trajectory) - 1, num_frames, dtype=int)

    fig, axes = plt.subplots(1, num_frames, figsize=(20, 2))

    for i, idx in enumerate(frame_indices):
        # trajectory[k] 张量位于 CPU，形状 [B, C, H, W]；这里只有 B=1
        frame = trajectory[idx][0, 0].numpy()
        # 反归一化到 [0, 1]
        frame = (frame + 1) / 2
        frame = np.clip(frame, 0, 1)

        axes[i].imshow(frame, cmap='gray')
        axes[i].set_title(f't={idx/num_steps:.2f}', fontsize=10)  # 显示对应的时间比例
        axes[i].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'generation_process_class_{class_label}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved generation process to {save_path}")
    plt.close()


def generate_specific_digits(checkpoint_path, digits, num_per_digit=10, num_steps=100, device='cuda', save_dir='samples'):
    """
    只生成给定的若干类别，并按（行=样本编号，列=类别）布局保存。

    采样流程与 generate_samples 相同：对每个条件 c，用 ODE（欧拉离散）把噪声推到数据分布。
    此函数便于对某几个目标类别做更细致的观察（比如 0/1/2/3/4）。

    参数:
        checkpoint_path: checkpoint 路径
        digits:          要生成的类别列表（如 [0, 1, 2, 3, 4]）
        num_per_digit:   每个类别生成多少张
        num_steps:       ODE 时间离散步数
        device:          设备
        save_dir:        保存目录
    """
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 加载模型并包装 CFM
    model = load_model(checkpoint_path, device)
    cfm = ConditionalFlowMatching(model, device=device)

    # 构造条件标签序列：digits 中每个类别重复 num_per_digit 次
    c = torch.tensor([d for d in digits for _ in range(num_per_digit)]).to(device)

    print(f"Generating {len(c)} samples for digits {digits}...")
    samples = cfm.sample(c, num_steps=num_steps)  # [len(c), 1, H, W]

    # 反归一化到 [0, 1]，并搬到 CPU 以便 numpy/绘图
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)
    samples = samples.cpu().numpy()

    # 网格：行 = 每类样本编号（0...num_per_digit-1），列 = 类别顺序
    num_digits = len(digits)
    fig, axes = plt.subplots(num_per_digit, num_digits, figsize=(2 * num_digits, 2 * num_per_digit))

    if num_per_digit == 1:
        axes = axes[np.newaxis, :]   # 统一形状
    if num_digits == 1:
        axes = axes[:, np.newaxis]

    for i in range(num_per_digit):
        for j in range(num_digits):
            # 线性索引：第 j 个类别的第 i 张
            idx = j * num_per_digit + i
            axes[i, j].imshow(samples[idx, 0], cmap='gray')
            if i == 0:
                axes[i, j].set_title(f'Digit {digits[j]}', fontsize=12)
            axes[i, j].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'specific_digits_{"_".join(map(str, digits))}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved specific digits to {save_path}")
    plt.close()


if __name__ == '__main__':
    # ==================== 配置参数 ====================
    # 可以直接修改这里的参数来配置采样

    # 选择运行模式: 'generate', 'process', 'specific'
    # - 'generate': 生成所有类别的均衡样本（0-9），并保存为一个大网格图
    # - 'process':  可视化单个类别从噪声到清晰图像的“时间演化”轨迹
    # - 'specific': 只生成指定的若干类别，按（列=类别，行=样本编号）排布
    MODE = 'process'

    # 通用配置
    CONFIG = {
        'checkpoint_path': 'checkpoints/model_epoch_50.pt',  # checkpoint 路径
        'device': 'cuda',                                    # 设备 ('cuda' 或 'cpu')
        'num_steps': 100,                                    # ODE 积分步数（越大越精细）
        'save_dir': 'samples',                               # 保存目录
    }

    # 各模式的特定配置
    GENERATE_CONFIG = {
        'num_samples': 100,  # 总生成样本数（建议为 10 的倍数，确保按类均衡）
    }

    PROCESS_CONFIG = {
        'class_label': 5,    # 要可视化轨迹的数字类别（0-9）
    }

    SPECIFIC_CONFIG = {
        'digits': [0, 1, 2, 3, 4],  # 要生成的特定类别列表
        'num_per_digit': 10,        # 每个类别生成的张数
    }
    # =================================================

    # 根据模式运行相应的函数
    if MODE == 'generate':
        generate_samples(
            checkpoint_path=CONFIG['checkpoint_path'],
            num_samples=GENERATE_CONFIG['num_samples'],
            num_steps=CONFIG['num_steps'],
            device=CONFIG['device'],
            save_dir=CONFIG['save_dir']
        )
    elif MODE == 'process':
        visualize_generation_process(
            checkpoint_path=CONFIG['checkpoint_path'],
            class_label=PROCESS_CONFIG['class_label'],
            num_steps=CONFIG['num_steps'],
            device=CONFIG['device'],
            save_dir=CONFIG['save_dir']
        )
    elif MODE == 'specific':
        generate_specific_digits(
            checkpoint_path=CONFIG['checkpoint_path'],
            digits=SPECIFIC_CONFIG['digits'],
            num_per_digit=SPECIFIC_CONFIG['num_per_digit'],
            num_steps=CONFIG['num_steps'],
            device=CONFIG['device'],
            save_dir=CONFIG['save_dir']
        )
    else:
        print(f"未知模式: {MODE}，请选择 'generate', 'process', 或 'specific'")
