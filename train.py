import torch
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from model import UNet
from flow_matching import ConditionalFlowMatching
from data_loader import get_mnist_dataloaders


def train(
    epochs=50,
    batch_size=128,
    lr=1e-4,
    device='cuda',
    data_path='../Data',
    checkpoint_dir='checkpoints',
    log_interval=100,
    save_interval=10,
):
    """
    训练条件式 Flow Matching（CFM）模型。

    参数:
        epochs: 训练总轮数（对整个训练集迭代的次数）
        batch_size: 训练时每个批次包含的样本数量
        lr: 学习率（优化器的步长）
        device: 训练设备（'cuda' 优先使用 GPU；若不可用则自动回退为 'cpu'）
        data_path: MNIST 数据集所在根目录（PyTorch 会在该目录下寻找 MNIST 子目录）
        checkpoint_dir: 用于保存模型权重与优化器状态的目录
        log_interval: 日志打印间隔（以「训练步」为单位，每隔多少步在进度条中显示一次 loss）
        save_interval: 保存 checkpoint 的间隔（以「训练轮」为单位，每隔多少个 epoch 保存一次）

    训练流程概述:
        1) 准备设备、数据与模型（U-Net 作为速度场预测器）。
        2) 用 ConditionalFlowMatching 封装训练逻辑，计算 MSE 损失：
           - 采样 t~U[0,1]、x0~N(0,I)，构造 x_t = t*x1 + (1-t)*x0
           - 目标速度 v* = x1 - x0
           - 回归模型预测的速度 v_theta(x_t, t, c) 到 v*
        3) 反向传播并用 Adam 更新参数。
        4) 定期保存 checkpoint，并用当前模型生成条件样本以便可视化质量。
        5) 训练结束后保存最终模型与训练损失曲线。
    """
    # 创建（若不存在则新建）checkpoint 目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 设备选择：优先使用 GPU；不可用则回退到 CPU
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载 MNIST 数据（train_loader / test_loader）
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        num_workers=0 if device.type == 'cpu' else 4  # CPU 上避免多进程开销，GPU 上提高数据吞吐
    )

    # 构建模型（U-Net 作为速度场预测网络）
    print("Creating model...")
    model = UNet(
        in_channels=1,
        out_channels=1,
        num_classes=10,
        base_channels=64,
        time_emb_dim=128,
        cond_emb_dim=64,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # 构建 Flow Matching 训练器（封装损失计算与采样）
    cfm = ConditionalFlowMatching(model, device=device)

    # 选择优化器（Adam）
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ========= 训练循环开始 =========
    print("Starting training...")
    global_step = 0
    train_losses = []  # 记录每步的损失，用于绘制曲线

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for batch_idx, (x, c) in enumerate(pbar):
            # 将数据与标签移动到目标设备
            x = x.to(device)
            c = c.to(device)

            # 前向计算：使用 CFM 的损失定义（回归速度 v = x1 - x0）
            loss = cfm.compute_loss(x, c)

            # 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录日志
            epoch_loss += loss.item()
            train_losses.append(loss.item())
            global_step += 1

            # 按设定步数间隔在进度条上显示当前 loss
            if global_step % log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 该轮训练的平均损失（越低越好）
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")

        # 按设定轮数间隔保存一次 checkpoint（包含模型+优化器状态与平均损失）
        if epoch % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            # 同步生成一批可视化样本，便于观测训练进展
            generate_samples(cfm, epoch, checkpoint_dir, device)

    # 训练完成后保存最终模型（仅保存最后权重与优化器状态）
    final_path = os.path.join(checkpoint_dir, 'model_final.pt')
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    print(f"Final model saved: {final_path}")

    # 绘制训练损失曲线（按训练步）
    plot_loss(train_losses, checkpoint_dir)


def generate_samples(cfm, epoch, save_dir, device):
    """使用当前模型按类别生成一组样本并保存到磁盘。"""
    cfm.model.eval()  # 推理模式（关闭 Dropout/BN 的训练行为）

    # 生成每个类别（0~9）各一张图，方便粗看条件生成效果
    c = torch.arange(10).to(device)

    # 用 ODE 积分采样（默认步数 100），返回形状 [10, 1, H, W]
    samples = cfm.sample(c, num_steps=100)

    # 反归一化：训练数据通常预处理为 [-1, 1]，这里变回 [0, 1] 以便可视化
    samples = (samples + 1) / 2
    samples = samples.clamp(0, 1)  # 数值安全裁剪

    # 绘图（每个类别一列）
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        axes[i].imshow(samples[i, 0].cpu().numpy(), cmap='gray')
        axes[i].set_title(f'{i}')
        axes[i].axis('off')  # 隐藏坐标轴

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'samples_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Samples saved: {save_path}")


def plot_loss(losses, save_dir):
    """绘制训练损失曲线（横轴为训练步，纵轴为 loss）。"""
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    save_path = os.path.join(save_dir, 'loss_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved: {save_path}")


if __name__ == '__main__':
    # ==================== 配置参数 ====================
    # 可直接修改下方字典进行训练配置
    CONFIG = {
        'epochs': 25,              # 训练轮数
        'batch_size': 128,         # 批次大小（每步处理的样本数）
        'lr': 1e-4,                # 学习率（建议从 1e-4 起试）
        'device': 'cuda',          # 设备 ('cuda' 或 'cpu')
        'data_path': '../Data',    # MNIST 数据集路径（PyTorch 会自动查找 MNIST 子目录）
        'checkpoint_dir': 'checkpoints',  # checkpoint 保存目录
        'log_interval': 100,       # 日志打印间隔（按训练步）
        'save_interval': 10,       # checkpoint 保存间隔（按训练轮）
    }
    # =================================================

    # 启动训练
    train(
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        lr=CONFIG['lr'],
        device=CONFIG['device'],
        data_path=CONFIG['data_path'],
        checkpoint_dir=CONFIG['checkpoint_dir'],
        log_interval=CONFIG['log_interval'],
        save_interval=CONFIG['save_interval'],
    )
