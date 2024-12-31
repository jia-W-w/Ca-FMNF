import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import glob
import pydicom
from skimage.transform import resize
from basicsr.models.archs.LEARN import LEARN, ProjOperator, FBPOperator
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio 
from skimage.metrics import structural_similarity as ssim1
from skimage.metrics import mean_squared_error

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def compare(recon0, recon1, verbose=True):
    mse_recon = mean_squared_error(recon0, recon1)
    # np.mean((recon0-recon1)**2)

    small_side = np.min(recon0.shape)
    if small_side < 7:
        if small_side % 2:  # if odd
            win_size = small_side
        else:
            win_size = small_side - 1
    else:
        win_size = None

    ssim_recon = ssim1(recon0, recon1,
                       data_range=recon0.max() - recon0.min(), win_size=win_size)

    psnr_recon = peak_signal_noise_ratio(recon0, recon1,
                                         data_range=recon0.max() - recon0.min())

    if verbose:
        err_string = 'MSE: {:.8f}, SSIM: {:.3f}, PSNR: {:.3f}'
        print(err_string.format(mse_recon, ssim_recon, psnr_recon))
    return (mse_recon, ssim_recon, psnr_recon)

def clear(x):
  x = x.detach().cpu().squeeze().numpy()
  return x

class DicomDataset256(Dataset):
    def __init__(self, root_dir, transform=None):
        super(DicomDataset256, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.size = 256
        self.file_list = glob.glob(os.path.join(self.root_dir, '**/*.IMA'), recursive=True)
        
        # 初始化投影和FBP算子
        self.projector = ProjOperator(
            N=256, M=256,
            pixel_size_x=0.15, pixel_size_y=0.15,
            det_pixels=624, det_pixel_size=0.2,
            angles=20,
            src_origin=59, det_origin=49
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 读取和预处理图像
        img_path = self.file_list[idx]
        image = pydicom.read_file(img_path).pixel_array
        image = np.float32(image)
        image[image > 3000] = 0
        image = image / 3000
        image = resize(image, (256, 256))
        image[image < 0] = 0
        
        # 转换为tensor并添加通道维度
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image)
            
        # 生成投影数据
        proj = self.projector(image)
        
        return image, proj

# 加载数据集
dataset = DicomDataset256('/home/wujia/daima/raw_data/full_3mm/L***/full_3mm')
val_dataset = DicomDataset256('/home/wujia/daima/raw_data/test/L***/full_3mm')
loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

# 初始化模型和优化器
model_restoration = LEARN(block_num=10).to(device)
l2_loss = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_restoration.parameters(), lr=1e-4, betas=(0.9, 0.999))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# 训练循环
num_epochs = 100
val_freq = 1
best_psnr = 0.0

for epoch in range(num_epochs):
    model_restoration.train()
    train_loss = 0.0
    train_metrics = {'mse': 0.0, 'ssim': 0.0, 'psnr': 0.0}
    total_samples = 0
    
    train_pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}, Train')
    for image, proj in train_pbar:
        batch_size = image.size(0)
        total_samples += batch_size
        
        image = image.to(device)
        proj = proj.to(device)
        
        # 使用FBP生成初始重建
        fbp = FBPOperator(
            N=256, M=256,
            pixel_size_x=0.15, pixel_size_y=0.15,
            det_pixels=624, det_pixel_size=0.2,
            angles=20,
            src_origin=59, det_origin=49,
            filter_type='Hann', frequency_scaling=1
        )
        input_noise = fbp(proj)
        
        optimizer.zero_grad()
        output = model_restoration(input_noise, proj)
        loss = l2_loss(output, image)
        loss.backward()
        optimizer.step()
        
        # 更新指标
        train_loss += loss.item() * batch_size
        mse, ssim, psnr = compare(clear(image), clear(output), verbose=False)
        train_metrics['mse'] += mse * batch_size
        train_metrics['ssim'] += ssim * batch_size
        train_metrics['psnr'] += psnr * batch_size
        
        # 更新进度条
        avg_metrics = {k: v/total_samples for k, v in train_metrics.items()}
        train_pbar.set_postfix({'loss': train_loss/total_samples, **avg_metrics})
    
    # 验证循环
    if (epoch + 1) % val_freq == 0:
        model_restoration.eval()
        val_loss = 0.0
        val_metrics = {'mse': 0.0, 'ssim': 0.0, 'psnr': 0.0}
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}, Val')
            for image, proj in val_pbar:
                image = image.to(device)
                proj = proj.to(device)
                input_noise = fbp(proj)
                output = model_restoration(input_noise, proj)
                
                loss = l2_loss(output, image)
                val_loss += loss.item() * image.size(0)
                
                mse, ssim, psnr = compare(clear(image), clear(output), verbose=False)
                val_metrics['mse'] += mse * image.size(0)
                val_metrics['ssim'] += ssim * image.size(0)
                val_metrics['psnr'] += psnr * image.size(0)
        
        # 计算平均指标
        val_loss /= len(val_dataset)
        val_metrics = {k: v/len(val_dataset) for k, v in val_metrics.items()}
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(dataset):.4f}, Val Loss: {val_loss:.4f}')
        print(f'Val Metrics: {val_metrics}')
        
        # 保存最佳模型
        if val_metrics['psnr'] > best_psnr:
            best_psnr = val_metrics['psnr']
            model_path = '/home/wujia/daima/NeRP-main/上传代码/pre_trainmodel/best_model.pth'
            torch.save(model_restoration.state_dict(), model_path)
            print(f'Model saved at {model_path} with PSNR: {best_psnr:.4f}')
    scheduler.step()