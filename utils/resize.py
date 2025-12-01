import torch
import torchvision.transforms.functional as F
from torchvision.transforms import transforms
from PIL import Image
import math

class ResizeToGivenSize:
    """
    将图片处理到给定的size (h, w)
    按图片的w对齐给定的size的w, 如果对齐后h超过给定的size就中心crop, 否则就补上黑边
    """
    
    def __init__(self, size):
        """
        参数:
            size (tuple): 目标尺寸 (height, width)
        """
        self.size = size
        self.target_h, self.target_w = size
    
    def __call__(self, img):
        """
        处理单张图片
        
        参数:
            img (PIL Image or Tensor): 输入图片
            
        返回:
            PIL Image or Tensor: 处理后的图片
        """
        # 获取原始图片尺寸
        if isinstance(img, torch.Tensor):
            _, orig_h, orig_w = img.shape
        else:
            orig_w, orig_h = img.size
        
        # 计算等比例缩放后的高度
        scale_factor = self.target_w / orig_w
        new_h = int(orig_h * scale_factor)
        
        # 等比例缩放到目标宽度
        if isinstance(img, torch.Tensor):
            # 对于Tensor，使用interpolate进行缩放
            img_resized = F.resize(img, [new_h, self.target_w], 
                                 interpolation=F.InterpolationMode.BICUBIC)
        else:
            # 对于PIL Image，使用resize
            img_resized = img.resize((self.target_w, new_h), Image.BICUBIC)
        
        # 判断是否需要crop或pad
        if new_h > self.target_h:
            # 高度超过目标尺寸，进行中心裁剪
            if isinstance(img_resized, torch.Tensor):
                # Tensor的中心裁剪
                start_h = (new_h - self.target_h) // 2
                img_final = img_resized[:, start_h:start_h+self.target_h, :]
            else:
                # PIL Image的中心裁剪
                start_h = (new_h - self.target_h) // 2
                img_final = F.crop(img_resized, start_h, 0, self.target_h, self.target_w)
        
        elif new_h < self.target_h:
            # 高度不足，进行上下黑边填充
            pad_top = (self.target_h - new_h) // 2
            pad_bottom = self.target_h - new_h - pad_top
            pad_left = 0
            pad_right = 0
            img_final = F.pad(img_resized, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
        else:
            # 高度正好等于目标尺寸
            img_final = img_resized
        
        return img_final
    
    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size})"
    

class ResizeWithPadding:
    """
    将图像调整到指定尺寸并保持原始比例，不足部分用黑边填充
    
    Args:
        size (tuple): 目标尺寸 (height, width)
        interpolation (int, optional): 插值方法，默认为 Image.BILINEAR
    """
    
    def __init__(self, size, interpolation=Image.BICUBIC):
        assert isinstance(size, (int, tuple, list))
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size  # (height, width)
        self.interpolation = interpolation
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image or Tensor): 输入图像
            
        Returns:
            PIL Image or Tensor: 调整后的图像
        """
        # 获取原始尺寸
        if isinstance(img, torch.Tensor):
            # 如果是张量，假设形状为 (C, H, W)
            original_height, original_width = img.shape[-2], img.shape[-1]
            is_tensor = True
            # 转换为PIL图像进行处理
            img_pil = F.to_pil_image(img)
        else:
            # 如果是PIL图像
            original_width, original_height = img.size
            img_pil = img
            is_tensor = False
        
        target_height, target_width = self.size
        
        # 计算缩放比例
        scale = min(target_height / original_height, target_width / original_width)
        
        # 计算缩放后的尺寸
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # 缩放图像
        img_resized = img_pil.resize((new_width, new_height), self.interpolation)
        
        # 创建目标图像（黑底）
        if img_resized.mode == 'RGBA':
            # 如果是RGBA模式，创建透明背景
            result = Image.new('RGBA', (target_width, target_height), (0, 0, 0, 0))
        else:
            # RGB或其他模式，创建黑色背景
            result = Image.new(img_resized.mode, (target_width, target_height), (0, 0, 0))
        
        # 计算粘贴位置（居中）
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        
        # 将缩放后的图像粘贴到目标图像上
        result.paste(img_resized, (paste_x, paste_y))
        
        # 如果是张量输入，转换回张量
        if is_tensor:
            result = F.to_tensor(result)
        
        return result
    
    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, self.interpolation)
    


# 示例用法
if __name__ == "__main__":
    # 处理PIL图片
    img_pil = Image.open("Hi.jpg")

    transform = transforms.Compose([
        ResizeWithPadding((518, 392)),
    ])
    img_resized_pil = transform(img_pil)
    img_resized_pil.show()  # 显示处理后的图片