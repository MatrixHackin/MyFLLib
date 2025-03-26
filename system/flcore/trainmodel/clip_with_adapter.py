from torch import nn
import torch
import clip
import torch.nn.functional as F

class CLIPWithAdapter(nn.Module):
    # 支持的CLIP pretrained模型
    CLIP_MODELS = [
            'RN50',
            'RN101',
            'RN50x4',
            'RN50x16',
            'RN50x64',
            'ViT-B/32',
            'ViT-B/16',
            'ViT-L/14',
            'ViT-L/14@336px'
        ]
    input_texts=[
        # 'a picture of a normal fundus',
        # 'a picture of a mild non-proliferative diabetic retinopathy fundus',
        # 'a picture of a moderate non-proliferative diabetic retinopathy fundus',
        # 'a picture of a severe non-proliferative diabetic retinopathy fundus',
        # 'a picture of an early proliferative diabetic retinopathy',
        # 'a picture of a late proliferative diabetic retinopathy'
        'a picture of normal fundus',
        'a picture of age-related macular degeneration funuds'
    ]
    
    
    def __init__(self, clip_model_name='ViT-B/32'):
        super().__init__()

        # 检查模型是否支持
        if clip_model_name not in self.CLIP_MODELS:
            raise ValueError(f"Unsupported CLIP model: {clip_model_name}, supported models: {self.CLIP_MODELS}")
        
        # 加载原始CLIP模型和对应的预处理器，这个预处理器会把PIL图像转换为模型所需的格式
        self.clip_model, self.preprocess = clip.load(name=clip_model_name)
        print(f'CLIP model {clip_model_name} loaded')
        print(f'preprocess: {self.preprocess}') # 输出预处理器信息
        
        # 冻结所有CLIP参数梯度
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.clip_model.visual.class_embedding.requires_grad = True
        self.clip_model.visual.positional_embedding.requires_grad = True
        self.clip_model.visual.proj.requires_grad = True

        #tokenize
        self.text_tokens=clip.tokenize(self.input_texts).to('cuda')
        
        # 获取各模态维度
        text_dim = self.clip_model.text_projection.shape[1] # 文本特征维度

        # 为视觉编码器和文本编码器添加Adapter,具体结构可以在类里面修改
        self.image_adapter1 = ImageAdapter(input_dim=1024).half()
        self.image_adapter2=ImageAdapter(input_dim=1024).half()
        self.text_adapter = TextAdapter(input_dim=text_dim).half()

        self.subimage_adapters=[]
        for i in range(0,12):
            self.subimage_adapters.append(ImageAdapter(input_dim=1024).half())
        
    def forward(self, pixel_values):
        x=self.clip_model.visual.conv1(pixel_values)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2] (10*1024*49)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] (10*49*1024)

        # 加入类别嵌入
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        # 加入位置嵌入
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)

        # layer norm
        x = self.clip_model.visual.ln_pre(x) #10*50*1024

        x=self.image_adapter1(x).to(torch.float16)

        x = x.permute(1, 0, 2)  # NLD -> LND

        x=self.clip_model.visual.transformer(x)

        x = x.permute(1, 0, 2)
        x=self.image_adapter2(x)

        x=self.image_adapter1(x).to(torch.float16)

        x=self.clip_model.visual.ln_post(x[:,0,:])
        if self.clip_model.visual.proj is not None:
            x=x@self.clip_model.visual.proj

        # image_features = self.clip_model.encode_image(pixel_values)  # [B, vision_dim]
        # image_features = self.image_adapter(image_features)          # 经过Adapter
        image_features = x/ x.norm(dim=-1, keepdim=True)   # 归一化
        # 文本特征处理
        text_features = self.clip_model.encode_text(self.text_tokens)       # [num_classes, text_dim]
        text_features = self.text_adapter(text_features)             # 经过Adapter
        text_features = text_features/ text_features.norm(dim=-1, keepdim=True)  
        # 处理两个特征得到预测概率,这里的100可以做退火
        similarity = (100.0 * image_features @ text_features.T)
        return similarity   # [B, num_classes]用来计算对称损失

class ImageAdapter(nn.Module):
    def __init__(self, input_dim, reduction_factor=4):
        super().__init__()
        hidden_dim = input_dim // reduction_factor
        self.attention = nn.MultiheadAttention(input_dim, 4, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        aten_output, _ = self.attention(x,x,x)
        x = x + self.norm1(aten_output)
        
        x = x + self.norm2(self.mlp(x))
        return x

class TextAdapter(nn.Module):
    def __init__(self, input_dim, reduction_factor=4):
        super().__init__()
        hidden_dim = input_dim // reduction_factor
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        return x + self.norm(self.model(x))