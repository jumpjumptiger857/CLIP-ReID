from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim) 
        self.num_heads = num_heads

    def forward(self, x): 
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC  #32,2048,7,7 ->49, 32, 2048
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC  50,32,2048
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        ) 

        return x 

class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=1) 
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x): 
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype) 
        x = stem(x) 
        x = self.layer1(x) 
        x = self.layer2(x) 
        x3 = self.layer3(x) 
        x4 = self.layer4(x3) 
        xproj = self.attnpool(x4) 

        return x3, x4, xproj 


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# ResidualAttentionBlock =
# LayerNorm → Multi-Head Self-Attention → Residual Add → LayerNorm → MLP → Residual Add
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # 核心组件1：多头注意力层（d_model=特征维度，n_head=注意力头数）
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # 核心组件2：第一层归一化（注意力层输入前用）
        self.ln_1 = LayerNorm(d_model)
        # 核心组件3：MLP前馈网络（特征非线性变换）
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),      # 升维：d_model → 4*d_model
            ("gelu", QuickGELU()),                          # 非线性激活（比ReLU更平滑）
            ("c_proj", nn.Linear(d_model * 4, d_model))     # 降维：4*d_model → d_model
        ]))
        # 核心组件4：第二层归一化（MLP层输入前用）
        self.ln_2 = LayerNorm(d_model)
        # 注意力掩码（ViT中一般为None，所有token可互相关注）
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        # 掩码设备/数据类型对齐（避免CPU/GPU、float32/float16不匹配报错）
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # 调用多头注意力，仅返回特征输出（need_weights=False不返回注意力权重）
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        # 第一步：注意力分支 + 残差连接
        # ln_1(x)：先归一化，再做注意力；x + ... 是残差连接（原始输入+注意力输出）
        x = x + self.attention(self.ln_1(x))
        # 第二步：MLP分支 + 残差连接
        # ln_2(x)：归一化后做MLP；x + ... 再次残差连接
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width       # 特征维度（对应ViT的width=768，MultiheadAttention的d_model）
        self.layers = layers     # Transformer的层数（ViT中通常是12层）
        # nn.Sequential： 把多个 module 串联成一个大 module，forward 时 按顺序执行
        # self.resblocks 使用 nn.Sequential 将多个 ResidualAttentionBlock 顺序堆叠，构成标准的 Transformer Encoder。
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, h_resolution: int, w_resolution: int, patch_size: int, stride_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution*w_resolution + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, cv_emb = None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if cv_emb != None: 
            x[:,0] = x[:,0] + cv_emb
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        x11 = self.transformer.resblocks[:11](x) 
        x12 = self.transformer.resblocks[11](x11) 
        x11 = x11.permute(1, 0, 2)  # LND -> NLD  
        x12 = x12.permute(1, 0, 2)  # LND -> NLD  

        x12 = self.ln_post(x12)  

        if self.proj is not None:
            xproj = x12 @ self.proj   

        return x11, x12, xproj


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 vision_stride_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 h_resolution: int, 
                 w_resolution: int
                 ):
        super().__init__()

        self.context_length = context_length

        # 视觉分支：ResNet / ViT 二选一
        # 判断依据：tuple / list → ResNet-CLIP； int → ViT-CLIP
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=h_resolution*w_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                h_resolution = h_resolution,
                w_resolution = w_resolution,
                patch_size = vision_patch_size,
                stride_size = vision_stride_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )
            
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            # attention mask 是因果 Mask（Causal Mask），保证 token 只能看到自己之前的 token，用于文本自回归建模
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            # 处理ResNet的注意力池化层（attnpool）
            if self.visual.attnpool is not None:
                # 计算注意力层的初始化标准差：1/√输入维度（Xavier初始化的核心思想）
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                # 注意力层的Q/K/V/输出投影权重，均用该std初始化
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            # 处理ResNet的4个残差层（layer1-layer4）
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    # 找到所有bn3.weight（第三个批归一化层的权重），初始化为0
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        # 计算不同组件的初始化标准差
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        # 遍历所有Transformer残差块
        for block in self.transformer.resblocks:
            # 1. 注意力层的输入投影权重（in_proj_weight）：用attn_std
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            # 2. 注意力层的输出投影权重（out_proj.weight）：用proj_std
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            # 3. MLP的升维层（c_fc.weight）：用fc_std
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            # 4. MLP的降维层（c_proj.weight）：用proj_std
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # 文本投影层初始化
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf

        # 步骤1：创建空的二维掩码矩阵，形状为 [context_length, context_length]
        # context_length：序列长度（比如NLP中是文本长度，ViT中是patch数+1=197）
        mask = torch.empty(self.context_length, self.context_length)
        # 步骤2：将掩码矩阵所有元素填充为 -inf（负无穷）
        mask.fill_(float("-inf"))
        # 步骤3：triu_(1) → 保留上三角（对角线以上）为原值，下三角（含对角线）置0
        # triu_是pytorch的原地操作，参数1表示“对角线偏移量”：
        # - 偏移量=1：对角线（i==j）为0，i<j的位置（上三角）保留-inf，i>j的位置（下三角）置0
        # - 偏移量=0：对角线以上（含）为-inf，以下为0
        mask.triu_(1)  # zero out the lower diagonal
        # 步骤4：返回因果掩码矩阵
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text): 
        x = self.token_embedding(text).type(self.dtype)  

        x = x + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype) 

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection 

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


#这个函数的核心目标是：遍历模型的所有层，将卷积层、线性层、多头注意力层、投影层 的权重 / 偏置参数强制转换为 float32 类型，避免推理 / 训练时因数据类型不匹配（如 float16 和 float32 混合）导致的数值溢出或计算错误。
def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        # 第一步：转换卷积层（Conv1d/Conv2d）和线性层（Linear）的参数
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            # 权重转为float32
            l.weight.data = l.weight.data.float()
            # 偏置如果存在，也转为float32
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        # 第二步：转换多头注意力层（MultiheadAttention）的参数
        if isinstance(l, nn.MultiheadAttention):
            # 遍历注意力层的所有关键参数：输入投影权重、Q/K/V投影权重、偏置等
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                # 获取参数（如l.in_proj_weight、l.q_proj_weight等）
                tensor = getattr(l, attr)
                # 如果参数存在（非None），转为float32
                if tensor is not None:
                    tensor.data = tensor.data.float()

        # 第三步：转换CLIP的投影层参数（text_projection/visual.proj）
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, h_resolution: int, w_resolution: int, vision_stride_size: int):
    # 判断是不是 ViT 版本 CLIP
    # ViT-CLIP 有：visual.proj； RN50-CLIP 没有 visual.proj
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else: #RN50
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    # 图像 & 文本最终共享的 embedding 维度（通常 512）
    embed_dim = state_dict["text_projection"].shape[1]
    # 最大 token 数（CLIP 固定 77）
    context_length = state_dict["positional_embedding"].shape[0] #77 (77,512)
    # BPE 词表大小（49408）
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    # 文本 Transformer block 数（通常 12）
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size, vision_stride_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        h_resolution, w_resolution
    )
    # 调整位置嵌入：适配自定义的h_resolution/w_resolution
    if vit:
        state_dict["visual.positional_embedding"] = resize_pos_embed(state_dict["visual.positional_embedding"], model.visual.positional_embedding, h_resolution, w_resolution)
    else: #RN50
        state_dict["visual.attnpool.positional_embedding"] = resize_pos_embed(state_dict["visual.attnpool.positional_embedding"], model.visual.attnpool.positional_embedding, h_resolution, w_resolution)

    # 删除state_dict中无关的参数（这些参数不是模型可学习参数，而是记录的配置）
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # 转换权重类型（如把FP32转为FP16，适配混合精度）
    convert_weights(model)

    # 加载预训练权重到模型
    model.load_state_dict(state_dict)
    # 返回评估模式的模型（禁用Dropout/BatchNorm的训练行为）
    return model.eval()

import math
def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
      
    ntok_new = posemb_new.shape[0] #129,2048

    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid))) #14
    print('Position embedding resize to height:{} width: {}'.format(hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2) 
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear') 
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid.squeeze()], dim=0)
    return posemb