import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_out = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim, bias=False),
                            nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, _ = x.shape
        q = self.proj_q(x).view(B, N, self.heads, self.head_dim).permute([0, 2, 1, 3])
        k = self.proj_k(x).view(B, N, self.heads, self.head_dim).permute([0, 2, 1, 3])
        v = self.proj_v(x).view(B, N, self.heads, self.head_dim).permute([0, 2, 1, 3])
        product = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(product, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)

        # combine heads
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B, N, self.embed_dim)
        return self.proj_out(out)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.proj_out = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim, bias=False),
                            nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, xq, xk, xv):
        B, N, _ = xq.shape
        _, M, _ = xk.shape
        q = self.proj_q(xq).view(B, N, self.heads, self.head_dim).permute([0, 2, 1, 3])
        k = self.proj_k(xk).view(B, M, self.heads, self.head_dim).permute([0, 2, 1, 3])
        v = self.proj_v(xv).view(B, M, self.heads, self.head_dim).permute([0, 2, 1, 3])
        product = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = F.softmax(product, dim=-1)
        # weights = self.dropout(weights)
        out = torch.matmul(weights, v)

        # combine heads
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B, N, self.embed_dim)
        return self.proj_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SelfEncoderLayer(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64, num_heads=4):
        super().__init__()
        self.Attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.FeedForward = FeedForward(embed_dim, hidden_dim)
        self.Identity = nn.Identity()

    def forward(self, x):
        residual = self.Identity(x)
        a = residual + self.Attention(self.norm1(x))
        residual = self.Identity(a)
        a = residual + self.FeedForward(self.norm2(a))
        return a


class CrossEncoderLayer(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=64, num_heads=4):
        super().__init__()
        self.Attention = MultiHeadCrossAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.FeedForward = FeedForward(embed_dim, hidden_dim)
        self.Identity = nn.Identity()

    def forward(self, xq, xk, xv):
        residual = self.Identity(xq)
        a = residual + self.Attention(self.norm1(xq), self.norm1(xk), self.norm1(xv))
        residual = self.Identity(a)
        a = residual + self.FeedForward(self.norm2(a))
        return a  

  
class PositionEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=64):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        d_model = embed_dim if embed_dim%2==0 else embed_dim+1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0, dtype=torch.float)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :x.size(2)]
        return self.dropout(x)


class Embedding(nn.Module):
    def __init__(self, in_channels: int, time_step: int, kernLenght: int = 64, F1: int = 8, D: int = 2, F2: int = 21, dropout_size = 0.5):
        super().__init__()
        self.Chans = in_channels
        self.Samples = time_step
        self.kernLenght = kernLenght
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropoutRate = dropout_size
        self.block1 = nn.Sequential(
            nn.ZeroPad2d((self.kernLenght // 2 - 1,
                          self.kernLenght - self.kernLenght // 2, 0,
                          0)),  # left, right, up, bottom
            nn.Conv2d(in_channels=1,
                      out_channels=self.F1,
                      kernel_size=(1, self.kernLenght),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1),
            # DepthwiseConv2d
            nn.Conv2d(in_channels=self.F1,
                      out_channels=self.F1 * self.D,
                      kernel_size=(self.Chans, 1),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(p=self.dropoutRate))

        self.block2 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F1 * self.D,
                      kernel_size=(1, 16),
                      stride=1,
                      groups=self.F1 * self.D,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F1 * self.D),
            nn.Conv2d(in_channels=self.F1 * self.D,
                      out_channels=self.F2,
                      kernel_size=(1, 1),
                      stride=1,
                      bias=False),
            nn.BatchNorm2d(num_features=self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.dropoutRate))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block1(x)
        output = self.block2(output)
        output = output.squeeze(2).permute([0, 2, 1])
        return output
    
    def MaxNormConstraint(self):
        for n, p in self.block1.named_parameters():
            if n == '3.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=1.0)
        for n, p in self.classifier_block.named_parameters():
            if n == '0.weight':
                p.data = torch.renorm(p.data, p=2, dim=0, maxnorm=0.25)
                

class MulT(nn.Module):
    def __init__(self, CH, DL, L=1, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = 21
        self.layer = L
        self.Embedding = Embedding(in_channels=CH, time_step=DL)
        self.PE = PositionEncoding(64)
        self.cls_s = nn.Parameter(torch.rand(1, 1, self.embed_dim))

        self.blocks = nn.ModuleList([
            SelfEncoderLayer(embed_dim=self.embed_dim, hidden_dim=self.embed_dim, num_heads=3)
            for i in range(self.layer)])

        self.logits_eeg = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, x, eye):
        embed = self.Embedding(x) # [B, 28, 21]
        cls_token = self.cls_s.expand(x.shape[0], -1, -1) 
        EEG_embed = torch.cat([cls_token, embed, eye.unsqueeze(1)], dim=1)
        EEG_embed = self.PE(EEG_embed)

        for i, blk in enumerate(self.blocks):
            EEG_embed = blk(EEG_embed)

        output = self.logits_eeg(EEG_embed[:, 0, :])
        return output



class MulTCross(nn.Module):
    def __init__(self, CH, DL, L=1, num_classes=3):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = 21
        self.layer = L
        self.Embedding = Embedding(in_channels=CH, time_step=DL)
        self.PE = PositionEncoding(64)
        self.cls_s = nn.Parameter(torch.rand(1, 1, self.embed_dim))

        self.SelfEncoderLayer = SelfEncoderLayer(embed_dim=self.embed_dim, hidden_dim=self.embed_dim, num_heads=3)
        self.CrossEncoderLayer = CrossEncoderLayer(embed_dim=self.embed_dim, hidden_dim=self.embed_dim, num_heads=3)

        self.logits_eeg = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )
        self.logits_eye = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, x, eye):
        eeg_embed = self.Embedding(x) # [B, 28, 21]
        eye_embed = eye.unsqueeze(1)
        
        cls_token = self.cls_s.expand(x.shape[0], -1, -1)
        sel_eeg_embed = torch.cat([cls_token, eeg_embed, eye_embed], dim=1)
        sel_eeg_embed = self.PE(sel_eeg_embed)

        sel_eeg_embed = self.SelfEncoderLayer(sel_eeg_embed)
        
        eye_embed = sel_eeg_embed[:, -1, :].unsqueeze(1)
        aug_eye_embed = self.CrossEncoderLayer(eye_embed, eeg_embed, eeg_embed)

        output_eeg = self.logits_eeg(sel_eeg_embed[:, 0, :])
        output_eye = self.logits_eye(aug_eye_embed[:, 0, :])
        output = (output_eeg + output_eye)/2.0
        return output

    

if __name__ == "__name__":
    from torchsummary import summary
    net = MulT(19, 900).cuda()
    print(summary(net, [(1, 19, 900), (21, )], batch_size=1))