import torch
import torch.nn as nn

class NerfEmbedding(nn.Module):
    """NeRF-style positional embedding"""
    def __init__(self, input_dims, num_freqs, log_space=True):
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.output_dims = input_dims * (2 * num_freqs)
        
        if log_space:
            freq_bands = 2.**torch.linspace(0., num_freqs-1, num_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(num_freqs-1), num_freqs)
        
        self.register_buffer('freq_bands', freq_bands)
    
    def forward(self, x):
        out = []
        for freq in self.freq_bands:
            for func in [torch.sin, torch.cos]:
                out.append(func(x * freq))
        return torch.cat(out, dim=-1)


class CameraPoseEmbedding(nn.Module):
    """
    相机位姿嵌入模块
    使用 NeRF 风格的频率编码器对相机的平移、旋转、内参进行编码
    """
    def __init__(self, embed_dim=1024, pos_freqs=10, rot_freqs=4, intri_freqs=4, hidden_dims=[128, 256, 512]):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 位姿编码器
        self.trans_embedding = NerfEmbedding(3, pos_freqs)   # translation 3D
        self.rot_embedding = NerfEmbedding(4, rot_freqs)   # quaternion 4D
        self.intri_embedding = NerfEmbedding(2, intri_freqs) # field of view (2D)
        
        camera_dim = self.trans_embedding.output_dims + self.rot_embedding.output_dims + self.intri_embedding.output_dims
        
        self.camera_mlp = nn.Sequential(
            nn.Linear(camera_dim, hidden_dims[0]),
            nn.GELU(),
        )
        
        for i in range(len(hidden_dims)-1):
            self.camera_mlp = nn.Sequential(
                self.camera_mlp,
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.GELU(),
            )
            
        self.camera_mlp = nn.Sequential(
            self.camera_mlp,
            nn.Linear(hidden_dims[-1], embed_dim),
        )
        
        self.norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)
    
    def forward(self, img_shape, pose_encoding):
        """
        Args:
            img_shape: (B, S)，B=batch size, S=sequence length
            pose_encoding: (B, S, 9)，相机位姿编码，9维分别为：
                - [:3] = absolute translation vector T (3D)
                - [3:7] = rotation as quaternion quat (4D)
                - [7:] = field of view (2D)
        
        Returns:
            camera_tokens: [B*S, 1, embed_dim]
        """
        B, S = img_shape
        C = self.embed_dim

        # 编码
        trans_encoded = self.trans_embedding(pose_encoding[:, :, :3])  # [B, S, pos_dim]
        rot_encoded = self.rot_embedding(pose_encoding[:, :, 3:7])  # [B, S, rot_dim]
        intri_encoded = self.intri_embedding(pose_encoding[:, :, 7:])  # [B, S, intri_dim]
        camera_features = torch.cat([trans_encoded, rot_encoded, intri_encoded], dim=-1)
        camera_tokens = self.camera_mlp(camera_features)  # [B, S, C]
        camera_tokens = self.norm_layer(camera_tokens)

        return camera_tokens.view(B * S, 1, C)
    


class CameraPoseEmbedding3dof(nn.Module):
    """
    相机位姿嵌入模块
    使用 NeRF 风格的频率编码器对相机的平移、旋转进行编码
    """
    def __init__(self, embed_dim=1024, pos_freqs=10, rot_freqs=10, hidden_dims=[128, 256, 512]):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 位姿编码器
        self.trans_embedding = NerfEmbedding(2, pos_freqs)   # translation 2D
        self.rot_embedding = NerfEmbedding(1, rot_freqs)   # northdeg 1D
        # self.intri_embedding = NerfEmbedding(2, intri_freqs) # field of view (2D)

        camera_dim = self.trans_embedding.output_dims + self.rot_embedding.output_dims # + self.intri_embedding.output_dims

        self.camera_mlp = nn.Sequential(
            nn.Linear(camera_dim, hidden_dims[0]),
            nn.GELU(),
        )
        
        for i in range(len(hidden_dims)-1):
            self.camera_mlp = nn.Sequential(
                self.camera_mlp,
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.GELU(),
            )
            
        self.camera_mlp = nn.Sequential(
            self.camera_mlp,
            nn.Linear(hidden_dims[-1], embed_dim),
        )
        
        self.norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)
    
    def forward(self, pose_encoding, img_shape):
        """
        Args:
            img_shape: (B, S), B=batch size, S=sequence length
            pose_encoding: (B, S, 3), 相机位姿编码, 3维分别为:
                - [:2] = relative translation vector T (2D)
                - [2:3] = rotation as northdeg (1D)

        Returns:
            camera_tokens: [B*S, 1, embed_dim]
        """
        B, S = img_shape
        C = self.embed_dim

        # 编码
        trans_encoded = self.trans_embedding(pose_encoding[:, :, :2])  # [B, S, pos_dim]
        rot_encoded = self.rot_embedding(pose_encoding[:, :, 2:3])  # [B, S, rot_dim]
        # intri_encoded = self.intri_embedding(pose_encoding[:, :, 7:])  # [B, S, intri_dim]
        # camera_features = torch.cat([trans_encoded, rot_encoded, intri_encoded], dim=-1)
        camera_features = torch.cat([trans_encoded, rot_encoded], dim=-1)
        camera_tokens = self.camera_mlp(camera_features)  # [B, S, C]
        camera_tokens = self.norm_layer(camera_tokens)

        return camera_tokens.view(B * S, 1, C)
    


class CameraPoseEmbeddingYaw(nn.Module):
    """
    相机位姿嵌入模块
    使用 NeRF 风格的频率编码器对相机的旋转进行编码
    """
    def __init__(self, embed_dim=1024, rot_freqs=10, hidden_dims=[128, 256, 512]):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 位姿编码器
        self.rot_embedding = NerfEmbedding(1, rot_freqs)   # northdeg 1D
        # self.intri_embedding = NerfEmbedding(2, intri_freqs) # field of view (2D)

        camera_dim = self.rot_embedding.output_dims # + self.intri_embedding.output_dims

        self.camera_mlp = nn.Sequential(
            nn.Linear(camera_dim, hidden_dims[0]),
            nn.GELU(),
        )
        
        for i in range(len(hidden_dims)-1):
            self.camera_mlp = nn.Sequential(
                self.camera_mlp,
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.GELU(),
            )
            
        self.camera_mlp = nn.Sequential(
            self.camera_mlp,
            nn.Linear(hidden_dims[-1], embed_dim),
        )
        
        self.norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)
    
    def forward(self, pose_encoding, img_shape):
        """
        Args:
            img_shape: (B, S), B=batch size, S=sequence length
            pose_encoding: (B, S), 相机位姿Yaw编码, 3维分别为:

        Returns:
            camera_tokens: [B*S, 1, embed_dim]
        """
        B, S = img_shape
        C = self.embed_dim

        # 编码
        # trans_encoded = self.trans_embedding(pose_encoding[:, :, :2])  # [B, S, pos_dim]
        rot_encoded = self.rot_embedding(pose_encoding.unsqueeze(-1))  # [B, S, rot_dim]
        # intri_encoded = self.intri_embedding(pose_encoding[:, :, 7:])  # [B, S, intri_dim]
        # camera_features = torch.cat([trans_encoded, rot_encoded, intri_encoded], dim=-1)
        # camera_features = torch.cat([trans_encoded, rot_encoded], dim=-1)
        camera_tokens = self.camera_mlp(rot_encoded)  # [B, S, C]
        camera_tokens = self.norm_layer(camera_tokens)

        return camera_tokens.view(B * S, 1, C)