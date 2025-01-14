import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width
        
        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))


    def compute_projection(
        self,
        means3D: torch.Tensor,          # (N, 3)
        covs3d: torch.Tensor,           # (N, 3, 3)
        K: torch.Tensor,                # (3, 3)
        R: torch.Tensor,                # (3, 3)
        t: torch.Tensor                 # (3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N = means3D.shape[0]
        
        # 1. Transform points to camera space
        cam_points = means3D @ R.T + t.unsqueeze(0) # (N, 3)
        
        # 2. Get depths before projection for proper sorting and clipping
        depths = cam_points[:, 2].clamp(min=1.)  # (N, )
        
        # 3. Project to screen space using camera intrinsics
        screen_points = cam_points @ K.T  # (N, 3)
        means2D = screen_points[..., :2] / screen_points[..., 2:3]  # (N, 2)

        # 4. Transform covariance to camera space and then to 2D
        # Compute Jacobian of perspective projection
        J_proj = torch.zeros((N, 3, 3), device=means3D.device)
        ### FILL:
        ### J_proj = ...
        # Extract 3D coordinates from means3D
        x = cam_points[..., 0]  # (N,)
        y = cam_points[..., 1]  # (N,)
        z = cam_points[..., 2]  # (N,)

        # Camera intrinsic matrix components
        K1 = K[0, 0]  # f_x
        K2 = K[1, 1]  # f_y

        # Compute the Jacobian J_proj
        J_proj[..., 0, 0] = K1 / z  # ∂u/∂x
        J_proj[..., 0, 2] = -K1 * x / (z ** 2)  # ∂u/∂z
        J_proj[..., 1, 1] = K2 / z  # ∂v/∂y
        J_proj[..., 1, 2] = -K2 * y / (z ** 2)  # ∂v/∂z
        
        # Transform covariance to camera space
        ### FILL: Aplly world to camera rotation to the 3d covariance matrix
        ### covs_cam = ...  # (N, 3, 3)
        #breakpoint()
        R = R.unsqueeze(0)
        
        T = torch.matmul(R, J_proj)
        sigma = torch.matmul(T.permute(0, 2, 1), torch.matmul(covs3d, T))  # (N, 3, 3)
        covs2D = sigma[:, :2, :2]  # (N, 2, 2)
        covs2D[:,0,0] += 0.3
        covs2D[:,1,1] += 0.3
        return means2D, covs2D, depths

    def compute_gaussian_values(
        self,
        means2D: torch.Tensor,    # (N, 2)
        covs2D: torch.Tensor,     # (N, 2, 2)
        pixels: torch.Tensor      # (H, W, 2)
    ) -> torch.Tensor:           # (N, H, W)
        N = means2D.shape[0]
        H, W = pixels.shape[:2]
        
        # Compute offset from mean (N, H, W, 2)
        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2) 

        #breakpoint()
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0) # (N, 2, 2)
        gaussian = torch.zeros((N, H, W), device=means2D.device)
        
        dx_reshape = dx.reshape(N, H * W, 2)  # (N, H*W, 2)
        cov_inv = torch.inverse(covs2D)  # (N, 2, 2)
        factor = 1.0 / (2 * np.pi * torch.det(covs2D))  # (N,)
        quadratic_term = torch.sum(dx_reshape * torch.matmul(cov_inv.unsqueeze(1), dx_reshape.unsqueeze(-1)).squeeze(-1), dim=-1)  # (N, H*W)
        ga = factor.unsqueeze(1) * torch.exp(-0.5 * quadratic_term)  # (N, H*W)
        gaussian = ga.reshape(N, H, W)
        # # Compute determinant for normalization
        # ### FILL: compute the gaussian values
        # ### gaussian = ... ## (N, H, W)
        # cov_inv = torch.inverse(covs2D)  # (N, 2, 2)
        # for i in range(N):
        #     cov = covs2D[i]
        #     factor = 1.0 / (2 * np.pi * torch.det(cov))
        #     for j in range(H):
        #         for k in range(W):
        #             x = dx[i, j, k, :]
        #             weighted_dx = torch.dot(x, torch.matmul(cov, x))
        #             gaussian[i, j, k] = factor * torch.exp(-0.5 * weighted_dx)
        #breakpoint()
        return gaussian

    def forward(
            self,
            means3D: torch.Tensor,          # (N, 3)
            covs3d: torch.Tensor,           # (N, 3, 3)
            colors: torch.Tensor,           # (N, 3)
            opacities: torch.Tensor,        # (N, 1)
            K: torch.Tensor,                # (3, 3)
            R: torch.Tensor,                # (3, 3)
            t: torch.Tensor                 # (3, 1)
    ) -> torch.Tensor:
        N = means3D.shape[0]
        
        # 1. Project to 2D, means2D: (N, 2), covs2D: (N, 2, 2), depths: (N,)
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        
        # 2. Depth mask
        valid_mask = (depths > 1.) & (depths < 50.0)  # (N,)
        
        # 3. Sort by depth
        indices = torch.argsort(depths, dim=0, descending=False)  # (N, )
        means2D = means2D[indices]      # (N, 2)
        covs2D = covs2D[indices]       # (N, 2, 2)
        colors = colors[ indices]      # (N, 3)
        opacities = opacities[indices] # (N, 1)
        valid_mask = valid_mask[indices] # (N,)
        #breakpoint()
        # 4. Compute gaussian values
        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)  # (N, H, W)
        
        # 5. Apply valid mask
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)  # (N, H, W)
        
        # 6. Alpha composition setup
        alphas = opacities.view(N, 1, 1) * gaussian_values  # (N, H, W)
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W)  # (N, 3, H, W)
        colors = colors.permute(0, 2, 3, 1)  # (N, H, W, 3)
        
        # 7. Compute weights
        ### FILL:
        ### weights = ... # (N, H, W)
        #breakpoint()
        weights = torch.cumprod(1 - alphas + 1e-8, dim=0)  # (N, H, W)
        weights = weights[:-1,:,:]
        ones = torch.ones(1, self.H, self.W, device=weights.device)
        weights = torch.cat((ones, weights), dim=0)
        weights = alphas * weights
        # 8. Final rendering
        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)  # (H, W, 3)
        
        return rendered
