import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            #print ('flow size 1 ',flow.shape)
            #print ('new_locs size 2 ',new_locs.shape)
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            #print ('new_locs size 3 ',new_locs.shape)
            new_locs = new_locs[..., [2, 1, 0]]  # img: 128*192*128   flow: 128*192*128*3
            #print ('new_locs size 4 ',new_locs.shape)

        #return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        return nnf.grid_sample(src, new_locs, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()
        
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class VecInt_range_flow(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()
        
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer_range_flow(inshape)

    def forward(self, vec,range_flow):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec,range_flow)
        return vec

class SpatialTransformer_range_flow_accumulate_flow(nn.Module):
    """
    N-D Spatial Transformer with Flow Accumulation
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)  # Shape: (len(size), *size)
        grid = torch.unsqueeze(grid, 0)  # Add batch dimension
        grid = grid.type(torch.FloatTensor)  # Convert to FloatTensor

        # Register the grid as a buffer
        self.register_buffer('grid', grid)

    def forward(self, src, flows, range_flow):
        """
        Forward method for flow accumulation and image deformation.

        Args:
            src (torch.Tensor): Source image of shape (B, C, *size).
            flows (list of torch.Tensor): List of flows to accumulate, each of shape (B, len(size), *size).
            range_flow (float): Scaling factor for flow.

        Returns:
            torch.Tensor: Deformed source image.
        """
        # Initialize the accumulated flow with zeros
        accumulated_flow = torch.zeros_like(self.grid)  # Shape: (B, len(size), *size)

        # Loop through each flow and accumulate
        for flow in flows:
            # Normalize the grid for sampling ([-1, 1] range)
            shape = flow.shape[2:]
            normalized_grid = 2 * (self.grid / torch.tensor(shape, device=self.grid.device).view(-1, 1, 1, 1)) - 1

            # Warp the current flow using the accumulated flow
            warped_flow = nn.functional.grid_sample(
                flow, normalized_grid.permute(0, 2, 3, 4, 1), align_corners=True, mode=self.mode, padding_mode="border"
            )

            # Accumulate the warped flow
            accumulated_flow += warped_flow

        # Compute the final grid
        final_grid = self.grid + accumulated_flow * range_flow

        # Normalize the final grid for sampling
        for i in range(len(shape)):
            final_grid[:, i, ...] = 2 * (final_grid[:, i, ...] / (shape[i] - 1) - 0.5)

        # Move channels to the last position for sampling
        final_grid = final_grid.permute(0, 2, 3, 4, 1)

        # Deform the source image using the final accumulated flow
        return nn.functional.grid_sample(src, final_grid, align_corners=True, mode=self.mode)
    
class SpatialTransformer_range_flow(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow,range_flow):
        #print ('info JJ range_flow ',range_flow)
        # new locations
        new_locs = self.grid + flow*range_flow #This is grid def
        shape = flow.shape[2:]
        #print (new_locs)
        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        #return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
        return nnf.grid_sample(src, new_locs, mode=self.mode,align_corners=False,)

class CompositionTransform(nn.Module):
    def __init__(self):
        super(CompositionTransform, self).__init__()

    def forward(self, flow_1, flow_2, sample_grid, range_flow):
        size_tensor = sample_grid.size()
        grid = sample_grid + (flow_2.permute(0,2,3,4,1) * range_flow)
        grid[0, :, :, :, 0] = (grid[0, :, :, :, 0] - ((size_tensor[3] - 1) / 2)) / (size_tensor[3] - 1) * 2
        grid[0, :, :, :, 1] = (grid[0, :, :, :, 1] - ((size_tensor[2] - 1) / 2)) / (size_tensor[2] - 1) * 2
        grid[0, :, :, :, 2] = (grid[0, :, :, :, 2] - ((size_tensor[1] - 1) / 2)) / (size_tensor[1] - 1) * 2
        compos_flow = F.grid_sample(flow_1, grid, mode='bilinear', align_corners=True) + flow_2
        return compos_flow
        
class SpatialTransformer_range_flow_flow1_flow2(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        self.compos_trans=CompositionTransform().cuda()
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow_list,range_flow):
        # new locations
        #print ('info range_flow ',range_flow)
        flow1=flow_list[0]
        flow2=flow_list[1]

        
        shape = flow1.shape[2:]
        grid = self.grid
        new_locs1 = grid + flow1 * range_flow
        for i in range(len(shape)):
            new_locs1[:, i, ...] = 2 * (new_locs1[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs1 = new_locs1.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs1 = new_locs1.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        deform_1_img=nnf.grid_sample(src, new_locs1, mode=self.mode)

        if 2>3:
            #start to compose dvf on 2nd dvf

            
            new_locs2 = grid + flow2* range_flow
            shape = flow2.shape[2:]

            #Normalize it 
            for i in range(len(shape)):
                new_locs2[:, i, ...] = 2 * (new_locs2[:, i, ...] / (shape[i] - 1) - 0.5)


            if len(shape) == 2:
                new_locs2 = new_locs2.permute(0, 2, 3, 1)[..., [1, 0]]
            elif len(shape) == 3:
                new_locs2 = new_locs2.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

            
            compos_flow = F.grid_sample(flow1, new_locs2, mode='bilinear') + flow2
            
            compos_flow=compos_flow.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
            deform_2_img = nnf.grid_sample(src, compos_flow, mode=self.mode)
        if 3>2:
            # 将 flow2 重新采样到被 flow1 变形后的网格上
            flow2_resampled = nnf.grid_sample(flow2*range_flow, new_locs1,mode=self.mode)

            # 累积变形场：flow_total = flow1 + resampled(flow2)
            flow_total = flow1 + flow2_resampled / range_flow

            # Step 2: 使用 flow_total 变形图像
            new_locs_total = grid + flow_total * range_flow

            #new_locs_total = grid + flow2 * range_flow

            for i in range(len(shape)):
                new_locs_total[:, i, ...] = 2 * (new_locs_total[:, i, ...] / (shape[i] - 1) - 0.5)
            if len(shape) == 2:
                new_locs_total = new_locs_total.permute(0, 2, 3, 1)[..., [1, 0]]
            elif len(shape) == 3:
                new_locs_total = new_locs_total.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

            # 应用累积变形场到源图像
            deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode)

        #Flow_1_2 = self.compos_trans(flow1, flow2, grid, range_flow)


        return deform_2_img,deform_2_img
    

class SpatialTransformer_range_flow_flow_list222(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow_list,range_flow):
        # new locations
        #print ('info range_flow ',range_flow)
        flow1=flow_list[0] #Previous 
        flow2=flow_list[1] #Current one 

        
        shape = flow1.shape[2:]
        grid = self.grid
        new_locs1 = grid + flow1 * range_flow
        for i in range(len(shape)):
            new_locs1[:, i, ...] = 2 * (new_locs1[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs1 = new_locs1.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs1 = new_locs1.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        #deform_1_img=nnf.grid_sample(src, new_locs1, mode=self.mode,align_corners=True,)
        deform_1_img=nnf.grid_sample(src, new_locs1, mode=self.mode)

        
        if 3>2:
            B, C, *dims = flow1.size()
            print ('flow1 size ',flow1.size())
            grid1 = torch.stack(torch.meshgrid(
                [torch.linspace(-1, 1, size, device=flow1.device) for size in reversed(dims)]
            ), dim=-1)  # (D, H, W, len(dims)) 或 (H, W, 2)
            
            #grid1 = grid1.expand(B, *grid.shape)
            grid1=grid1[None,:]
            print (grid1.size()) #[1, 128, 192, 128, 3])
            grid1=grid1.permute(0,4,2,3,1) #1,3,192,128,128
            grid1=grid1.permute(0,1,4,3,2)
            grid1=grid1.permute(0,1,2,4,3)
            #grid1=grid1.permute(0,1,3,2)
            print(grid1.size())
            # 应用 flow2 更新网格
            
            grid2 = grid + flow2* range_flow
            print ('grid2 size ',grid2.size())
            grid2=grid2.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
            #grid2=grid2.permute(0,1,4,3,2)
            #grid2=grid2.permute(0,1,2,4,3)

            print ('grid2 size ',grid2.size())
            # 使用 grid_sample 将 flow1 应用到更新后的网格 grid2 上
            #flow1_warped = F.grid_sample(flow1, grid2.permute(0, 2, 3, 4, 1), mode='bilinear', padding_mode='border', align_corners=True)
            #flow1_warped = F.grid_sample(flow1, grid2.permute(0, 2, 3, 4, 1), mode='bilinear', align_corners=True)
            #flow1_warped = F.grid_sample(flow1, grid2, mode='bilinear', align_corners=True,padding_mode='border')
            flow1_warped = F.grid_sample(flow1, grid2, mode='bilinear')

            # 合成最终的流场
            #flow1_warped=flow1* range_flow
            warped_flow=flow1_warped+flow2 

            new_locs_total = grid+ warped_flow* range_flow
            #new_locs_total = grid +flow2* range_flow

            out_flow=flow1_warped+flow2
            for i in range(len(shape)):
                new_locs_total[:, i, ...] = 2 * (new_locs_total[:, i, ...] / (shape[i] - 1) - 0.5)
            if len(shape) == 2:
                new_locs_total = new_locs_total.permute(0, 2, 3, 1)[..., [1, 0]]
            elif len(shape) == 3:
                new_locs_total = new_locs_total.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

            # 应用累积变形场到源图像
            #deform_2_img = nnf.grid_sample(deform_1_img, new_locs_total, mode=self.mode,align_corners=True,)
            #deform_2_img = nnf.grid_sample(deform_1_img, new_locs_total, mode=self.mode)
            deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode)
            #deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode)

        #Flow_1_2 = self.compos_trans(flow1, flow2, grid, range_flow)


        return deform_2_img,out_flow

#Accumulate all flow list >2
class SpatialTransformer_range_flow_flow_list2_1(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow_list,final_flow,range_flow):
        # new locations
        #print ('info range_flow ',range_flow)
        flow1=flow_list[0] #Previous 
        flow2=final_flow#flow_list[1] #Current one 

        flow_list_=flow_list.copy()
        flow_list_.append(final_flow)

        composed_dvf = flow_list_[0]  # Initialize with the first dvf
        shape = composed_dvf.shape[2:]

        if 1>2:
            grid2 = grid + flow1* range_flow
            
            grid2=grid2.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
            
            flow1_warped = F.grid_sample(flow2, grid2, mode='bilinear')

            # 合成最终的流场
            #flow1_warped=flow1* range_flow
            warped_flow=flow1_warped+flow1 

        for dvf in flow_list_[1:]:
            
            grid = self.grid
            
            

            grid2 = grid + composed_dvf* range_flow

            grid2=grid2.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

            flow1_warped = F.grid_sample(dvf, grid2, mode='bilinear')
            # 合成最终的流场
            #flow1_warped=flow1* range_flow
            composed_dvf=flow1_warped+composed_dvf 

            
        
        

        

        new_locs_total = grid+ composed_dvf* range_flow


        out_flow=composed_dvf
        for i in range(len(shape)):
            new_locs_total[:, i, ...] = 2 * (new_locs_total[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs_total = new_locs_total.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs_total = new_locs_total.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode)


        return deform_2_img,out_flow



class SpatialTransformer_range_flow_flow_list2(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow_list,range_flow):
        # new locations
        #print ('info range_flow ',range_flow)
        flow1=flow_list[0] #Previous 
        flow2=flow_list[1] #Current one 

        
        shape = flow1.shape[2:]
        grid = self.grid
        new_locs1 = grid + flow1 * range_flow
        for i in range(len(shape)):
            new_locs1[:, i, ...] = 2 * (new_locs1[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs1 = new_locs1.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs1 = new_locs1.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        #deform_1_img=nnf.grid_sample(src, new_locs1, mode=self.mode,align_corners=True,)
        deform_1_img=nnf.grid_sample(src, new_locs1, mode=self.mode)

        
        if 3>2:
            ## Apply the first dvf to the grid
            #deformed_grid = grid + dvf1

            ## Deform the second dvf with the deformed grid
            #composed_dvf = F.grid_sample(dvf2, deformed_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True).permute(0, 3, 1, 2)

            # Return the accumulated dvf
            #dvf1 + composed_dvf

            # 2. Apply the first flow
            #warped_grid1 = grid + flow1

            # 3. Apply the second flow on the intermediate warped grid
            #warped_grid2 = warped_grid1 + F.grid_sample(flow2, (warped_grid1/torch.tensor([W-1,H-1], device=image.device)*2-1).permute(0,2,3,1), mode='bilinear', padding_mode='zeros', align_corners=True).permute(0, 3, 1, 2)

            # 4. Warp the original image using the combined flow (warped_grid2)
            #warped_image = F.grid_sample(image, (warped_grid2 / torch.tensor([W-1,H-1], device=image.device) * 2 - 1).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
            
            grid2 = grid + flow2* range_flow

            #Not sure do I need to do this 
            for i in range(len(shape)):
                grid2[:, i, ...] = 2 * (grid2[:, i, ...] / (shape[i] - 1) - 0.5)
            if len(shape) == 2:
                grid2 = grid2.permute(0, 2, 3, 1)[..., [1, 0]]
            elif len(shape) == 3:
                grid2 = grid2.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

            print ('grid2 size ',grid2.size())
            # 使用 grid_sample 将 flow1 应用到更新后的网格 grid2 上
            #flow1_warped = F.grid_sample(flow1, grid2.permute(0, 2, 3, 4, 1), mode='bilinear', padding_mode='border', align_corners=True)
            #flow1_warped = F.grid_sample(flow1, grid2.permute(0, 2, 3, 4, 1), mode='bilinear', align_corners=True)
            #flow1_warped = F.grid_sample(flow1, grid2, mode='bilinear', align_corners=True,padding_mode='border')
            flow1_warped = F.grid_sample(flow1, grid2, mode='bilinear')

            # 合成最终的流场
            #flow1_warped=flow1* range_flow
            #warped_flow=flow1_warped+flow1 
            warped_flow=flow2 + flow1_warped# flow2 

            new_locs_total = grid+ warped_flow* range_flow
            #new_locs_total = grid +flow2* range_flow

            out_flow=flow1_warped+flow2
            for i in range(len(shape)):
                new_locs_total[:, i, ...] = 2 * (new_locs_total[:, i, ...] / (shape[i] - 1) - 0.5)
            if len(shape) == 2:
                new_locs_total = new_locs_total.permute(0, 2, 3, 1)[..., [1, 0]]
            elif len(shape) == 3:
                new_locs_total = new_locs_total.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

            # 应用累积变形场到源图像
            #deform_2_img = nnf.grid_sample(deform_1_img, new_locs_total, mode=self.mode,align_corners=True,)
            #deform_2_img = nnf.grid_sample(deform_1_img, new_locs_total, mode=self.mode)
            deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode,align_corners=True,)
            #deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode)

        #Flow_1_2 = self.compos_trans(flow1, flow2, grid, range_flow)


        return deform_2_img,out_flow


class SpatialTransformer_range_flow_flow_list2_Multi_Flow(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow_list,final_flow,range_flow):
        # new locations
        #print ('info range_flow ',range_flow)
        

        flow1=flow_list[0] #Previous 
        flow2=final_flow#flow_list[1] #Current one 

        flow_list_=flow_list.copy()
        flow_list_.append(final_flow)

        composed_dvf = flow_list_[0]  # Initialize with the first dvf
        shape = composed_dvf.shape[2:]

        

        for dvf in flow_list_[1:]:
            
            grid = self.grid
            grid2 = grid + dvf* range_flow
            for i in range(len(shape)):
                grid2[:, i, ...] = 2 * (grid2[:, i, ...] / (shape[i] - 1) - 0.5)
            if len(shape) == 2:
                grid2 = grid2.permute(0, 2, 3, 1)[..., [1, 0]]
            elif len(shape) == 3:
                grid2 = grid2.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

            flow1_warped = F.grid_sample(composed_dvf, grid2, mode='bilinear')
            # 
            #flow1_warped=flow1* range_flow
            composed_dvf=flow1_warped+dvf 
        
        

        new_locs_total = grid+ composed_dvf* range_flow
        #new_locs_total = grid +flow2* range_flow

        out_flow=flow1_warped+flow2
        for i in range(len(shape)):
            new_locs_total[:, i, ...] = 2 * (new_locs_total[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs_total = new_locs_total.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs_total = new_locs_total.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        # 应用累积变形场到源图像
        #deform_2_img = nnf.grid_sample(deform_1_img, new_locs_total, mode=self.mode,align_corners=True,)
        #deform_2_img = nnf.grid_sample(deform_1_img, new_locs_total, mode=self.mode)
        deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode,align_corners=False ,)
        #deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode)

        #Flow_1_2 = self.compos_trans(flow1, flow2, grid, range_flow)
        return deform_2_img,out_flow
        

class SpatialTransformer_range_flow_flow_list2_Multi_Flow_Jue_Copy_12202024(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow_list,final_flow,range_flow):
        # new locations
        #print ('info range_flow ',range_flow)
        

        flow1=flow_list[0] #Previous 
        flow2=final_flow#flow_list[1] #Current one 

        flow_list_=flow_list.copy()
        flow_list_.append(final_flow)

        composed_dvf = flow_list_[0]  # Initialize with the first dvf
        shape = composed_dvf.shape[2:]

        

        for dvf in flow_list_[1:]:
            
            grid = self.grid
            grid2 = grid + dvf* range_flow
            for i in range(len(shape)):
                grid2[:, i, ...] = 2 * (grid2[:, i, ...] / (shape[i] - 1) - 0.5)
            if len(shape) == 2:
                grid2 = grid2.permute(0, 2, 3, 1)[..., [1, 0]]
            elif len(shape) == 3:
                grid2 = grid2.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

            flow1_warped = F.grid_sample(composed_dvf, grid2, mode='bilinear')
            # 
            #flow1_warped=flow1* range_flow
            composed_dvf=flow1_warped+dvf 
        
        

        new_locs_total = grid+ composed_dvf* range_flow
        #new_locs_total = grid +flow2* range_flow

        out_flow=flow1_warped+flow2
        for i in range(len(shape)):
            new_locs_total[:, i, ...] = 2 * (new_locs_total[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs_total = new_locs_total.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs_total = new_locs_total.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        # 应用累积变形场到源图像
        #deform_2_img = nnf.grid_sample(deform_1_img, new_locs_total, mode=self.mode,align_corners=True,)
        #deform_2_img = nnf.grid_sample(deform_1_img, new_locs_total, mode=self.mode)
        deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode,align_corners=False ,)
        #deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode)

        #Flow_1_2 = self.compos_trans(flow1, flow2, grid, range_flow)
        return deform_2_img,out_flow
    
class SpatialTransformer_range_flow_flow_list22(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        self.compos_trans=CompositionTransform().cuda()
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow_list,range_flow):
        # new locations
        #print ('info range_flow ',range_flow)
        flow1=flow_list[0] #Previous 
        flow2=flow_list[1] #Current one 

        
        shape = flow1.shape[2:]
        grid = self.grid
        new_locs1 = grid + flow1 * range_flow
        for i in range(len(shape)):
            new_locs1[:, i, ...] = 2 * (new_locs1[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs1 = new_locs1.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs1 = new_locs1.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        #deform_1_img=nnf.grid_sample(src, new_locs1, mode=self.mode,align_corners=True,)
        deform_1_img=nnf.grid_sample(src, new_locs1, mode=self.mode)

        
        if 3>2:

            
            # 应用 flow2 更新网格
            grid2 = grid + flow2* range_flow

            print ('grid2 size ',grid2.size())
            # 使用 grid_sample 将 flow1 应用到更新后的网格 grid2 上
            #flow1_warped = F.grid_sample(flow1, grid2.permute(0, 2, 3, 4, 1), mode='bilinear', padding_mode='border', align_corners=True)
            #flow1_warped = F.grid_sample(flow1, grid2.permute(0, 2, 3, 4, 1), mode='bilinear', align_corners=True)
            flow1_warped = F.grid_sample(flow1, grid2.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]], mode='bilinear')

            # 合成最终的流场
            #flow1_warped=flow1* range_flow

            #new_locs_total = grid+ flow1_warped + flow2* range_flow

            warped_flow=flow1_warped+flow2 

            new_locs_total = grid+ warped_flow* range_flow

            #new_locs_total = grid +flow2* range_flow

            out_flow=flow1_warped+flow2

            for i in range(len(shape)):
                new_locs_total[:, i, ...] = 2 * (new_locs_total[:, i, ...] / (shape[i] - 1) - 0.5)
            if len(shape) == 2:
                new_locs_total = new_locs_total.permute(0, 2, 3, 1)[..., [1, 0]]
            elif len(shape) == 3:
                new_locs_total = new_locs_total.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

            # 应用累积变形场到源图像
            #deform_2_img = nnf.grid_sample(deform_1_img, new_locs_total, mode=self.mode,align_corners=True,)
            #deform_2_img = nnf.grid_sample(deform_1_img, new_locs_total, mode=self.mode)
            deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode)
            #deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode)

        #Flow_1_2 = self.compos_trans(flow1, flow2, grid, range_flow)


        return deform_2_img,out_flow
    
        
class SpatialTransformer_range_flow_flow_list(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode
        self.compos_trans=CompositionTransform().cuda()
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow_list,range_flow):
        # new locations
        #print ('info range_flow ',range_flow)
        flow1=flow_list[0] #Previous 
        flow2=flow_list[1] #Current one 

        
        shape = flow1.shape[2:]
        grid = self.grid
        new_locs1 = grid + flow1 * range_flow
        for i in range(len(shape)):
            new_locs1[:, i, ...] = 2 * (new_locs1[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs1 = new_locs1.permute(0, 2, 3, 1)[..., [1, 0]]
        elif len(shape) == 3:
            new_locs1 = new_locs1.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

        #deform_1_img=nnf.grid_sample(src, new_locs1, mode=self.mode,align_corners=True,)
        deform_1_img=nnf.grid_sample(src, new_locs1, mode=self.mode)

        
        if 3>2:
            # 将 flow2 重新采样到被 flow1 变形后的网格上
            flow2_resampled = nnf.grid_sample(flow2, new_locs1,mode=self.mode)

            # 累积变形场：flow_total = flow1 + resampled(flow2)
            flow_total = flow1 + flow2_resampled #/ range_flow

            # Step 2: 使用 flow_total 变形图像
            new_locs_total = grid + flow_total * range_flow

            #new_locs_total = grid + flow2 * range_flow

            for i in range(len(shape)):
                new_locs_total[:, i, ...] = 2 * (new_locs_total[:, i, ...] / (shape[i] - 1) - 0.5)
            if len(shape) == 2:
                new_locs_total = new_locs_total.permute(0, 2, 3, 1)[..., [1, 0]]
            elif len(shape) == 3:
                new_locs_total = new_locs_total.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]

            # 应用累积变形场到源图像
            #deform_2_img = nnf.grid_sample(deform_1_img, new_locs_total, mode=self.mode,align_corners=True,)
            #deform_2_img = nnf.grid_sample(deform_1_img, new_locs_total, mode=self.mode)
            #deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode)
            deform_2_img = nnf.grid_sample(src, new_locs_total, mode=self.mode)

        #Flow_1_2 = self.compos_trans(flow1, flow2, grid, range_flow)


        return deform_2_img,flow_total
    
class SpatialTransformer_range_flow_mask(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow,range_flow,mask):
        # new locations

        range_flow=range_flow*(1-mask)
        new_locs = self.grid + flow*range_flow
        #tep_grid=self.grid*mask
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
            #tep_grid = tep_grid.permute(0, 2, 3, 4, 1)
            #tep_grid = tep_grid[..., [2, 1, 0]]

        new_locs1=new_locs.clone()
        #new_locs1[tep_grid==0]=0
        new_locs=new_locs#+new_locs1#tep_grid
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
