import torch
import time
from .nerf_helpers import get_minibatches, ndc_rays
from .nerf_helpers import sample_pdf_2 as sample_pdf
from .nerf_helpers import dump_rays
from .volume_rendering_utils import volume_render_radiance_field


def run_network(network_fn, pts, ray_batch, chunksize, embed_fn, embeddirs_fn, expressions = None, latent_code = None, N_samples = -1, background_prior = None):

    pts_flat = pts.reshape((-1, pts.shape[-1]))
    
    # ray_batch 2048*8
    # pts 2048*64*3
    # pts_flat 131072*3
    embedded = embed_fn(pts_flat)
    # embedded 131072*3
    if embeddirs_fn is not None:
        ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
        viewdirs = ray_batch[..., None, -3:]
        # viewdirs 2048*1*3
        input_dirs = viewdirs.expand(pts.shape)
        # input_dirs 2048*64*3
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        # input_dirs_flat 131072*3
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        # embedded_dirs 131072*24
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)
        # embedded 131072*87

    embedded = torch.cat((ray_batch[..., :3], ray_batch[..., 3:6]), dim=-1)
    batches = get_minibatches(embedded, chunksize=chunksize)
    if expressions is None:
        preds = [network_fn(batch) for batch in batches]
    elif latent_code is not None:
        # batch [2048,87] expressions [76] latent_code [32]
        rgb, disp, weight = network_fn(embedded, expressions, latent_code, N_samples = N_samples, background_prior = background_prior)
    else:
        preds = [network_fn(batch, expressions) for batch in batches]
    # tensorf rgbmap 2048 * 3 depth map 2048
    # nerface
    del embedded, input_dirs_flat
    return rgb, disp, weight


def predict_and_render_radiance(
    ray_batch,
    model_coarse,
    model_fine,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    expressions = None,
    background_prior = None,
    latent_code = None,
    ray_dirs_fake = None
):
    # TESTED
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6].clone() # TODO remove clone ablation rays
    bounds = ray_batch[..., 6:8].view((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]
    N_samples = getattr(options.nerf, mode).num_coarse
    # TODO: Use actual values for "near" and "far" (instead of 0. and 1.)
    # when not enabling "ndc".
    t_vals = torch.linspace(
        0.0,
        1.0,
        getattr(options.nerf, mode).num_coarse,
        dtype=ro.dtype,
        device=ro.device,
    )

    if not getattr(options.nerf, mode).lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)

    z_vals = z_vals.expand([num_rays, getattr(options.nerf, mode).num_coarse])

    if getattr(options.nerf, mode).perturb:
        # Get intervals between samples.
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        # Stratified samples in those intervals.
        t_rand = torch.rand(z_vals.shape, dtype=ro.dtype, device=ro.device)
        z_vals = lower + (upper - lower) * t_rand
    # pts -> (num_rays, N_samples, 3)
    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]
    # Find the min and max of each coordinate
    # aabb = torch.tensor([[-0.221, -0.152, -0.303], [0.2120, 0.107, 0.3590]], device='cuda:0')
    # x_min, _ = torch.min(torch.min(pts[:,:,0], dim=-1).values, 0)
    # x_max, _ = torch.max(torch.max(pts[:,:,0], dim=-1).values, 0)
    # if x_min < -0.222 or x_max > 0.2120:
    #     print("x out of bounds", x_min, x_max)
    # y_min, _= torch.min(torch.min(pts[:,:,1], dim=1).values, 0)
    # y_max, _= torch.max(torch.max(pts[:,:,1], dim=1).values, 0)
    # if y_min < -0.154 or y_max > 0.107:
    #     print("y out of bounds", y_min, y_max)
    # z_min, _= torch.min(torch.min(pts[:,:,2], dim=1).values, 0)
    # z_max, _= torch.max(torch.max(pts[:,:,2], dim=1).values, 0)
    #
    # if z_min < -0.304 or z_max > 0.3590:
    #     print("z out of bounds", z_min, z_max)

    # Uncomment to dump a ply file visualizing camera rays and sampling points
    #dump_rays(ro.detach().cpu().numpy(), pts.detach().cpu().numpy())
    rgb_coarse, disp_coarse, weights  = run_network(
        model_coarse,
        pts,
        ray_batch,
        getattr(options.nerf, mode).chunksize,
        encode_position_fn,
        encode_direction_fn,
        expressions,
        latent_code,
        N_samples,
        background_prior
    )
    # make last RGB values of each ray, the background
    # if background_prior is not None:
    #     radiance_field[:,-1,:3] = background_prior
    # (
    #     rgb_coarse,
    #     disp_coarse,
    #     acc_coarse,
    #     weights,
    #     depth_coarse,
    # ) = volume_render_radiance_field(
    #     radiance_field,
    #     z_vals,
    #     rd,
    #     radiance_field_noise_std=getattr(options.nerf, mode).radiance_field_noise_std,
    #     white_background=getattr(options.nerf, mode).white_background,
    #     background_prior=background_prior
    # )

    rgb_fine, disp_fine, acc_fine = None, None, None

    #return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine, depth_fine #added depth fine
    return rgb_coarse, disp_coarse, None, rgb_fine, disp_fine, acc_fine, weights[:,-1] #changed last return val to fine_weights


def run_one_iter_of_nerf(
    height,
    width,
    focal_length,
    model_coarse,
    model_fine,
    ray_origins,
    ray_directions,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
    expressions = None,
    background_prior=None,
    latent_code = None,
    ray_directions_ablation = None
):
    is_rad = torch.is_tensor(ray_directions_ablation)
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
        None,
        None,
        None,
        ray_directions.shape[:-1], # depth map
    ]
    if model_fine:
        restore_shapes += restore_shapes
        restore_shapes += [ray_directions.shape[:-1]] # to return fine depth map
    if options.dataset.no_ndc is False:
        #print("calling ndc")
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        # print("calling ndc")
        #"caling normal rays (not NDC)"
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
        if is_rad:
            rd_ablations = ray_directions_ablation.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])

    rays = torch.cat((ro, rd, near, far), dim=-1)
    if is_rad:
        rays_ablation = torch.cat((ro, rd_ablations, near, far), dim=-1)
    # if options.nerf.use_viewdirs: # TODO uncomment
    #     rays = torch.cat((rays, viewdirs), dim=-1)
    #
    viewdirs = None  # TODO remove this paragraph
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        if is_rad:
            viewdirs = ray_directions_ablation
            viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
            viewdirs = viewdirs.view((-1, 3))


    if is_rad:
        batches_ablation = get_minibatches(rays_ablation, chunksize=getattr(options.nerf, mode).chunksize)
    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    assert(batches[0].shape == batches[0].shape)
    background_prior = get_minibatches(background_prior, chunksize=getattr(options.nerf, mode).chunksize) if\
        background_prior is not None else background_prior
    #print("predicting")

    if is_rad:
        pred = [
            predict_and_render_radiance(
                batch,
                model_coarse,
                model_fine,
                options,
                mode,
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions = expressions,
                background_prior = background_prior[i] if background_prior is not None else background_prior,
                latent_code = latent_code,
                ray_dirs_fake = batches_ablation
            )
            for i,batch in enumerate(batches)
        ]
    else:
        pred = [
            predict_and_render_radiance(
                batch,
                model_coarse,
                model_fine,
                options,
                mode,
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions = expressions,
                background_prior = background_prior[i] if background_prior is not None else background_prior,
                latent_code = latent_code,
                ray_dirs_fake = None
            )
            for i,batch in enumerate(batches)
        ]
    #print("predicted")
    synthesized_images = list(zip(*pred))
    # print("synthesized_images", synthesized_images)
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    if mode == "validation":

        synthesized_images = [
            image.view(shape) if (image is not None) and (shape is not None) else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]
        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images)

    return tuple(synthesized_images)



def run_one_iter_of_conditional_nerf(
    height,
    width,
    focal_length,
    model_coarse,
    model_fine,
    ray_origins,
    ray_directions,
    expression,
    options,
    mode="train",
    encode_position_fn=None,
    encode_direction_fn=None,
):
    viewdirs = None
    if options.nerf.use_viewdirs:
        # Provide ray directions as input
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.view((-1, 3))
    # Cache shapes now, for later restoration.
    restore_shapes = [
        ray_directions.shape,
        ray_directions.shape[:-1],
        ray_directions.shape[:-1],
    ]
    if model_fine:
        restore_shapes += restore_shapes
        restore_shapes += ray_directions.shape[:-1] # for fine depth map

    if options.dataset.no_ndc is False:
        ro, rd = ndc_rays(height, width, focal_length, 1.0, ray_origins, ray_directions)
        ro = ro.view((-1, 3))
        rd = rd.view((-1, 3))
    else:
        ro = ray_origins.view((-1, 3))
        rd = ray_directions.view((-1, 3))
    near = options.dataset.near * torch.ones_like(rd[..., :1])
    far = options.dataset.far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if options.nerf.use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=getattr(options.nerf, mode).chunksize)
    pred = [
        predict_and_render_radiance(
            batch,
            model_coarse,
            model_fine,
            options,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
        )
        for batch in batches
    ]
    synthesized_images = list(zip(*pred))
    synthesized_images = [
        torch.cat(image, dim=0) if image[0] is not None else (None)
        for image in synthesized_images
    ]
    if mode == "validation":
        synthesized_images = [
            image.view(shape) if image is not None else None
            for (image, shape) in zip(synthesized_images, restore_shapes)
        ]

        # Returns rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine
        # (assuming both the coarse and fine networks are used).
        if model_fine:
            return tuple(synthesized_images)
        else:
            # If the fine network is not used, rgb_fine, disp_fine, acc_fine are
            # set to None.
            return tuple(synthesized_images + [None, None, None])

    return tuple(synthesized_images)



import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=5)
