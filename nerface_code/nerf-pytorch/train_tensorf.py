import argparse
import glob
import os
import time
import sys

sys.path.insert(1, './nerf')
os.environ['GPU_DEBUG']='3'
import numpy as np
import torch
import torchvision
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from nerf.load_flame import load_flame_data

from nerf import (CfgNode, get_embedding_function, get_ray_bundle, img2mse,
                  load_llff_data, meshgrid_xy, models,
                  mse2psnr, run_one_iter_of_nerf, dump_rays, GaussianSmoothing)
from dataLoader import dataset_dict
from TensoRF.tensoRF import TensorVM, TensorCP, raw2alpha, TensorVMSplit, AlphaGridMask
#from gpu_profile import gpu_profile

def main():
    dataset = dataset_dict['blender']
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to (.yml) config file."
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default="",
        help="Path to load saved checkpoint from.",
    )
    configargs = parser.parse_args()

    # Read config file.
    cfg = None
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # # (Optional:) enable this to track autograd issues when debugging
    # torch.autograd.set_detect_anomaly(True)

    # If a pre-cached dataset is available, skip the dataloader.
    USE_CACHED_DATASET = False
    train_paths, validation_paths = None, None
    images, poses, render_poses, hwf, i_split, expressions = None, None, None, None, None, None
    H, W, focal, i_train, i_val, i_test = None, None, None, None, None, None
    if hasattr(cfg.dataset, "cachedir") and os.path.exists(cfg.dataset.cachedir):
        train_paths = glob.glob(os.path.join(cfg.dataset.cachedir, "train", "*.data"))
        validation_paths = glob.glob(
            os.path.join(cfg.dataset.cachedir, "val", "*.data")
        )
        USE_CACHED_DATASET = True
    else:
        # Load dataset
        images, poses, render_poses, hwf, expressions = None, None, None, None, None
        if cfg.dataset.type.lower() == "blender":
            from torch.utils.data import DataLoader
            from torch.utils.data import Dataset
            from nerf import nerface_dataloader
            # training_data = dataset('../../../datasets/nerface_dataset/person_4/', split='train', is_stack=False,N_vis = 100)
            training_data = nerface_dataloader.NerfaceDataset(
                mode='train',
                cfg=cfg,
                N_max=100
            )


            validation_data = nerface_dataloader.NerfaceDataset(
                mode='val',
                cfg=cfg,
                N_max=1
            )

            H = training_data.H
            W = training_data.W
            #img, pose, [H, W, focal], expression, probs = training_data[50]
            #train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)


            # images, poses, render_poses, hwf, i_split, expressions, _, bboxs = load_flame_data(
            #     cfg.dataset.basedir,
            #     half_res=cfg.dataset.half_res,
            #     testskip=cfg.dataset.testskip,
            # )
            # i_train, i_val, i_test = i_split
            # H, W, focal = hwf
            # H, W = int(H), int(W)
            # hwf = [H, W, focal]
    print("done loading data")
    # Seed experiment for repeatability
    seed = cfg.experiment.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Device on which to run.
    if torch.cuda.is_available():
        device = "cuda" #+ ":" + str(cfg.experiment.device)
    else:
        device = "cpu"

    encode_position_fn = get_embedding_function(
        num_encoding_functions=cfg.models.coarse.num_encoding_fn_xyz,
        include_input=cfg.models.coarse.include_input_xyz,
        log_sampling=cfg.models.coarse.log_sampling_xyz,
    )
    model_name = 'TensorVMSplit'
    # aabb = torch.tensor([[-0.19, -0.1872, -0.23], [0.1776, 0.1927, 0.3645]], device='cuda:0')
    # aabb = torch.tensor([[-0.05672, -0.1672, -0.23], [0.1176, 0.2927, 0.3645]], device='cuda:0')
    # aabb = torch.tensor([[-0.42067725,-0.43867725,-0.38817725], [0.41167725,0.39367725,0.44417725]], device='cuda:0')
    # aabb = torch.tensor([[-0.221, -0.152, -0.303], [0.2120, 0.107, 0.3590]], device='cuda:0')
    # aabb = torch.tensor([[-0.2172, -0.2672, -0.296], [0.1676, 0.3927, 0.3475]], device='cuda:0')
    aabb = torch.tensor([[-0.15, -0.15, -0.15], [0.15, 0.15, 0.15]], device='cuda:0')
    # aabb = torch.tensor([[-0.15672, -0.2672, -0.169], [0.1576, 0.3927, 0.1545]], device='cuda:0')
    # aabb = torch.tensor([[-0.05672, -0.2672, 0.169], [0.1576, 0.3927, 0.3545]], device='cuda:0')
    # aabb = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], device='cuda:0')
    # aabb = torch.tensor([[-0.0438, -0.797, -0.4], [0.1719, 0.491, 1.204]], device='cuda:0')
    reso_cur = [128, 128, 128]
    n_lamb_sigma = [16, 16, 16]
    n_lamb_sh = [48, 48, 48]
    data_dim_color = 27
    near_far = [0.2, 0.8]
    shadingMode = 'MLP_Fea'
    alphaMask_thresq = 0.0001
    density_shift = -10
    distance_scale = 25
    pos_pe = 6
    view_pe = 2
    fea_pe = 2
    featureC = 64
    step_ratio = 1.0
    fea2denseAct = 'softplus'
    tensorf = eval(model_name)(aabb, reso_cur, device,
                               density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=data_dim_color, near_far=near_far,
                               shadingMode=shadingMode, alphaMask_thres = alphaMask_thresq, density_shift = density_shift, distance_scale = distance_scale,
                               pos_pe = pos_pe, view_pe = view_pe, fea_pe = fea_pe, featureC = featureC, step_ratio = step_ratio)

    encode_direction_fn = None
    if cfg.models.coarse.use_viewdirs:
        encode_direction_fn = get_embedding_function(
            num_encoding_functions=cfg.models.coarse.num_encoding_fn_dir,
            include_input=cfg.models.coarse.include_input_dir,
            log_sampling=cfg.models.coarse.log_sampling_dir,
        )

    model_coarse = tensorf
    model_coarse.to(device)
    print(model_coarse)
    # If a fine-resolution model is specified, initialize it.
    model_fine = None
    model_fine = tensorf
    model_fine.to(device)

    ###################################
    ###################################
    train_background = False
    supervised_train_background = False
    blur_background = False

    train_latent_codes = True
    disable_expressions = False # True to disable expressions
    disable_latent_codes = False # True to disable latent codes
    fixed_background = True # Do False to disable BG
    regularize_latent_codes = True # True to add latent code LOSS, false for most experiments
    ###################################
    ###################################

    supervised_train_background = train_background and supervised_train_background
    # Avg background
    #images[i_train]
    if train_background: # TODO doesnt support dataloader!
        with torch.no_grad():
            avg_img = torch.mean(images[i_train],axis=0)
            # Blur Background:
            if blur_background:
                avg_img = avg_img.permute(2,0,1)
                avg_img = avg_img.unsqueeze(0)
                smoother = GaussianSmoothing(channels=3, kernel_size=11, sigma=11)
                print("smoothed background initialization. shape ", avg_img.shape)
                avg_img = smoother(avg_img).squeeze(0).permute(1,2,0)
            #avg_img = torch.zeros(H,W,3)
            #avg_img = torch.rand(H,W,3)
            #avg_img = 0.5*(torch.rand(H,W,3) + torch.mean(images[i_train],axis=0))
            background = torch.tensor(avg_img, device=device)
        background.requires_grad = True

    if fixed_background: # load GT background
        print("loading GT background to condition on")
        from PIL import Image
        background = Image.open(os.path.join(cfg.dataset.basedir,'bg','00050.png'))
        background.thumbnail((H,W))
        background = torch.from_numpy(np.array(background).astype(np.float32)).to(device)
        background=background[:,:,:3]
        background = background/255
        print("bg shape", background.shape)
        #print("should be ", training_data[0].shape)
        #assert background.shape[:2] == [training_data.H,training_data.W]
    else:
        background = None

    # Initialize optimizer.
    grad_vars = model_coarse.get_optparam_groups(0.02, 0.001)
    latent_codes = torch.zeros(len(training_data), 32, device=device)
    grad_vars.append({'params': latent_codes})
    grad_vars.append({'params': background, 'lr':0.0005})
    latent_codes.requires_grad = True
    lr_factor = 0.9999

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
    # Setup logging.
    logdir = os.path.join(cfg.experiment.logdir, cfg.experiment.id)
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)
    # Write out config parameters.
    with open(os.path.join(logdir, "config.yaml"), "w") as f:
        f.write(cfg.dump())  # cfg, f, default_flow_style=False)

    # By default, start at iteration 0 (unless a checkpoint is specified).
    start_iter = 0

    # Load an existing checkpoint, if a path is specified.
    if os.path.exists(configargs.load_checkpoint):
        checkpoint = torch.load(configargs.load_checkpoint)
        model_coarse.load_state_dict(checkpoint["model_coarse_state_dict"])
        if checkpoint["model_fine_state_dict"]:
            model_fine.load_state_dict(checkpoint["model_fine_state_dict"])
        if checkpoint["background"] is not None:
            print("loaded bg from checkpoint")
            background = torch.nn.Parameter(checkpoint['background'].to(device))
        if checkpoint["latent_codes"] is not None:
            print("loaded latent codes from checkpoint")
            latent_codes = torch.nn.Parameter(checkpoint['latent_codes'].to(device))

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]

    # # TODO: Prepare raybatch tensor if batching random rays

    # Prepare importance sampling maps
    # ray_importance_sampling_maps = []
    # p = 0.9
    # print("computing bounding boxes probability maps")
    # for i in i_train:
    #     bbox = bboxs[i]
    #     probs = np.zeros((H,W))
    #     probs.fill(1-p)
    #     probs[bbox[0]:bbox[1],bbox[2]:bbox[3]] = p
    #     probs = (1/probs.sum()) * probs
    #     ray_importance_sampling_maps.append(probs.reshape(-1))


    print("Starting loop")
    for i in trange(start_iter, cfg.experiment.train_iters):
        # if i == 3000:
        #     tensorf.gridSize = [512,512,64]
        #     tensorf.update_stepSize([512,512,64])
        #     tensorf.init_svd_volume(512, device)
        # if i == 1000:
        #     # print("that's it")
        #     # if reso_cur[0] * reso_cur[1] * reso_cur[2] < 256 ** 3:  # update volume resolution
        #     #     reso_mask = reso_cur
        #     # new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
        #     # if i == 3000:
        #     #     tensorf.shrink(new_aabb)
        #     tensorf.update_stepSize([512,512,64])

        model_coarse.train()
        # if model_fine:
        #     model_coarse.train()

        rgb_coarse, rgb_fine = None, None
        target_ray_values = None
        background_ray_values = None

        if USE_CACHED_DATASET:
            print("USE CACHE")
            datafile = np.random.choice(train_paths)
            cache_dict = torch.load(datafile)
            ray_bundle = cache_dict["ray_bundle"].to(device)
            ray_origins, ray_directions = (
                ray_bundle[0].reshape((-1, 3)),
                ray_bundle[1].reshape((-1, 3)),
            )
            target_ray_values = cache_dict["target"][..., :3].reshape((-1, 3))
            select_inds = np.random.choice(
                ray_origins.shape[0],
                size=(cfg.nerf.train.num_random_rays),
                replace=False,
            )
            ray_origins, ray_directions = (
                ray_origins[select_inds],
                ray_directions[select_inds],
            )
            target_ray_values = target_ray_values[select_inds].to(device)
            # target_ray_values = target_ray_values[select_inds].to(device)
            # ray_bundle = torch.stack([ray_origins, ray_directions], dim=0).to(device)

            rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                cache_dict["height"],
                cache_dict["width"],
                cache_dict["focal_length"],
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions=expressions
            )
        else:
            img_idx = np.random.choice(len(training_data))
            img, pose, [H, W, focal], expression, probs = training_data[img_idx]

            img_target = img.to(device)
            pose_target = pose[:3, :4].to(device)



            if not disable_expressions:
                expression_target = expression.to(device) # vector
            else: # zero expr
                expression_target = torch.zeros(76, device=device)
            #bbox = bboxs[img_idx]
            if not disable_latent_codes:
                latent_code = latent_codes[img_idx].to(device) if train_latent_codes else None
            else:
                latent_codes = torch.zeros(32, device=device)
            #latent_code = torch.zeros(32).to(device)
            ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose_target)

            coords = torch.stack(
                meshgrid_xy(torch.arange(H).to(device), torch.arange(W).to(device)),
                dim=-1,
            )
            # 生成图像的坐标集合
            # Only randomly choose rays that are in the bounding box !
            # coords = torch.stack(
            #     meshgrid_xy(torch.arange(bbox[0],bbox[1]).to(device), torch.arange(bbox[2],bbox[3]).to(device)),
            #     dim=-1,
            # )
            # 选像素坐标生成光线

            coords = coords.reshape((-1, 2))
            
            # select_inds = np.random.choice(
            #     coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False
            # )

            # Use importance sampling to sample mainly in the bbox with prob p
            select_inds = np.random.choice(
                coords.shape[0], size=(cfg.nerf.train.num_random_rays), replace=False, p=probs
            )

            select_inds = coords[select_inds]
            ray_origins = ray_origins[select_inds[:, 0], select_inds[:, 1], :]
            ray_directions = ray_directions[select_inds[:, 0], select_inds[:, 1], :]
            
            #dump_rays(ray_origins, ray_directions)

            # batch_rays = torch.stack([ray_origins, ray_directions], dim=0)
            target_s = img_target[select_inds[:, 0], select_inds[:, 1], :]
            background_ray_values = background[select_inds[:, 0], select_inds[:, 1], :] if (train_background or fixed_background) else None
            #if i<10000:
            #   background_ray_values = None
            #background_ray_values = None
            then = time.time()
            rgb_coarse, _, _, rgb_fine, _, _, weights = run_one_iter_of_nerf(
                H,
                W,
                focal,
                model_coarse,
                model_fine,
                ray_origins,
                ray_directions,
                cfg,
                mode="train",
                encode_position_fn=encode_position_fn,
                encode_direction_fn=encode_direction_fn,
                expressions = expression_target,
                background_prior=background_ray_values,
                latent_code = latent_code if not disable_latent_codes else torch.zeros(32,device=device)
            )

            target_ray_values = target_s
        coarse_loss = torch.nn.functional.mse_loss(
            rgb_coarse[..., :3], target_ray_values[..., :3]
        )
        fine_loss = None
        if rgb_fine is not None:
            fine_loss = torch.nn.functional.mse_loss(
                rgb_fine[..., :3], target_ray_values[..., :3]
            )
        # loss = torch.nn.functional.mse_loss(rgb_pred[..., :3], target_s[..., :3])
        loss = 0.0
        # if fine_loss is not None:
        #     loss = fine_loss
        # else:
        #     loss = coarse_loss

        latent_code_loss = torch.zeros(1, device=device)
        if train_latent_codes and not disable_latent_codes:
            latent_code_loss = torch.norm(latent_code) * 0.0005
            #latent_code_loss = torch.zeros(1)

        background_loss = torch.zeros(1, device=device)
        if supervised_train_background:
            background_loss = torch.nn.functional.mse_loss(
                background_ray_values[..., :3], target_ray_values[..., :3], reduction='none'
            ).sum(1)
            background_loss = torch.mean(background_loss*weights) * 0.001

        loss = coarse_loss + (fine_loss if fine_loss is not None else 0.0)
        psnr = mse2psnr(loss.item())

        #loss_total = loss #+ (latent_code_loss if latent_code_loss is not None else 0.0)
        loss = loss + (latent_code_loss*10 if regularize_latent_codes else 0.0)
        loss_total = loss + (background_loss if supervised_train_background is not None else 0.0)
        #loss.backward()
        loss_total.backward()
        #psnr = mse2psnr(loss.item())
        optimizer.step()
        optimizer.zero_grad()

        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * lr_factor

        if i % cfg.experiment.print_every == 0 or i == cfg.experiment.train_iters - 1:
            tqdm.write(
                "[TRAIN] Iter: "
                + str(i)
                + " Loss: "
                + str(loss.item())
                + " BG Loss: "
                + str(background_loss.item())
                + " PSNR: "
                + str(psnr)
                + " LatentReg: "
                + str(latent_code_loss.item())
            )
        #writer.add_scalar("train/loss", loss.item(), i)
        if train_latent_codes:
            writer.add_scalar("train/code_loss", latent_code_loss.item(), i)
        if supervised_train_background:
            writer.add_scalar("train/bg_loss", background_loss.item(), i)

        writer.add_scalar("train/coarse_loss", coarse_loss.item(), i)
        if rgb_fine is not None:
            writer.add_scalar("train/fine_loss", fine_loss.item(), i)
        writer.add_scalar("train/psnr", psnr, i)

        # Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1 and False
        ):
            #torch.cuda.empty_cache()
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            if model_fine:
                model_coarse.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, _, rgb_fine, _, weights = run_one_iter_of_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                        expressions = expression_target,
                        latent_code = torch.zeros(32, device=device)
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    # Do all validation set...
                    loss = 0
                    for img_idx in range(len(validation_data)):
                        #img_target = images[img_idx].to(device)

                        #img_idx = np.random.choice(len(validation_data))
                        img, pose, [H, W, focal], expression, _ = training_data[img_idx]

                        img_target = img.to(device)
                        pose_target = pose[:3, :4].to(device)

                        #tqdm.set_description('val im %d' % img_idx)
                        #tqdm.refresh()  # to show immediately the update

                            # # save val image for debug ### DEBUG ####
                        # #GT = target_ray_values[..., :3]
                        # import PIL.Image
                        # #img = GT.permute(2, 0, 1)
                        # # Conver to PIL Image and then np.array (output shape: (H, W, 3))
                        # #im_numpy = img_target.detach().cpu().numpy()
                        # #im_numpy = np.array(torchvision.transforms.ToPILImage()(img_target.detach().cpu()))
                        #
                        # #                   im = PIL.Image.fromarray(im_numpy)
                        # im = img_target
                        # im = im.permute(2, 0, 1)
                        # img = np.array(torchvision.transforms.ToPILImage()(im.detach().cpu()))
                        # im = PIL.Image.fromarray(img)
                        # im.save('val_im_target_debug.png')
                        # ### DEBUG #### END

                        #pose_target = poses[img_idx, :3, :4].to(device)
                        ray_origins, ray_directions = get_ray_bundle(
                            H, W, focal, pose_target
                        )
                        rgb_coarse, _, _, rgb_fine, _, _ ,weights= run_one_iter_of_nerf(
                            H,
                            W,
                            focal,
                            model_coarse,
                            model_fine,
                            ray_origins,
                            ray_directions,
                            cfg,
                            mode="validation",
                            encode_position_fn=encode_position_fn,
                            encode_direction_fn=encode_direction_fn,
                            expressions = expression_target,
                            background_prior = background.view(-1,3) if (train_background or fixed_background) else None,
                            latent_code = torch.zeros(32).to(device) if train_latent_codes or disable_latent_codes else None,

                        )
                        #print("did one val")
                        target_ray_values = img_target
                        coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                        curr_loss, curr_fine_loss = 0.0, 0.0
                        if rgb_fine is not None:
                            curr_fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                            curr_loss = curr_fine_loss
                        else:
                            curr_loss = coarse_loss
                        loss += curr_loss + curr_fine_loss

                loss /= len(validation_data)
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_scalar("validation/psnr", psnr, i)
                writer.add_image(
                    "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                )
                if rgb_fine is not None:
                    writer.add_image(
                        "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                    )
                    writer.add_scalar("validation/fine_loss", fine_loss.item(), i)

                writer.add_image(
                    "validation/img_target",
                    cast_to_image(target_ray_values[..., :3]),
                    i,
                )
                if train_background or fixed_background:
                    writer.add_image(
                        "validation/background", cast_to_image(background[..., :3]), i
                    )
                    writer.add_image(
                        "validation/weights", (weights.detach().cpu().numpy()), i, dataformats='HW'
                    )
                tqdm.write(
                    "Validation loss: "
                    + str(loss.item())
                    + " Validation PSNR: "
                    + str(psnr)
                    + " Time: "
                    + str(time.time() - start)
                )

        #gpu_profile(frame=sys._getframe(), event='line', arg=None)

        if i % cfg.experiment.save_every == 0 or i == cfg.experiment.train_iters - 1:
            checkpoint_dict = {
                "iter": i,
                "model_coarse_state_dict": model_coarse.state_dict(),
                "model_fine_state_dict": None
                if not model_fine
                else model_fine.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
                "psnr": psnr,
                "background": None
                if not (train_background or fixed_background)
                else background.data,
                "latent_codes": None if not train_latent_codes else latent_codes.data
            }
            torch.save(
                checkpoint_dict,
                os.path.join(logdir, "checkpoint" + str(i).zfill(5) + ".ckpt"),
            )
            tqdm.write("================== Saved Checkpoint =================")

    print("Done!")


def cast_to_image(tensor):
    # Input tensor is (H, W, 3). Convert to (3, H, W).
    tensor = tensor.permute(2, 0, 1)
    tensor = tensor.clamp(0.0,1.0)
    # Conver to PIL Image and then np.array (output shape: (H, W, 3))
    img = np.array(torchvision.transforms.ToPILImage()(tensor.detach().cpu()))
    # Map back to shape (3, H, W), as tensorboard needs channels first.
    img = np.moveaxis(img, [-1], [0])
    return img


def handle_pdb(sig, frame):
    import pdb
    pdb.Pdb().set_trace(frame)


if __name__ == "__main__":
    import signal

    print("before signal registration")
    signal.signal(signal.SIGUSR1, handle_pdb)
    print("after registration")
    #sys.settrace(gpu_profile)

    main()

"""
# Validation
        if (
            i % cfg.experiment.validate_every == 0
            or i == cfg.experiment.train_iters - 1
        ):
            tqdm.write("[VAL] =======> Iter: " + str(i))
            model_coarse.eval()
            if model_fine:
                model_coarse.eval()

            start = time.time()
            with torch.no_grad():
                rgb_coarse, rgb_fine = None, None
                target_ray_values = None
                if USE_CACHED_DATASET:
                    datafile = np.random.choice(validation_paths)
                    cache_dict = torch.load(datafile)
                    rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                        cache_dict["height"],
                        cache_dict["width"],
                        cache_dict["focal_length"],
                        model_coarse,
                        model_fine,
                        cache_dict["ray_origins"].to(device),
                        cache_dict["ray_directions"].to(device),
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = cache_dict["target"].to(device)
                else:
                    img_idx = np.random.choice(i_val)
                    img_target = images[img_idx].to(device)

                    # # save val image for debug ### DEBUG ####
                    # #GT = target_ray_values[..., :3]
                    # import PIL.Image
                    # #img = GT.permute(2, 0, 1)
                    # # Conver to PIL Image and then np.array (output shape: (H, W, 3))
                    # #im_numpy = img_target.detach().cpu().numpy()
                    # #im_numpy = np.array(torchvision.transforms.ToPILImage()(img_target.detach().cpu()))
                    #
                    # #                   im = PIL.Image.fromarray(im_numpy)
                    # im = img_target
                    # im = im.permute(2, 0, 1)
                    # img = np.array(torchvision.transforms.ToPILImage()(im.detach().cpu()))
                    # im = PIL.Image.fromarray(img)
                    # im.save('val_im_target_debug.png')
                    # ### DEBUG #### END


                    pose_target = poses[img_idx, :3, :4].to(device)
                    ray_origins, ray_directions = get_ray_bundle(
                        H, W, focal, pose_target
                    )
                    rgb_coarse, _, _, rgb_fine, _, _ = run_one_iter_of_nerf(
                        H,
                        W,
                        focal,
                        model_coarse,
                        model_fine,
                        ray_origins,
                        ray_directions,
                        cfg,
                        mode="validation",
                        encode_position_fn=encode_position_fn,
                        encode_direction_fn=encode_direction_fn,
                    )
                    target_ray_values = img_target
                coarse_loss = img2mse(rgb_coarse[..., :3], target_ray_values[..., :3])
                loss, fine_loss = 0.0, 0.0
                if rgb_fine is not None:
                    fine_loss = img2mse(rgb_fine[..., :3], target_ray_values[..., :3])
                    loss = fine_loss
                else:
                    loss = coarse_loss
                loss = coarse_loss + fine_loss
                psnr = mse2psnr(loss.item())
                writer.add_scalar("validation/loss", loss.item(), i)
                writer.add_scalar("validation/coarse_loss", coarse_loss.item(), i)
                writer.add_scalar("validataion/psnr", psnr, i)
                writer.add_image(
                    "validation/rgb_coarse", cast_to_image(rgb_coarse[..., :3]), i
                )
                if rgb_fine is not None:
                    writer.add_image(
                        "validation/rgb_fine", cast_to_image(rgb_fine[..., :3]), i
                    )
                    writer.add_scalar("validation/fine_loss", fine_loss.item(), i)

                writer.add_image(
                    "validation/img_target",
                    cast_to_image(target_ray_values[..., :3]),
                    i,
                )
                tqdm.write(
                    "Validation loss: "
                    + str(loss.item())
                    + " Validation PSNR: "
                    + str(psnr)
                    + " Time: "
                    + str(time.time() - start)
                )
"""