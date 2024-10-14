import torch
import yaml
import numpy as np
import os
from tqdm import trange
import torch_geometric
from einops import rearrange
from src.data import dataset
from src.utils.setup_arg_parser import setup_arg_parser
from src.scalegmn.models import ScaleGMN_equiv
from src.utils.loss import select_criterion
from src.utils.optim import setup_optimization
from src.utils.helpers import overwrite_conf, count_parameters, set_seed, mask_input, mask_hidden
from src.scalegmn.inr import BatchSiren
import wandb
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def main(args=None):

    # read config file
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, vars(args))

    # only for sweeps
    torch.set_float32_matmul_precision('high')
    print(yaml.dump(conf, default_flow_style=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if conf["wandb"]:
        wandb.init(config=conf, **conf["wandb_args"])

    set_seed(conf['train_args']['seed'])

    # =============================================================================================
    #   SETUP DATASET AND DATALOADER
    # =============================================================================================
    equiv_on_hidden = mask_hidden(conf)
    get_first_layer_mask = mask_input(conf)
    train_set = dataset(conf['data'],
                        split='train',
                        debug=conf["debug"],
                        direction=conf['scalegmn_args']['direction'],
                        equiv_on_hidden=equiv_on_hidden,
                        get_first_layer_mask=get_first_layer_mask)
    conf['scalegmn_args']["layer_layout"] = train_set.get_layer_layout()

    val_set = dataset(conf['data'],
                      split='val',
                      debug=conf["debug"],
                      direction=conf['scalegmn_args']['direction'],
                      equiv_on_hidden=equiv_on_hidden,
                      get_first_layer_mask=get_first_layer_mask)

    test_set = dataset(conf['data'],
                       split='test',
                       debug=conf["debug"],
                       direction=conf['scalegmn_args']['direction'],
                       equiv_on_hidden=equiv_on_hidden,
                       get_first_layer_mask=get_first_layer_mask)

    print(f'Len train set: {len(train_set)}')
    print(f'Len val set: {len(val_set)}')
    print(f'Len test set: {len(test_set)}')

    inr_model = BatchSiren(**conf['inr_model']).to(device)

    train_loader = torch_geometric.loader.DataLoader(
        dataset=train_set,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
        pin_memory=True,
        sampler=None,
    )
    val_loader = torch_geometric.loader.DataLoader(
        dataset=val_set,
        batch_size=conf["batch_size"],
        shuffle=False,
    )
    test_loader = torch_geometric.loader.DataLoader(
        dataset=test_set,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
        pin_memory=True,
    )

    # =============================================================================================
    #   DEFINE MODEL
    # =============================================================================================
    net = ScaleGMN_equiv(conf['scalegmn_args'])

    print(net)
    cnt_p = count_parameters(net=net)
    if conf["wandb"]:
        wandb.log({'number of parameters': cnt_p}, step=0)

    for p in net.parameters():
        p.requires_grad = True

    net = net.to(device)

    # =============================================================================================
    #   DEFINE LOSS
    # =============================================================================================
    criterion = select_criterion(conf['train_args']['loss'], {})

    # =============================================================================================
    #   DEFINE OPTIMIZATION
    # =============================================================================================
    conf_opt = conf['optimization']
    model_params = [p for p in net.parameters() if p.requires_grad]
    optimizer, scheduler = setup_optimization(model_params, optimizer_name=conf_opt['optimizer_name'],
                                              optimizer_args=conf_opt['optimizer_args'],
                                              scheduler_args=conf_opt['scheduler_args'])

    best_val_loss = float("inf")
    best_test_results, best_val_results = None, None
    test_loss = -1.0
    global_step = 0
    start_epoch = 0

    epoch_iter = trange(start_epoch, conf['train_args']['num_epochs'])
    net.train()
    optimizer.zero_grad()

    for epoch in epoch_iter:
        for i, batch in enumerate(train_loader):
            params, w_b, img, input_image = batch
            params = params.to(device)
            w_b = w_b.to(device)
            weights = w_b.weights
            biases = w_b.biases
            img = img.to(device)

            optimizer.zero_grad()

            delta_weights, delta_biases = net(params, weights, biases)
            new_weights, new_biases = residual_param_update(weights, biases, delta_weights, delta_biases)

            outs = inr_model(new_weights, new_biases)
            outs = rearrange(outs, "b (h w) c -> b c h w", h=conf['data']['image_size'][0])
            loss = criterion(outs, img)
            loss.backward()

            log = {
                "train/loss": loss.item(),
                "global_step": global_step,
            }

            if conf['optimization']['clip_grad']:
                grad_norm = torch.nn.utils.clip_grad_norm_(list(filter(lambda p: p.requires_grad, net.parameters())), conf['optimization']['clip_grad_max_norm'])
                log["grad_norm"] = grad_norm

            optimizer.step()

            if scheduler[1] is not None and scheduler[1] != 'ReduceLROnPlateau':
                log["lr"] = scheduler[0].get_last_lr()[0]
                scheduler[0].step()

            if conf["wandb"]:
                if i == 0:
                    (
                        log["train/imgs/gt"],
                        log["train/imgs/pred"],
                        log["train/imgs/input"],
                    ) = log_images(
                        img[: conf['log_n_imgs']],
                        outs[: conf['log_n_imgs']],
                        input_image[: conf['log_n_imgs']]
                    )
                wandb.log(log)

                epoch_iter.set_description(
                    f"[{epoch} {i+1}], train loss: {loss.item():.3f}, test_loss: {test_loss:.3f}"
                )
            global_step += 1

            if (global_step + 1) % conf['train_args']['eval_every'] == 0:
                val_loss_dict = evaluate(
                    net,
                    val_loader,
                    device,
                    inr_model,
                    img_shape=conf['data']['image_size'],
                    log_n_imgs=conf['log_n_imgs'],
                )
                test_loss_dict = evaluate(
                    net,
                    test_loader,
                    device,
                    inr_model,
                    img_shape=conf['data']['image_size'],
                    log_n_imgs=conf['log_n_imgs'],
                )

                val_loss = val_loss_dict["avg_loss"]
                test_loss = test_loss_dict["avg_loss"]
                train_loss_dict = evaluate(
                    net,
                    train_loader,
                    device,
                    inr_model,
                    img_shape=conf['data']['image_size'],
                    log_n_imgs=conf['log_n_imgs'],
                    num_batches=100,
                )

                best_val_criteria = val_loss < best_val_loss

                if best_val_criteria:
                    best_test_results = test_loss_dict
                    best_val_results = val_loss_dict
                    best_val_loss = val_loss
                if conf["wandb"]:
                    log = {
                        "train/avg_loss": train_loss_dict["avg_loss"],
                        "val/best_loss": best_val_results["avg_loss"],
                        "test/best_loss": best_test_results["avg_loss"],
                        **{f"val/{k}": v for k, v in val_loss_dict.items()},
                        **{f"test/{k}": v for k, v in test_loss_dict.items()},
                        "epoch": epoch,
                        "global_step": global_step,
                    }

                    wandb.log(log)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    inr_model,
    img_shape=(28, 28),
    log_n_imgs=0,
    num_batches=None,
):
    model.eval()
    log_n_imgs = min(log_n_imgs, loader.batch_size)
    imgs, preds, input_imgs = [], [], []
    losses = []
    for i, batch in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break
        params, w_b, img, input_image = batch
        params = params.to(device)
        w_b = w_b.to(device)
        weights = w_b.weights
        biases = w_b.biases
        img = img.to(device)

        delta_weights, delta_biases = model(params, weights, biases)
        new_weights, new_biases = residual_param_update(weights, biases, delta_weights, delta_biases)

        outs = inr_model(new_weights, new_biases)
        outs = rearrange(outs, "b (h w) c -> b c h w", h=img_shape[0])
        loss = ((outs - img) ** 2).mean(dim=(1, 2, 3))
        losses.append(loss.detach().cpu())

        if i == 0 and log_n_imgs > 0:
            gt_img_plots, pred_img_plots, input_img_plots = log_images(
                img[:log_n_imgs],
                outs[:log_n_imgs],
                input_image[:log_n_imgs]
            )
            imgs.extend(gt_img_plots)
            preds.extend(pred_img_plots)
            input_imgs.extend(input_img_plots)

    losses = torch.cat(losses)
    losses = losses.mean()

    model.train()
    return {
        "avg_loss": losses,
        "imgs/gt": imgs,
        "imgs/pred": preds,
        "imgs/input": input_imgs,
    }


def residual_param_update(weights, biases, delta_weights, delta_biases):
    new_weights = [weights[j] + delta_weights[j] for j in range(len(weights))]
    new_biases = [biases[j] + delta_biases[j] for j in range(len(weights))]
    return new_weights, new_biases


def log_images(gt_images, pred_images, input_images):
    _gt_images = gt_images.detach().permute(0, 2, 3, 1).cpu().numpy()
    _pred_images = pred_images.detach().permute(0, 2, 3, 1).cpu().numpy()
    _input_images = input_images.detach().permute(0, 2, 3, 1).cpu().numpy()
    _pred_images = np.clip(_pred_images, 0.0, 1.0)
    gt_img_plots = [wandb.Image(img) for img in gt_images]
    pred_img_plots = [wandb.Image(img) for img in pred_images]
    input_img_plots = [wandb.Image(img) for img in _input_images]
    return gt_img_plots, pred_img_plots, input_img_plots


if __name__ == '__main__':
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    main(args=args)
