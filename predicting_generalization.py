import torch
import torch_geometric
import yaml
import torch.nn.functional as F
import os
from src.data import dataset
from tqdm import tqdm
from src.utils.setup_arg_parser import setup_arg_parser
from src.scalegmn.models import ScaleGMN
from src.utils.loss import select_criterion
from src.utils.optim import setup_optimization
from src.utils.helpers import overwrite_conf, count_parameters, assert_symms, set_seed, mask_input, mask_hidden, count_named_parameters
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import wandb
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def main():

    # read config file
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, vars(args))

    assert_symms(conf)

    print(yaml.dump(conf, default_flow_style=False))
    device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")

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

    train_loader = torch_geometric.loader.DataLoader(
        dataset=train_set,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
        pin_memory=True,
        sampler=None
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
    conf['scalegmn_args']["layer_layout"] = train_set.get_layer_layout()
    # conf['scalegmn_args']['input_nn'] = 'conv'
    net = ScaleGMN(conf['scalegmn_args'])
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
    optimizer, scheduler = setup_optimization(model_params, optimizer_name=conf_opt['optimizer_name'], optimizer_args=conf_opt['optimizer_args'], scheduler_args=conf_opt['scheduler_args'])
    # =============================================================================================
    # TRAINING LOOP
    # =============================================================================================
    step = 0
    best_val_tau = -float("inf")
    best_train_tau_TRAIN = -float("inf")
    best_test_results, best_val_results, best_train_results, best_train_results_TRAIN = None, None, None, None

    for epoch in range(conf['train_args']['num_epochs']):
        net.train()
        len_dataloader = len(train_loader)
        for i, batch in enumerate(tqdm(train_loader)):
            step = epoch * len_dataloader + i
            batch = batch.to(device)
            gt_test_acc = batch.y.to(device)

            optimizer.zero_grad()
            inputs = batch.to(device)
            pred_acc = F.sigmoid(net(inputs)).squeeze(-1)
            loss = criterion(pred_acc, gt_test_acc)
            loss.backward()
            log = {}
            if conf['optimization']['clip_grad']:
                log['grad_norm'] = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                                  conf['optimization']['clip_grad_max_norm']).item()

            optimizer.step()

            if conf["wandb"]:
                if step % 10 == 0:
                    log[f"train/{conf['train_args']['loss']}"] = loss.detach().cpu().item()
                    log["train/rsq"] = r2_score(gt_test_acc.cpu().numpy(), pred_acc.detach().cpu().numpy())

                wandb.log(log, step=step)

            if scheduler[1] is not None and scheduler[1] != 'ReduceLROnPlateau':
                scheduler[0].step()

        #############################################
        # VALIDATION
        #############################################
        if conf["validate"]:
            print(f"\nValidation after epoch {epoch}:")
            val_loss_dict = evaluate(net, val_loader, criterion, device)
            print(f"Epoch {epoch}, val L1 err: {val_loss_dict['avg_err']:.2f}, val loss: {val_loss_dict['avg_loss']:.2f}, val Rsq: {val_loss_dict['rsq']:.2f}, val tau: {val_loss_dict['tau']}")

            test_loss_dict = evaluate(net, test_loader, criterion, device)
            train_loss_dict = evaluate(net, train_loader, criterion, device)

            best_val_criteria = val_loss_dict['tau'] >= best_val_tau
            if best_val_criteria:
                best_val_tau = val_loss_dict['tau']
                best_test_results = test_loss_dict
                best_val_results = val_loss_dict
                best_train_results = train_loss_dict

            best_train_criteria = train_loss_dict['tau'] >= best_train_tau_TRAIN
            if best_train_criteria:
                best_train_tau_TRAIN = train_loss_dict['tau']
                best_train_results_TRAIN = train_loss_dict

            if conf["wandb"]:
                plt.clf()
                plot = plt.scatter(val_loss_dict['actual'], val_loss_dict['pred'])
                plt.xlabel("Actual model accuracy")
                plt.ylabel("Predicted model accuracy")
                wandb.log({
                    "train/l1_err": train_loss_dict['avg_err'],
                    "train/loss": train_loss_dict['avg_loss'],
                    "train/rsq": train_loss_dict['rsq'],
                    "train/kendall_tau": train_loss_dict['tau'],
                    "train/best_rsq": best_train_results['rsq'] if best_train_results is not None else None,
                    "train/best_tau": best_train_results['tau'] if best_train_results is not None else None,
                    "train/best_rsq_TRAIN_based": best_train_results_TRAIN['rsq'] if best_train_results_TRAIN is not None else None,
                    "train/best_tau_TRAIN_based": best_train_results_TRAIN['tau'] if best_train_results_TRAIN is not None else None,
                    "val/l1_err": val_loss_dict['avg_err'],
                    "val/loss": val_loss_dict['avg_loss'],
                    "val/rsq": val_loss_dict['rsq'],
                    "val/scatter": wandb.Image(plot),
                    "val/kendall_tau": val_loss_dict['tau'],
                    "val/best_rsq": best_val_results['rsq'] if best_val_results is not None else None,
                    "val/best_tau": best_val_results['tau'] if best_val_results is not None else None,
                    # test
                    "test/l1_err": test_loss_dict['avg_err'],
                    "test/loss": test_loss_dict['avg_loss'],
                    "test/rsq": test_loss_dict['rsq'],
                    "test/kendall_tau": test_loss_dict['tau'],
                    "test/best_rsq": best_test_results['rsq'] if best_test_results is not None else None,
                    "test/best_tau": best_test_results['tau'] if best_test_results is not None else None,
                    "epoch": epoch
                }, step=step)

            net.train()  # redundant


@torch.no_grad()
def evaluate(net, loader, loss_fn, device):
    net.eval()
    pred, actual = [], []
    err, losses = [], []
    for batch in loader:
        batch = batch.to(device)
        gt_test_acc = batch.y.to(device)
        inputs = batch.to(device)
        pred_acc = F.sigmoid(net(inputs)).squeeze(-1)

        err.append(torch.abs(pred_acc - gt_test_acc).mean().item())
        losses.append(loss_fn(pred_acc, gt_test_acc).item())
        pred.append(pred_acc.detach().cpu().numpy())
        actual.append(gt_test_acc.cpu().numpy())

    avg_err, avg_loss = np.mean(err), np.mean(losses)
    actual, pred = np.concatenate(actual), np.concatenate(pred)
    rsq = r2_score(actual, pred)
    tau = kendalltau(actual, pred).correlation

    return {
        "avg_err": avg_err,
        "avg_loss": avg_loss,
        "rsq": rsq,
        "tau": tau,
        "actual": actual,
        "pred": pred
    }


if __name__ == '__main__':
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]
    main()
