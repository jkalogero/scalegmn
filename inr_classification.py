import torch
import yaml
import torch.nn.functional as F
import os
import random
from src.data import dataset
from tqdm import tqdm, trange
from torch.utils import data
import torch_geometric
import torch.distributed as dist
from src.utils.setup_arg_parser import setup_arg_parser
from src.scalegmn.models import ScaleGMN
from src.utils.loss import select_criterion
from src.utils.optim import setup_optimization
from src.utils.helpers import overwrite_conf, count_parameters, set_seed, mask_input, mask_hidden, count_named_parameters
import wandb
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def main(args=None):

    # read config file
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, vars(args))

    torch.set_float32_matmul_precision('high')

    print(yaml.dump(conf, default_flow_style=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if conf["wandb"]:
        wandb.init(config=conf, **conf["wandb_args"])

    set_seed(conf['train_args']['seed'])

    # =============================================================================================
    #   SETUP DATASET AND DATALOADER
    # =============================================================================================
    extra_aug = conf['data'].pop('extra_aug') if 'extra_aug' in conf['data'] else 0
    equiv_on_hidden = mask_hidden(conf)
    get_first_layer_mask = mask_input(conf)

    train_set = dataset(conf['data'],
                        split='train',
                        debug=conf["debug"],
                        direction=conf['scalegmn_args']['direction'],
                        equiv_on_hidden=equiv_on_hidden,
                        get_first_layer_mask=get_first_layer_mask)
    conf['scalegmn_args']["layer_layout"] = train_set.get_layer_layout()

    # augment train set for the Augmented CIFAR-10 experiment
    if extra_aug > 0 and conf['data']['dataset'] == 'cifar_inr':
        aug_dsets = []
        for i in range(extra_aug):
            aug_dsets.append(dataset(conf['data'],
                                     split='train',
                                     debug=conf["debug"],
                                     prefix=f"randinit_smaller_aug{i}",
                                     direction=conf['scalegmn_args']['direction'],
                                     equiv_on_hidden=equiv_on_hidden,
                                     get_first_layer_mask=get_first_layer_mask
                                     )
                                    )
        train_set = data.ConcatDataset([train_set] + aug_dsets)
        print(f"Augmented training set with {len(train_set)} examples.")

    val_set = dataset(conf['data'],
                      split='val',
                      debug=conf["debug"],
                      # node_pos_embed=conf['scalegmn_args']['graph_constructor']['node_pos_embed'],
                      # edge_pos_embed=conf['scalegmn_args']['graph_constructor']['edge_pos_embed'],
                      direction=conf['scalegmn_args']['direction'],
                      equiv_on_hidden=equiv_on_hidden,
                      get_first_layer_mask=get_first_layer_mask)

    test_set = dataset(conf['data'],
                        split='test',
                        debug=conf["debug"],
                      # node_pos_embed=conf['scalegmn_args']['graph_constructor']['node_pos_embed'],
                      # edge_pos_embed=conf['scalegmn_args']['graph_constructor']['edge_pos_embed'],
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
    best_val_acc = -1
    best_train_acc = -1
    best_test_results, best_val_results, best_train_results, best_train_results_TRAIN = None, None, None, None
    last_val_accs = []
    patience = conf['train_args']['patience']
    if extra_aug:
        # run experiment like in NFN to have comparable results.
        train_on_steps(net, train_loader, val_loader, test_loader, optimizer, scheduler, criterion, conf, device)
    else:
        for epoch in range(conf['train_args']['num_epochs']):
            net.train()
            curr_loss = 0
            len_dataloader = len(train_loader)
            for i, batch in enumerate(tqdm(train_loader)):
                step = epoch * len_dataloader + i
                batch = batch.to(device)

                optimizer.zero_grad()
                out = net(batch)

                loss = criterion(out, batch.label)
                loss.backward()
                log = {}
                if conf['optimization']['clip_grad']:
                    log['grad_norm'] = torch.nn.utils.clip_grad_norm_(net.parameters(), conf['optimization']['clip_grad_max_norm'])

                optimizer.step()

                if conf["wandb"]:
                    log[f"train/{conf['train_args']['loss']}"] = loss.item()
                    log["epoch"] = epoch

                if scheduler[1] is not None and scheduler[1] != 'ReduceLROnPlateau':
                    log["lr"] = scheduler[0].get_last_lr()[0]
                    scheduler[0].step()

                if conf["wandb"]:
                    wandb.log(log, step=step)

            #############################################
            # VALIDATION
            #############################################
            if conf["validate"]:
                print(f"\nValidation after epoch {epoch}:")
                val_loss_dict = evaluate(net, val_loader, device=device)
                test_loss_dict = evaluate(net, test_loader, device=device)
                val_loss = val_loss_dict["avg_loss"]
                val_acc = val_loss_dict["avg_acc"]
                test_loss = test_loss_dict["avg_loss"]
                test_acc = test_loss_dict["avg_acc"]

                train_loss_dict = evaluate(net, train_loader, train_set, num_samples=len(val_set), batch_size=conf["batch_size"], num_workers=conf["num_workers"], device=device)

                best_val_criteria = val_acc >= best_val_acc

                if best_val_criteria:
                    best_val_acc = val_acc
                    best_test_results = test_loss_dict
                    best_val_results = val_loss_dict
                    best_train_results = train_loss_dict

                best_train_criteria = train_loss_dict["avg_acc"] >= best_train_acc
                if best_train_criteria:
                    best_train_acc = train_loss_dict["avg_acc"]
                    best_train_results_TRAIN = train_loss_dict

                if conf["wandb"]:
                    log = {
                        "train/avg_loss": train_loss_dict["avg_loss"],
                        "train/acc": train_loss_dict["avg_acc"],
                        "train/conf_mat": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=train_loss_dict["gt"],
                            preds=train_loss_dict["predicted"],
                            class_names=range(10),
                        ),
                        "train/best_loss": best_train_results["avg_loss"],
                        "train/best_acc": best_train_results["avg_acc"],
                        "train/best_loss_TRAIN_based": best_train_results_TRAIN["avg_loss"],
                        "train/best_acc_TRAIN_based": best_train_results_TRAIN["avg_acc"],
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "val/best_loss": best_val_results["avg_loss"],
                        "val/best_acc": best_val_results["avg_acc"],
                        "val/conf_mat": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=val_loss_dict["gt"],
                            preds=val_loss_dict["predicted"],
                            class_names=range(10),
                        ),
                        "test/loss": test_loss,
                        "test/acc": test_acc,
                        "test/best_loss": best_test_results["avg_loss"],
                        "test/best_acc": best_test_results["avg_acc"],
                        "test/conf_mat": wandb.plot.confusion_matrix(
                            probs=None,
                            y_true=test_loss_dict["gt"],
                            preds=test_loss_dict["predicted"],
                            class_names=range(10),
                        ),
                        "epoch": epoch,
                    }

                    wandb.log(log)

                net.train()

                last_val_accs.append(val_acc)
                # Keep only the last accuracies
                if len(last_val_accs) > patience:
                    last_val_accs.pop(0)
                # Check if the accuracies are decreasing
                if len(last_val_accs) == patience and all(x > y for x, y in zip(last_val_accs, last_val_accs[1:])):
                    print(f"Validation accuracy has been dropping for {patience} consecutive epochs:\n{last_val_accs}\nExiting.")
                    return 1

@torch.no_grad()
def evaluate(model, loader, eval_dataset=None, num_samples=0, batch_size=0, num_workers=8, device=None):
    if eval_dataset is not None:
        # only when also evaluating on train split. Since it is a lot bigger, we only evaluate on a smaller subset.
        indices = random.sample(range(len(eval_dataset)), num_samples)
        subset_dataset = torch.utils.data.Subset(eval_dataset, indices)
        loader = torch_geometric.loader.DataLoader(
            dataset=subset_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            # sampler=sampler
        )

    model.eval()
    loss = 0.0
    correct = 0.0
    total = 0.0
    predicted, gt = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        loss += F.cross_entropy(out, batch.label, reduction="sum")
        total += len(batch.label)
        pred = out.argmax(1)
        correct += pred.eq(batch.label).sum()
        predicted.extend(pred.cpu().numpy().tolist())
        gt.extend(batch.label.cpu().numpy().tolist())

    model.train()
    avg_loss = loss / total
    avg_acc = correct / total

    return dict(avg_loss=avg_loss, avg_acc=avg_acc, predicted=predicted, gt=gt)


def train_on_steps(net, train_loader, val_loader, test_loader, optimizer, scheduler, criterion, train_conf, device):
    """
    Follow the same training procedure as in 'Zhou, Allan, et al. "Permutation equivariant neural functionals." NIPS (2024).'
    """
    import wandb

    def cycle(loader):
        while True:
            for blah in loader:
                yield blah

    best_val_acc = -1
    best_test_results, best_val_results = None, None

    train_iter = cycle(train_loader)
    outer_pbar = trange(0, train_conf['train_args']['max_steps'], position=0)

    for step in outer_pbar:

        if step > 0 and step % 3000 == 0 or step == train_conf['train_args']['max_steps'] - 1:
            val_loss_dict = evaluate(net, val_loader, device=device)
            test_loss_dict = evaluate(net, test_loader, device=device)
            val_loss = val_loss_dict["avg_loss"]
            val_acc = val_loss_dict["avg_acc"]
            test_loss = test_loss_dict["avg_loss"]
            test_acc = test_loss_dict["avg_acc"]
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_results = test_loss_dict
                best_val_results = val_loss_dict

            if train_conf["wandb"]:
                wandb.log({
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/best_loss": best_val_results["avg_loss"],
                    "val/best_acc": best_val_results["avg_acc"],
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "test/best_loss": best_test_results["avg_loss"],
                    "test/best_acc": best_test_results["avg_acc"],
                    "step": step,
                    "epoch": step // len(train_loader),
                })

        net.train()
        batch = next(train_iter)
        batch = batch.to(device)
        optimizer.zero_grad()
        inputs = (batch)
        out = net(inputs)
        loss = criterion(out, batch.label)
        loss.backward()

        log = {}
        if train_conf['optimization']['clip_grad']:
            log['grad_norm'] = torch.nn.utils.clip_grad_norm_(net.parameters(), train_conf['optimization']['clip_grad_max_norm'])

        optimizer.step()

        if train_conf["wandb"]:
            log[f"train/{train_conf['train_args']['loss']}"] = loss.item()
            log["step"] = step

        if scheduler[1] is not None and scheduler[1] != 'ReduceLROnPlateau':
            log["lr"] = scheduler[0].get_last_lr()[0]
            scheduler[0].step()

        if train_conf["wandb"]:
            wandb.log(log, step=step+1)

if __name__ == '__main__':
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    conf = yaml.safe_load(open(args.conf))

    main(args=args)
