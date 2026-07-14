import argparse
import time
import csv
import datetime
import math
import torchvision.transforms as transforms
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
from spikingjelly_custom.spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from spikingjelly_custom.spikingjelly.clock_driven import functional
import myTransform
from mixup import Mixup
from functions import *
from models.VGGSNN_PML_TACA import VGGSNN_PML_TACA



def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def save_on_master(state, path):
    if is_main_process():
        torch.save(state, path)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    dummy = torch.tensor([0.0], device=f"cuda:{args.gpu}")
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}, gpu {args.gpu}): {args.dist_url}', flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.all_reduce(dummy)
    dist.barrier()
    setup_for_distributed(args.rank == 0)


# ================================ Argument Parser ================================
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-j', '--workers', default=7, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=8, type=int, help='batch size per GPU')
parser.add_argument('--lr', default=0.05, type=float, help='initial learning rate')
parser.add_argument('--seed', default=[36, 42, 50], nargs='+', type=int, help='list of seeds')
parser.add_argument('--T', default=10, type=int, help='SNN simulation time')
# Loss
parser.add_argument('--TET', action='store_false', help='use TET loss')
parser.add_argument('--TET_class', type=str, default='TET_loss', help='TET loss type')
parser.add_argument('--loss_eta', default=0.1, type=float, help='eta for loss')
parser.add_argument('--loss_lamb', default=0.06, type=float, help='lambda for loss')
parser.add_argument('--loss_beta', default=0.001, type=float, help='beta for loss')
parser.add_argument('--loss_sigma', default=0.5, type=float, help='sigma for loss')
parser.add_argument('--loss_means', default=1.0, type=float, help='means for loss')
# System
parser.add_argument('--Tensorboard', action='store_true', help='Enable TensorBoard logging')
parser.add_argument('--amp', action='store_true', help='Enable mixed precision training')
# Distributed training parameters
parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist-url', default='env://', help='url for distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='local rank for distributed training (automatically set by torchrun)')
# Data
parser.add_argument('--data_name', default='CIFAR10DVS', type=str, help='dataset name')
parser.add_argument('--data_path', default='./dataset/SNN/CIFAR10-DVS/', type=str, help='data path')
parser.add_argument('--input_size', default=48, type=int, help='input size H*W')
parser.add_argument('--few_shot', action='store_true', help='use few-shot training')
parser.add_argument('--few_shot_ratio', default=0.01, type=float, help='ratio for few-shot')
# Model
parser.add_argument('--method', default='VGGSNN_PML', type=str, help='network type')
parser.add_argument('--tau', default=0.25, type=float, help='tau of LIF')
parser.add_argument('--TAU', default='Temporal_Attention_Unit_Concat', type=str, help='')
parser.add_argument('--weights', type=float, default=0.5, help='')
parser.add_argument('--num_class', default=10, type=int, help='')

# PML
parser.add_argument('--pml_temperature', type=float, default=2.8, help='Temperature for PML KD')
parser.add_argument('--pml_alpha', type=float, default=0.3, help='Alpha balance in PML')
parser.add_argument('--pml_lamb', type=float, default=0.1, help='Lambda scale in PML')
parser.add_argument('--pml_pads', type=int, default=256, help='Channels padding for surrogate blocks')

# Optimization
parser.add_argument('--opt', default='SGD', type=str, help='optimizer method')
parser.add_argument('--cos', action='store_true', help='use cosine learning rate')
parser.add_argument('--warmup_epochs', default=10, type=int, help='warmup epochs')
parser.add_argument('--cos_max_lr', default=0.1, type=float, help='max lr for cosine')
parser.add_argument('--cos_min_lr', default=0.00001, type=float, help='min lr for cosine')
parser.add_argument('--mixup', action='store_true', help='use mixup')
# Output
parser.add_argument('--out_dir', default='./logs/', type=str, help='output directory')
parser.add_argument('--resume', type=str, help='resume from checkpoint')
parser.add_argument('--auto-resume', action='store_true', help='auto resume from checkpoint in output dir')
parser.add_argument('--tta', type=int, default=0, metavar='N', help='Test/inference time augmentation factor')

args = parser.parse_args()

# Initialize distributed mode
init_distributed_mode(args)

# Set device
device = torch.device(f"cuda:{args.gpu}") if args.distributed else torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")

# Worker optimization
num_cpus_per_process = max(1, os.cpu_count() // args.world_size)
args.workers = min(args.workers, num_cpus_per_process)
torch.set_num_threads(1)

if is_main_process():
    print(f"Set torch.set_num_threads to {torch.get_num_threads()}")
    print(f"Calculated num_workers per process: {args.workers}")


# ================================ Functions ================================
def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def adjust_learning_rate_cos(optimizer, epoch, args):
    if epoch < args.warmup_epochs:
        lr = args.cos_min_lr + (args.cos_max_lr - args.cos_min_lr) * (epoch / args.warmup_epochs)
    else:
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        lr = args.cos_min_lr + 0.5 * (args.cos_max_lr - args.cos_min_lr) * (1 + math.cos(math.pi * progress))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(model, device, train_loader, criterion, optimizer, epoch, args, logger=None, writer=None):
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp) if args.amp else None
    data_iterator = tqdm.tqdm(train_loader) if is_main_process() else train_loader

    total_loss = 0.0
    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)

    mixup_fn = None
    if args.mixup:
        mixup_args = dict(
            mixup_alpha=0.5, cutmix_alpha=0., cutmix_minmax=None,
            prob=0.5, switch_prob=0.5, mode="batch",
            label_smoothing=0.1, num_classes=args.num_class)
        mixup_fn = Mixup(**mixup_args)

    start_time = time.time()

    for i, (images, labels) in enumerate(data_iterator):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.float().to(device)

        if args.mixup and mixup_fn:
            images, labels_mix = mixup_fn(images, labels)
            labels_acc = labels_mix.argmax(dim=-1)
        else:
            labels_acc = labels

        with torch.amp.autocast("cuda", enabled=args.amp):
            outputs = model(images)


            if isinstance(outputs, list):
                final_out = outputs[0]
                mean_outputs = [out.mean(1) for out in outputs]
            else:
                final_out = outputs
                mean_outputs = [outputs.mean(1)]

            mean_final_out = mean_outputs[0]


            if args.TET:
                loss_labels = labels_mix if args.mixup and mixup_fn else labels
                if args.TET_class == "TET_loss":
                    base_loss = TET_loss(final_out, loss_labels, criterion, args.loss_means, args.loss_lamb)
                else:
                    base_loss = criterion(mean_final_out, labels_acc)
            else:
                base_loss = criterion(mean_final_out, labels_acc)


            if isinstance(outputs, list) and len(outputs) > 1:
                base_guide_loss = 0.0
                pml_loss = 0.0
                num_surrogates = len(outputs) - 1

                for k in range(1, len(outputs)):

                    if args.TET:
                        loss_labels = labels_mix if args.mixup and mixup_fn else labels
                        if args.TET_class == "TET_loss":
                            base_guide_loss += TET_loss(outputs[k], loss_labels, criterion, args.loss_means, args.loss_lamb)
                        else:
                            base_guide_loss += criterion(mean_outputs[k], labels_acc)
                    else:
                        base_guide_loss += criterion(mean_outputs[k], labels_acc)


                    if k == len(outputs) - 1:
                        teacher_out = mean_outputs[0].detach()
                    else:
                        teacher_out = mean_outputs[k + 1].detach()

                    student_out = mean_outputs[k]


                    student_log_probs = F.log_softmax(student_out / args.pml_temperature, dim=1)
                    teacher_probs = F.softmax(teacher_out / args.pml_temperature, dim=1)
                    kl_div = nn.KLDivLoss(reduction='batchmean')(student_log_probs, teacher_probs) * (
                                args.pml_temperature * args.pml_temperature)
                    pml_loss += kl_div


                loss = (1 - args.pml_lamb) * base_loss + args.pml_lamb * (
                    (1 - args.pml_alpha) * (base_guide_loss / num_surrogates) + args.pml_alpha * (
                            pml_loss / num_surrogates))
            else:
                loss = base_loss

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)

        total += labels_acc.size(0)
        _, predicted = mean_final_out.max(1)
        correct += predicted.eq(labels_acc).sum()
        total_loss += loss.item() * labels_acc.size(0)


    total_val = 0.0
    correct_val = 0.0
    if args.distributed:
        dist.barrier()
        reduced_loss = torch.tensor(total_loss, device=device)
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        total_loss = reduced_loss.item()
        correct_val = correct.item()
        total_val = total.item()
    else:
        correct_val = correct.item()
        total_val = total.item()

    train_loss = total_loss / total_val if total_val > 0 else 0.0
    train_acc = 100. * correct_val / total_val if total_val > 0 else 0.0

    if is_main_process() and writer and args.Tensorboard:
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

    return train_loss, train_acc, time.time() - start_time


@torch.no_grad()
def test(model, test_loader, device, criterion, epoch=None, logger=None, writer=None):
    model.eval()
    total_loss = 0.0
    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)

    data_iterator = tqdm.tqdm(test_loader) if is_main_process() else test_loader

    for images, labels in data_iterator:
        images = images.float().to(device)
        labels = labels.to(device)

        with torch.amp.autocast("cuda", enabled=args.amp):
            if args.tta == 2:
                outputs_orig = model(images)
                if isinstance(outputs_orig, list): outputs_orig = outputs_orig[0]
                mean_out_orig = outputs_orig.mean(1)
                loss_orig = criterion(mean_out_orig, labels)

                images_flip = transforms.functional.hflip(images)
                outputs_flip = model(images_flip)
                if isinstance(outputs_flip, list): outputs_flip = outputs_flip[0]
                mean_out_flip = outputs_flip.mean(1)
                loss_flip = criterion(mean_out_flip, labels)

                loss = (loss_orig + loss_flip) / 2.0
                probs_avg = (mean_out_orig.softmax(dim=-1) + mean_out_flip.softmax(dim=-1)) / 2.0
                _, predicted = probs_avg.max(1)
            else:
                outputs = model(images)
                if isinstance(outputs, list): outputs = outputs[0]
                mean_out = outputs.mean(1)
                loss = criterion(mean_out, labels)
                _, predicted = mean_out.max(1)

        functional.reset_net(model)

        total += labels.size(0)
        correct += predicted.eq(labels).sum()
        total_loss += loss.item() * labels.size(0)

    total_val = 0.0
    correct_val = 0.0
    if args.distributed:
        dist.barrier()
        reduced_loss = torch.tensor(total_loss, device=device)
        dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        total_loss = reduced_loss.item()
        correct_val = correct.item()
        total_val = total.item()
    else:
        correct_val = correct.item()
        total_val = total.item()

    test_loss = total_loss / total_val if total_val > 0 else 0.0
    test_acc = 100. * correct_val / total_val if total_val > 0 else 0.0

    if is_main_process() and writer and args.Tensorboard and epoch is not None:
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_acc, epoch)

    return test_loss, test_acc


def build_dataset():
    transform_train = transforms.Compose([
        myTransform.ToTensor(),
        transforms.Resize(size=(args.input_size, args.input_size)),
        transforms.RandomCrop(args.input_size, padding=4),
        transforms.RandomHorizontalFlip(), ])

    transform_test = transforms.Compose([
        myTransform.ToTensor(),
        transforms.Resize(size=(args.input_size, args.input_size))
    ])


    if args.data_name == 'CIFAR10DVS':
        train_set = CIFAR10DVS(root=args.data_path, train=True, data_type='frame', frames_number=args.T,
                               split_by='number', transform=transform_train)
        test_set = CIFAR10DVS(root=args.data_path, train=False, data_type='frame', frames_number=args.T,
                              split_by='number', transform=transform_test)
    else:
        print(args.data_name, " is not supported")
        raise ValueError("Dataset not supported")
    return train_set, test_set


if __name__ == '__main__':
    best_acc_list = []


    class SimpleLogger:
        def __init__(self, log_path):
            self.log_path = log_path
            if is_main_process():
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                open(self.log_path, 'w').close()

        def info(self, msg):
            if is_main_process():
                print(msg)
                with open(self.log_path, 'a') as f:
                    f.write(msg + '\n')

        def set_log_path(self, new_path):
            self.log_path = new_path
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            open(self.log_path, 'a').close()


    try:
        for seed in args.seed:
            seed_all(seed)
            train_dataset, val_dataset = build_dataset()

            if args.few_shot:
                original_size = len(train_dataset)
                sample_size = int(original_size * args.few_shot_ratio)
                indices = torch.randperm(original_size)[:sample_size]
                train_dataset = torch.utils.data.Subset(train_dataset, indices)
                if is_main_process():
                    print(f"Few-shot: Using {sample_size}/{original_size} samples")

            if args.distributed:
                test_sampler = DistributedSampler(val_dataset, num_replicas=args.world_size, rank=args.rank,
                                                  shuffle=False)
                test_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=args.batch_size * 2, sampler=test_sampler,
                    num_workers=args.workers, pin_memory=True, drop_last=False)
            else:
                test_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=args.batch_size * 2, shuffle=False,
                    num_workers=args.workers, pin_memory=True, drop_last=False)


            if args.method == 'VGGSNN_PML_TACA':

                pml_places = [1, 2, 3]
                pml_kernels = [[7, 5, 3], [7, 5, 3], [7, 5, 3]]
                model = VGGSNN_PML_TACA(tau=args.tau, T=args.T, num_class=args.num_class, input_size=args.input_size,
                                   pml_places=pml_places, pml_kernels=pml_kernels, pml_pads=args.pml_pads)


            else:
                raise NotImplementedError(f"Model {args.method} not implemented")

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model.to(device)

            parallel_model = model
            if args.distributed:
                parallel_model = DDP(model, device_ids=[args.gpu], find_unused_parameters=False)
                if is_main_process():
                    print(f"Using DDP on {args.world_size} GPUs!")

            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.SGD(parallel_model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0,
                                                                   T_max=args.epochs) if not args.cos else None

            # Setup Logs
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            log_file_base_name = (
                f'{args.data_name}_{args.method}_T{args.T}_opt_{args.opt}_'
                f'epoch{args.epochs}_lr{args.lr}_bs{args.batch_size}_tta{args.tta}')
            if args.TET:
                log_file_base_name += f'_{args.TET_class}_lamb{args.loss_lamb}'
            if args.pml_temperature >= 0:
                log_file_base_name += f'_pml_lamb{args.pml_lamb}_alpha{args.pml_alpha}_temperature{args.pml_temperature}'
            if args.mixup:
                log_file_base_name += "_mixup"
            if args.amp:
                log_file_base_name += "_amp"

            out_dir = os.path.join(args.out_dir, log_file_base_name)

            if is_main_process():
                os.makedirs(out_dir, exist_ok=True)
            if args.distributed:
                dist.barrier()

            seed_acc_dir = os.path.join(out_dir, f"seed_{seed}")
            if is_main_process():
                os.makedirs(seed_acc_dir, exist_ok=True)
            if args.distributed:
                dist.barrier()

            log_path = os.path.join(seed_acc_dir, 'train.log')

            logger = SimpleLogger(log_path)
            csv_path = os.path.join(seed_acc_dir, 'train.csv')

            if is_main_process():
                with open(csv_path, 'w', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    csvwriter.writerow(
                        ['Epoch', 'LR', 'Train Loss', 'Train Acc', 'Test Loss', 'Test Acc', 'Best Acc', 'Best Epoch'])

                if args.Tensorboard:
                    tb_logs_dir = os.path.join(seed_acc_dir, 'logs')
                    os.makedirs(tb_logs_dir, exist_ok=True)
                    writer = SummaryWriter(tb_logs_dir)
                else:
                    writer = None
            else:
                writer = None

            logger.info(f"Starting training with seed: {seed}")
            logger.info(f"Model parameters: {n_parameters}")

            best_acc, best_test_loss, best_epoch, best_model_state = 0, float('inf'), 0, None

            checkpoint_path = os.path.join(seed_acc_dir, 'checkpoint.pth')
            checkpoint = None
            if args.resume and os.path.exists(args.resume):
                checkpoint = torch.load(args.resume, map_location=f"cuda:{args.gpu}")
            elif args.auto_resume and os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{args.gpu}")

            if checkpoint:
                if args.distributed:
                    parallel_model.module.load_state_dict(checkpoint['model'])
                else:
                    parallel_model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                if scheduler and checkpoint['scheduler']:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1
                best_acc = checkpoint.get('best_acc', 0)
                if is_main_process():
                    print(f"Resumed from epoch {checkpoint['epoch'] + 1}")


            for epoch in range(args.start_epoch, args.epochs):
                epoch_start_time = time.time()

                main_seed = seed + epoch
                random.seed(main_seed)
                np.random.seed(main_seed)
                torch.manual_seed(main_seed)
                torch.cuda.manual_seed(main_seed)

                def worker_init_fn(worker_id):
                    worker_seed = seed + worker_id + epoch * args.workers * args.world_size + get_rank() * args.workers
                    random.seed(worker_seed)
                    np.random.seed(worker_seed)
                    torch.manual_seed(worker_seed)
                    torch.cuda.manual_seed(worker_seed)


                if args.distributed:
                    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank,
                                                       shuffle=True, seed=seed)
                    train_sampler.set_epoch(epoch)
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                        num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)
                else:
                    train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn, drop_last=True)

                if args.cos:
                    current_lr = adjust_learning_rate_cos(optimizer, epoch, args)
                else:
                    current_lr = optimizer.param_groups[0]['lr']

                train_loss, train_acc, train_time = train(
                    parallel_model, device, train_loader, criterion, optimizer,
                    epoch, args, logger, writer
                )

                test_loss, test_acc = test(parallel_model, test_loader, device, criterion, epoch, logger, writer)

                if not args.cos and scheduler:
                    scheduler.step()

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_epoch = epoch
                    best_model_state = parallel_model.module.state_dict() if args.distributed else model.state_dict()

                if is_main_process():
                    epoch_time = time.time() - epoch_start_time
                    with open(csv_path, 'a', newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow([epoch + 1, current_lr, train_loss, train_acc, test_loss, test_acc, best_acc,
                                            best_epoch + 1])

                    epoch_line = (f"Epoch: {epoch + 1:03d}/{args.epochs:03d} | LR: {current_lr:.6f} | "
                                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                                  f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
                                  f"Best Acc: {best_acc:.2f}% (Epoch {best_epoch + 1})")
                    print(epoch_line)
                    logger.info(epoch_line)

                checkpoint = {
                    'model': parallel_model.module.state_dict() if args.distributed else parallel_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'args': args
                }
                save_on_master(checkpoint, os.path.join(seed_acc_dir, 'checkpoint.pth'))
                if args.distributed:
                    dist.barrier()

            if is_main_process() and best_model_state:
                save_on_master({'model': best_model_state, 'args': args},
                               os.path.join(seed_acc_dir, 'best_checkpoint.pth'))
                best_acc_list.append(best_acc)
                new_log_path = os.path.join(seed_acc_dir, f"train_{best_acc:.2f}.log")
                if os.path.exists(log_path):
                    os.rename(log_path, new_log_path)
                logger.info(f"Seed {seed} finished. Best Acc: {best_acc:.2f}%")
                if writer: writer.close()

            if args.distributed:
                dist.barrier()

        if is_main_process():
            if len(best_acc_list) > 0:
                acc_array = np.array(best_acc_list)
                avg_acc = np.mean(acc_array)
                std_acc = np.std(acc_array)
            else:
                avg_acc = 0.0
                std_acc = 0.0

            formatted_list = [float(f"{x:.2f}") for x in best_acc_list]

            print("-" * 50)
            print(f"Results across seeds: {formatted_list}")
            print(f"Mean Accuracy: {avg_acc:.2f}%")
            print(f"Standard Deviation: {std_acc:.2f}")
            print("-" * 50)

    except Exception as e:
        if is_main_process():
            print(f"\nFATAL ERROR: {str(e)}", force=True)
            import traceback

            traceback.print_exc()
        raise
    finally:
        if args.distributed and dist.is_initialized():
            dist.destroy_process_group()