import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
from scipy import sparse
from src import models
from src.dataset import (
    Central_ID_Bank,
    MarketRunDataset,
    MarketTrainDataset,
    run_batch_collate,
    train_batch_collate,
    load_n_items,
)
from src.utils import (
    get_run_mf,
    get_evaluations_final,
    read_qrel_file,
    write_run_file,
    get_run_mf_index,
)


# fmt: off
def create_arg_parser():
    """Create argument parser for our baseline. """
    parser = argparse.ArgumentParser('VAE')
    
    # DATA  Arguments
    parser.add_argument('--data_dir', help='dataset directory', type=str, default='DATA/')
    parser.add_argument('--tgt_market', help='specify a target market name', type=str, default='t1') 
    parser.add_argument('--src_markets', help='specify none ("") or a few source markets ("-" seperated) to augment the data for training', type=str, default='') 
    parser.add_argument('--use_processed_data', action='store_true', help='in exp')
    
    parser.add_argument('--valid_file', help='specify validation run file for target market', type=str, default='valid_run.tsv')
    parser.add_argument('--test_file', help='specify test run file for target market', type=str, default='test_run.tsv') 
    
    parser.add_argument('--train_file', help='the file name of the train data',type=str, default='train_5core.tsv')
    parser.add_argument('--valid_qrel', help='specify validation run file for target market', type=str, default='valid_qrel.tsv')
    
    # MODEL arguments 
    parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
    parser.add_argument('--wd', type=float, default=0.00,
                        help='weight decay coefficient')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=200,
                        help='upper epoch limit')
    parser.add_argument('--total_anneal_steps', type=int, default=200000,
                        help='the total number of gradient updates for annealing')
    parser.add_argument('--anneal_cap', type=float, default=0.2,
                        help='largest annealing parameter')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    
    return parser
# fmt: on

parser = create_arg_parser()
args = parser.parse_args()
# Set the random seed manually for reproductibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")
Path("checkpoints").mkdir(parents=True, exist_ok=True)


###############################################################################
# Load data
###############################################################################
if args.src_markets == "":
    src_market_list = []
    src_files = None
else:
    src_market_list = args.src_markets.split('-')
    src_files = []
    for src_maket in src_market_list:
        if args.use_processed_data:
            src_files.append(os.path.join(args.data_dir, src_maket, f"train_5core_{args.tgt_market}.tsv"))
        else:
            src_files.append(os.path.join(args.data_dir, src_maket, args.train_file))

print(f"Target market: {args.tgt_market}")
print(f"Source markets: {src_market_list}")
tgt_file = os.path.join(args.data_dir, args.tgt_market, args.train_file)

run_valid_file = os.path.join(args.data_dir, args.tgt_market, args.valid_file)
run_test_file = os.path.join(args.data_dir, args.tgt_market, args.test_file)

id_index_bank = load_n_items(tgt_file, run_valid_file, src_files=src_files)

valid_qrel = read_qrel_file(
    os.path.join(args.data_dir, args.tgt_market, args.valid_qrel), id_index_bank
)

train_data = MarketTrainDataset(tgt_file, id_index_bank, src_files=src_files)
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=False,
    # collate_fn=train_batch_collate,
)

val_data = MarketRunDataset(train_data, run_valid_file, id_index_bank)
val_loader = torch.utils.data.DataLoader(
    val_data, batch_size=args.batch_size, shuffle=False
)

test_data = MarketRunDataset(train_data, run_test_file, id_index_bank,)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=args.batch_size, shuffle=False, 
)

###############################################################################
# Build the model
###############################################################################

n_items = train_data.n_items
update_count = 0

p_dims = [200, 600, n_items]
model = models.MultiVAE(p_dims, conditional=True, num_labels=1+len(src_market_list)).to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
criterion = models.loss_function

###############################################################################
# Training code
###############################################################################
def train():
    # Turn on training mode
    model.train()
    train_loss = 0.0
    global update_count

    for train_data, market_class in train_loader:
        train_data, market_class = train_data.to(device), market_class.to(device)
        if args.total_anneal_steps > 0:
            anneal = min(args.anneal_cap, 1.0 * update_count / args.total_anneal_steps)
        else:
            anneal = args.anneal_cap

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(train_data, c=market_class)

        loss = criterion(recon_batch, train_data, mu, logvar, anneal)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        update_count += 1

    return train_loss


def evaluate():
    # Turn on evaluation mode
    model.eval()
    total_loss = 0.0
    global update_count
    task_unq_users = set()
    valid_rec_all = []

    with torch.no_grad():
        for data_tensor, market_class, user_ids, item_ids in val_loader:
            data_tensor, market_class, user_ids, item_ids = (
                data_tensor.to(device),
                market_class.to(device),
                user_ids.to(device),
                item_ids.to(device),
            )
            if args.total_anneal_steps > 0:
                anneal = min(
                    args.anneal_cap, 1.0 * update_count / args.total_anneal_steps
                )
            else:
                anneal = args.anneal_cap

            recon_batch, mu, logvar = model(data_tensor, market_class)

            loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)
            total_loss += loss.item()

            recon_batch = torch.gather(recon_batch, 1, item_ids)

            task_unq_users = task_unq_users.union(set(user_ids.cpu().numpy()))
            user_ids = user_ids.unsqueeze(1).repeat(1, item_ids.size(1))

            valid_rec_all.append(
                torch.cat(
                    (
                        user_ids.view(-1, 1),
                        item_ids.view(-1, 1),
                        recon_batch.view(-1, 1),
                    ),
                    dim=1,
                )
            )

        valid_rec_all = torch.cat(valid_rec_all, dim=0).cpu().numpy()
        valid_run_mf = get_run_mf_index(valid_rec_all, task_unq_users)
        valid_res, _ = get_evaluations_final(valid_run_mf, valid_qrel)
        return total_loss, valid_res


best_ndcg10 = 0.0
try:
    for epoch in range(1, args.epochs + 1):
        train_loss = train()
        val_loss, ndcg10 = evaluate()
        ndcg10 = ndcg10["ndcg_cut_10"]
        print("-" * 89)
        print(
            f"Epoch {epoch} | train_loss: {train_loss} | val_loss: {val_loss} | ndcg@10: {ndcg10}"
        )
        print("-" * 89)
        exit(0)
        if ndcg10 > best_ndcg10:
            torch.save(model.state_dict(), os.path.join("checkpoints", args.save))
            best_ndcg10 = ndcg10

except KeyboardInterrupt:
    print("-" * 89)
    print("Exiting from training early")

print("=" * 89)
print(f"| End of training | Best ndcg@10: {best_ndcg10}")
print("=" * 89)

checkpoint = torch.load(os.path.join("checkpoints", args.save))
model.load_state_dict(checkpoint)

loaders = [val_loader, test_loader]
with torch.no_grad():
    for i, loader in enumerate(loaders):
        task_unq_users = set()
        rec_all = []
        for (data_tensor, market_class), user_ids, item_ids in val_loader:
            data_tensor, market_class, user_ids, item_ids = (
                data_tensor.to(device),
                market_class.to(device),
                user_ids.to(device),
                item_ids.to(device),
            )
            recon_batch, mu, logvar = model(data_tensor)

            recon_batch = torch.gather(recon_batch, 1, item_ids)

            task_unq_users = task_unq_users.union(set(user_ids.cpu().numpy()))
            user_ids = user_ids.view(-1, 1).repeat(1, item_ids.size(1))

            rec_all.append(
                torch.cat(
                    (
                        user_ids.view(-1, 1),
                        item_ids.view(-1, 1),
                        recon_batch.view(-1, 1),
                    ),
                    dim=1,
                )
            )

        rec_all = torch.cat(rec_all, dim=0).cpu().numpy()
        run_mf = get_run_mf(rec_all, task_unq_users, id_index_bank)
        if i == 0:
            output_file = f"output/{args.tgt_market}/valid_pred.tsv"
        else:
            output_file = f"output/{args.tgt_market}/test_pred.tsv"
        print(f"--Saving file: {output_file}")
        write_run_file(run_mf, output_file)
