# VAE for XMRec competition

## Install dependencies
```
pip install pytorch
pip install pytrec-eval
```

## Train VAE model
Example of train script using 3 source markets s1, s2, s3 and target market t2 for 200 epochs
```
python main.py \
    --src_markets s1-s2-s3 \  # source markets for training
    --tgt_market t2 \         # target market
    --save model_t2.pt \      # model checkpoint name
    --cuda \                  # using cuda
    --conditional \           # conditional VAE
    --epochs 200 \            # num epochs
```
