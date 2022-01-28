import torchvision.models as models
import model
import torch
import argparse
import torch.nn as nn

parser = argparse.ArgumentParser(description='PyTorch Transformer-Based model Parameters')
parser.add_argument('--nhid', type=int, default=1024,
                    help='size of word embeddings')
parser.add_argument('--nlayers', type=int, default=24,
                    help='number of layers')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--bptt', type=int, default=128,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--ninp', type=int, default=512,
                    help='size of word embeddings')
parser.add_argument('--nhead', type=int, default=16,
                    help='the number of heads in the encoder/decoder of the transformer model')

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total (Mb)': total_num/(1024*1024), 'Trainable (Mb)': trainable_num/(1024*1024)}

ntokens = 30522

# Only TransformerEncoder involved
# EncoderLayer: 24; FFN hidden Dim: 1024*4; MultiheadAttn: 16
# MultiheadAttn + LayerNorm + FFN + LayerNorm
model_bench = model.TransformerModel(ntokens, 
        args.ninp, 
        args.nhead, 
        args.nhid, 
        args.nlayers, 
        args.dropout).to(device)

model_trans = nn.Transformer(d_model=4096,
        nhead=16,
        num_encoder_layers=24,
        num_decoder_layers=0,
        dim_feedforward=1024,
        dropout=0.5,
        activation="gelu").to(device)
#model_bench = models.alexnet()
model_info = get_parameter_number(model_bench)
print(model_info)
