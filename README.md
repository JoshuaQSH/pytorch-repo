# Pretrained models test for Pytorch
This repo is for quick deployment and evaluation of different version of CUDA or Pytorch, also can be a helpful tool to reproduce some of the SOTA papers results.
Now added ConvNeXt.

### Test Models
Now includes:
| Name      | Lib-included       | Acc-Err |
| --------- | ------------------ | ------- |
| ResNet-18 | torchvision.models | /       |
| bert-case-uncased | transformers | /     |

### Test Datasets
- CIFAR-10
- ImageNet
- wikitext-2


### Script descriptions
- A CV model timinig evaluation [PASS]
```shell
# [model_name], [device] inside the code
$ python model_timing.py
```

- A CV model througput evaluation [PASS]
```shell
# Throughput Counts
$ python model_throughput.py

# FLOPs and Params Counts
$ python evaluate_models_flops.py 

# Max batch test - default resnet18
$ python model_maxbatch.py
```

- A transformers-based model [WiP]
```shell
# run Script failed
$ ./run.sh

# torch transformer model [PASS]
# Test `model.TransformerModel` and `nn.Transformer`
$ python model_para \
		--nhid 1024 \
		--nlayers 24 \
		--clip 0.25 \
		--epochs 40 \
		--bptt 128 \
		--dropout 0.1 \
		--ninp 512 \
		--nhead 16

```

- `./onnx`: generate/export onnx file and run [PASS]
- `nni_example`: NNI running examples [WiP]
- `./profiler` : Using `torch.profiler.profile` to export the json profiling results. Results in `./profile_results` [PASS]

### Results
- `logs`
- `profile_results`
