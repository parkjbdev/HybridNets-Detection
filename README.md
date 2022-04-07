## Welcome to HybridNets-Detection

### Configure Conda Environment
```bash
conda env create -n hybridnet --file environment.yaml
```

### Download Weights
```bash
# Download end-to-end weights
mkdir -p hnet/weights
curl -L -o hnet/weights/hybridnets.pth https://github.com/datvuthanh/HybridNets/releases/download/v1.0/hybridnets.pth
```

### Usage
```bash
conda activate hybridnet
python demo.py --source <DESTINITION TO VIDEO FILE>
```