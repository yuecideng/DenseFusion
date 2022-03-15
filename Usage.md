#Usage of Modified Version

The modification(currently) :
1. Use a new `dataset` with unified implementation
2. Replace original `knn` with `sklearn` KDTree in `loss.py` and `loss_refiner.py`
3. Support cpu training and evaluation for debug usage
4. (TODO) Support evaluation  

## How to Train
1. Prepare `BOP` format dataset.
2. In root dir of `Densefusion`:
```
python3 tools/train.py --dataset lm --dataset_root /path/to/dataset
```