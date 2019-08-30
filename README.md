# A pytorch implementation for paper "Semantic Structure-based Unsupervised Deep Hashing" IJCAI-2018

## REQUIREMENTS
1. pytorch 1.1
2. loguru
3. scipy

## DATASETS
1. [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
2. [Flickr25k](https://pan.baidu.com/s/1Bcr5K33l7QFwIRygNxwJ4w) password: ve86
3. [NUS-WIDE](https://pan.baidu.com/s/1f9mKXE2T8XpIq8p7y8Fa6Q) Password: uhr3

## USAGE
```
SSDH_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Dataset name.
  -r ROOT, --root ROOT  Path of dataset
  -c CODE_LENGTH, --code-length CODE_LENGTH
                        Binary hash code length.(default: 12)
  -T MAX_ITER, --max-iter MAX_ITER
                        Number of iterations.(default: 50)
  -l LR, --lr LR        Learning rate.(default: 1e-3)
  -q NUM_QUERY, --num-query NUM_QUERY
                        Number of query data points.(default: 1000)
  -t NUM_TRAIN, --num-train NUM_TRAIN
                        Number of training data points.(default: 5000)
  -w NUM_WORKERS, --num-workers NUM_WORKERS
                        Number of loading data threads.(default: 0)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size.(default: 24)
  -a ARCH, --arch ARCH  CNN architecture.(default: vgg16)
  -k TOPK, --topk TOPK  Calculate map of top k.(default: 5000)
  -v, --verbose         Print log.
  --train               Training mode.
  --resume              Resume mode.
  --evaluate            Evaluate mode.
  -g GPU, --gpu GPU     Using gpu.(default: False)
  -e EVALUATE_INTERVAL, --evaluate-interval EVALUATE_INTERVAL
                        Interval of evaluation.(default: 500)
  -s SNAPSHOT_INTERVAL, --snapshot-interval SNAPSHOT_INTERVAL
                        Interval of evaluation.(default: 800)
  -C CHECKPOINT, --checkpoint CHECKPOINT
                        Path of checkpoint.
  --alpha ALPHA         Hyper-parameter.(default:2)
  --beta BETA           Hyper-parameter.(default:2)
  ```

## EXPERIMENTS
cifar10: 1000 query images, 5000 training images.

nus-wide: Top 10 categories, 5000 query images, 5000 training images.

flickr25k: 2000 query images, 10000 training images.


 | | 16 bits | 32 bits | 64 bits | 128 bits 
   :-:   |  :-:    |   :-:   |   :-:   |   :-:     
cifar-10 MAP@5000 | 0.2511 | 0.2414 | | 
nus-wide MAP@5000 | | | | 
flickr25k MAP@5000 | | | |