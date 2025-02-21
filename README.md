# AutoDDH

## Graphic Abstract
![image](EGCD-YOLO.png)

## Overview
EGCD-YOLO is an advanced real-time garbage classification detector designed to accurately identify and classify waste with high efficiency. It ensures both real-time processing and precision, making waste detection smarter and more reliable.

## Dependencies
- Python 3.8
- PyTorch 2.4
- numpy
- pandas
- matplotlib

  
## Getting Started
### Data Preparation: Ensure that your dataset is organized in the following structure:
```
data/DDH/
├── train/
│   ├── img/
│   └── seg/
├── val/
│   ├── img/
│   └── seg/
└── test/
    ├── img/
    └── seg/

```
### Training: Run the training script using the provided command-line interface.
### Evaluation: Evaluate the model's performance on the test dataset.


## Usage
```bash
python train.py --trdir <path_to_training_data> --tedir <path_to_testing_data> --valdir <path_to_validation_data> --epochs <number_of_epochs> --lr <learning_rate> --batch-size <batch_size> --ncls <number_of_DDH_degrees> --nseg <number_of_key_structures>
```


