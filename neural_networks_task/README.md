
# Classification of bird chirps

![png-clipart-common-ostrich-cartoon-drawing-graphy-funny-animals-photography-cartoon](https://github.com/jtbandurski/ML-2-Project/assets/80387074/d0d4da20-f92e-45e0-98a0-ae22df5672d7)

## Description
This is an application of Convolutional Neural Networks (EfficientNet and MobileNet) on the binary classification problem. The task is to classify 264 different species of birds. Finally, I've decided to filter out classes for which we did not have more than 5 observations, otherwise stratified sampling could not be performed.
Models operate on mel spectrograms. I've tested only one set of hyperparameters because of the computational constraints and costs connected with renting a GPU. 
	
## How to run
**1. Install requirements**
```bash
pip install -r requirements.txt
```

**2. Run training**
```bash
python main.py --model_name <model name for results saving> --save_results <if we want to save csv with metrics> --save_plots <if we want to save plots with results> --train_with_augmentations <if we want to train with data augmentations>
```

If you want to overfit one batch first and see if the loss function is being minimized, simply run:
```bash
python main.py --overfit_single_batch=True
```

**Note**: I had to prepare augmentations before running code and then I've added them to the corresponding folders of species. You also have to convert audio files to tensors. To do both things, just run the `augmentations_generation.ipynb` from notebooks.


## Glimpse of results

![output](https://github.com/jtbandurski/ML-2-Project/assets/80387074/8b92130d-b8ef-443a-a66c-2e16858cc00c)
