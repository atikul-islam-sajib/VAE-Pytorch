# VAE: Variational Autoencoder 

<img src="https://github.com/atikul-islam-sajib/VAE-Pytorch/blob/main/artifacts/outputs/test_image/test_result.png" alt="Context Encoder GAN">

In machine learning, a variational autoencoder is an artificial neural network architecture introduced by Diederik P. Kingma and Max Welling. It belongs to the family of probabilistic graphical models and variational Bayesian methods.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*r1R0cxCnErWgE0P4Q-hI0Q.jpeg" alt="AC-GAN - Medical Image Dataset Generator: Generated Image with labels">

## Features

| Feature                          | Description                                                                                                                                                                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Efficient Implementation**     | Utilizes an optimized VAE model architecture for superior performance on diverse image segmentation tasks.                                                                                                          |
| **Custom Dataset Support**       | Features easy-to-use data loading utilities that seamlessly accommodate custom datasets, requiring minimal configuration.                                                                                             |
| **Training and Testing Scripts** | Provides streamlined scripts for both training and testing phases, simplifying the end-to-end workflow.                                                                                                               |
| **Visualization Tools**          | Equipped with tools for tracking training progress and visualizing segmentation outcomes, enabling clear insight into model effectiveness.                                                                            |
| **Custom Training via CLI**      | Offers a versatile command-line interface for personalized training configurations, enhancing flexibility in model training.                                                                                          |
| **Import Modules**               | Supports straightforward integration into various projects or workflows with well-documented Python modules, simplifying the adoption of Context Encoder functionality.                                                         |
| **Multi-Platform Support**       | Guarantees compatibility with various computational backends, including MPS for GPU acceleration on Apple devices, CPU, and CUDA for Nvidia GPU acceleration, ensuring adaptability across different hardware setups. |

## Getting Started

## Installation Instructions

Follow these steps to get the project set up on your local machine:

| Step | Instruction                                  | Command                                                       |
| ---- | -------------------------------------------- | ------------------------------------------------------------- |
| 1    | Clone this repository to your local machine. | **git clone https://github.com/atikul-islam-sajib/VAE-Pytorch.git** |
| 2    | Navigate into the project directory.         | **cd VAE-Pytorch**                                                  |
| 3    | Install the required Python packages.        | **pip install -r requirements.txt**                           |

## Project Structure
```
    .
    ├── Dockerfile
    ├── LICENSE
    ├── README.md
    ├── VAE_Pytorch.egg-info/
    │   ├── PKG-INFO
    │   ├── SOURCES.txt
    │   ├── dependency_links.txt
    │   ├── requires.txt
    │   └── top_level.txt
    ├── artifacts/
    │   ├── checkpoints/
    │   ├── files/
    │   ├── metrics/
    │   └── outputs/
    ├── config.yml
    ├── data/
    │   ├── processed/
    │   └── raw/
    ├── dvc.lock
    ├── dvc.yaml
    ├── logs/
    ├── mlruns/
    ├── mypy.ini
    ├── requirements.txt
    ├── research/
    │   ├── files/
    │   └── notebooks/
    ├── setup.py
    ├── src/
    │   ├── VAE.py
    │   ├── __init__.py
    │   ├── cli.py
    │   ├── dataloader.py
    │   ├── decoder.py
    │   ├── encoder.py
    │   ├── helper.py
    │   ├── kl_divergence.py
    │   ├── mse.py
    │   ├── tester.py
    │   ├── trainer.py
    │   └── utils.py
    └── unittest/
        └── test.py

```


### Dataset Organization for VAE

The dataset is organized into three categories for VAE. Each category directly contains paired images and their corresponding images, stored together to simplify the association images .

## Directory Structure:

```
dataset/
├── X/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
├── y/
│ │ ├── 1.png
│ │ ├── 2.png
│ │ ├── ...
```

### User Guide Notebook - CLI

For detailed documentation on the implementation and usage, visit the -> [VAE Notebook - CLI](https://github.com/atikul-islam-sajib/VAE-Pytorch/blob/main/research/notebooks/ModelTrain-CLI.ipynb).

### User Guide Notebook - Custom Modules

For detailed documentation on the implementation and usage, visit the -> [VAE Notebook - CM](https://github.com/atikul-islam-sajib/VAE-Pytorch/blob/main/research/notebooks/ModelTrain-CM.ipynb).

## Data Versioning with DVC
To ensure you have the correct version of the dataset and model artifacts.

Reproducing the Pipeline
To reproduce the entire pipeline and ensure consistency, use:

```bash
dvc repro
```

### Command Line Interface

The project is controlled via a command line interface (CLI) which allows for running different operational modes such as training, testing, and inference.

#### CLI Arguments
| Argument          | Description                                  | Type   | Default |
|-------------------|----------------------------------------------|--------|---------|
| `--image_path`    | Path to the image dataset                    | str    | None    |
| `--batch_size`    | Number of images per batch                   | int    | 1       |
| `--image_size`    | Size to resize images to                     | int    | 64      |
| `--epochs`        | Number of training epochs                    | int    | 100     |
| `--lr`            | Learning rate                                | float  | 0.0002  |
| `--lr_scheduler`| Enable learning rate scheduler              | bool   | False   |
| `--is_weight_init`| Apply weight initialization                  | bool   | False   |
| `--device`        | Computation device ('cuda', 'mps', 'cpu')    | str    | 'mps'   |
| `--adam`          | Use Adam optimizer                           | bool   | True    |
| `--SGD`           | Use Stochastic Gradient Descent optimizer    | bool   | False   |
| `--beta1`         | Beta1 parameter for Adam optimizer           | float  | 0.5     |
| `--train`         | Flag to initiate training mode               | action | N/A     |
| `--model`         | Path to a saved model for testing            | str    | None    |
| `--test`          | Flag to initiate testing mode                | action | N/A     |

### CLI Command Examples

Here is the table with the MPS commands removed:

| Task                     | CUDA Command                                                                                                              | CPU Command                                                                                                              |
|--------------------------|---------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| **Training a Model**     | `python cli.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --adam True --is_l1 True --device "cuda"` | `python cli.py --train --image_path "/path/to/dataset" --batch_size 32 --image_size 128 --epochs 50 --lr 0.001 --adam True --is_l1 True --device "cpu"` |
| **Testing a Model**      | `python cli.py --test --model "/path/to/saved_model.pth" --device "cuda"`                                              | `python cli.py --test --model "/path/to/saved_model.pth" --device "cpu"`                                              |

### Notes:
- **CUDA Command**: For systems with NVIDIA GPUs, using the `cuda` device will leverage GPU acceleration.
- **CPU Command**: Suitable for systems without dedicated GPU support or for testing purposes on any machine.

**Configure the Project**:
   Update the `config.yml` file with the appropriate paths and settings. An example `config.yml`:
   ```yaml

    path:
        RAW_DATA_PATH: "./data/raw/"
        PROCESSED_DATA_PATH: "./data/processed/"
        FILES_PATH: "./artifacts/files/"
        TRAIN_MODELS: "./artifacts/checkpoints/train_models/"
        TEST_MODELS: "./artifacts/checkpoints/best_model/"
        TRAIN_IMAGES_PATH: "./artifacts/outputs/train_images/"
        VALID_IMAGES_PATH: "./artifacts/outputs/test_image/"
        TRAIN_HISTORY_PATH: "./artifacts/metrics/"

    dataloader:
        image_path: "./data/raw/dataset.zip"
        channels: 3
        image_size: 256
        batch_size: 4
        split_size: 0.30

    VAE:
        channels: 3
        image_size: 256

    trainer:
        epochs: 3000   
        lr: 0.002
        weight_decay: 0.0001
        beta1: 0.5
        beta2: 0.999
        momentum: 0.95
        step_size: 10
        gamma: 0.85
        adam: True
        SGD: False
        device: "mps"
        verbose: True
        lr_scheduler: False
        weight_init: False
        l1_regularization: False
        l2_regularization: False
        MLFlow: True

    tester:
        model: "best" 
        device: "mps"

```


#### Initializing Data Loader - Custom Modules
```python
loader = Loader(image_path="path/to/dataset", batch_size=32, image_size=128)
loader.unzip_folder()
loader.create_dataloader()
```

##### To details about dataset
```python
loader.plot_images()           # It will display the images from dataset
```

#### Training the Model
```python
trainer = Trainer(
    epochs=100,                # Number of epochs to train the model
    lr=0.0002,                 # Learning rate for optimizer
    device='cuda',             # Computation device ('cuda', 'mps', 'cpu')
    adam=True,                 # Use Adam optimizer; set to False to use SGD if implemented
    SGD=False,                 # Use Stochastic Gradient Descent optimizer; typically False if Adam is True
    beta1=0.5,                 # Beta1 parameter for Adam optimizer
    lr_scheduler=False,        # Enable a learning rate scheduler
    weight_init=False,         # Enable custom weight initialization for the models
    display=True               # Display training progress and statistics

    ... ... ... 
    ... ... ...                # Check the trainer.py
)

# Start training
trainer.train()
```

#### Testing the Model
```python
tester = Tester(device="cuda", model="best") # use mps, cpu
test.test()
```


### Configuration for MLFlow

1. **Generate a Personal Access Token on DagsHub**:
   - Log in to [DagsHub](https://dagshub.com).
   - Go to your user settings and generate a new personal access token under "Personal Access Tokens".

2. **Set environment variables**:
   Set the following environment variables with your DagsHub credentials:
   ```bash
   export MLFLOW_TRACKING_URI="https://dagshub.com/<username>/<repo_name>.mlflow"
   export MLFLOW_TRACKING_USERNAME="<your_dagshub_username>"
   export MLFLOW_TRACKING_PASSWORD="<your_dagshub_token>"
   ```

   Replace `<username>`, `<repo_name>`, `<your_dagshub_username>`, and `<your_dagshub_token>` with your actual DagsHub username, repository name, and personal access token.

### Running the Training Script

To start training and logging the experiments to DagsHub, run the following command:

```bash
python src/cli.py --train 
python src/cli.py --test 

```

### Accessing Experiment Tracking

You can access the MLflow experiment tracking UI hosted on DagsHub using the following link:

[VAE Experiment Tracking on DagsHub](https://dagshub.com/atikul-islam-sajib/VAE-Pytorch/experiments)

### Using MLflow UI Locally

If you prefer to run the MLflow UI locally, use the following command:

```bash
mlflow ui
```


## Contributing
Contributions to improve this implementation of VAE are welcome. Please follow the standard fork-branch-pull request workflow.

## License
Specify the license under which the project is made available (e.g., MIT License).
