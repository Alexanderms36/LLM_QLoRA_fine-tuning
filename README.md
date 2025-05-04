# Fine-tuning a large language model using QLoRA

---

Fine-tuning Mistral 24B Instruct with custom dataset. The dataset contains 56B tokens.

## Table of Contents
- [File Structure](#file-structure)
- [Dataset](#dataset)
- [Libraries](#libraries)

## File Structure
 - `train.ipynb`: Training proccess jupyter notebook
 - `inference.ipynb`: Testing the fine-tuned model

## Dataset

The data was collected with python parsing using BeautifulSoup4. Only websites which allows to use their data was parsed. Also the training sample contains free distributed datasets about food and health.

Dataset structure:
 - `recipes (43%)`
 - `product compositions (25%)`
 - `articles (18%)`
 - `code (10%)`
 - `product descriptions (3%)`
 - `chemical additions (1%)`

## Libraries
- `transformers 4.51.3`
- `torch 2.7.0`
- `peft 0.15.2`
- `wandb 0.19.10`
- `bitsandbytes 0.45.5`
- `tf_keras 2.19.0`
- `datasets 3.5.1`
- `accelerate 1.6.0`
- `dotenv 1.1.0`