# 💡 Light Bulb

Light Bulb is an labeling tool built with state of the art active learning and semi supervised learning techniques. Currently supports text classification and image classification.

See the Medium post [here](https://medium.com/@chriszhu12/light-bulb-machine-learning-made-easy-43e64f5124bd).

## Table of Contents

- [Getting Started](https://github.com/czhu12/labelling-tool#getting-started)
  - [Mac OSX](https://github.com/czhu12/labelling-tool#mac-osx)
- [Usage](https://github.com/czhu12/labelling-tool#usage)
  - [Configuration](https://github.com/czhu12/labelling-tool#configuration)
  - [Example Text Classification](https://github.com/czhu12/labelling-tool#example-text-classification)
  - [Example Image Classification](https://github.com/czhu12/labelling-tool#example-image-classification)
- [How It Works](https://github.com/czhu12/labelling-tool#how-it-works)
  - [Semi Supervised Text](https://github.com/czhu12/labelling-tool#semi-supervised-text)
  - [Semi Supervised Image](https://github.com/czhu12/labelling-tool#semi-supervised-image)
  - [Active Learning](https://github.com/czhu12/labelling-tool#active-learning)
- [Coming Soon](https://github.com/czhu12/labelling-tool#coming-soon)

## Getting Started
#### Mac OSX
```
brew install yarn
git clone https://github.com/czhu12/labelling-tool && cd labelling-tool
make
```
#### Cat / not cat demo dataset
```
make dataset/cat_not_cat # Download and set up dataset.
python code/server.py --config config/cat_not_cat.yml # Server set up on localhost:5000
```

## Usage
### Configuration
Heres an example configuration:
```
task:
  title: What kind of animal is this?
  description: Select the type of animal you see, if there is none, select "Skip"
dataset:
  directory: dataset/image_classification/
  data_type: images
  judgements_file: outputs/image_multiclass_classification/labels.csv
label:
  type: classification
  classes:
    - Dog
    - Cat
    - Giraffe
    - Dolphin
    - Skip
model:
  directory: outputs/image_multiclass_classification/models/
user: chris
```

#### task
```
task:
  title: What kind of animal is this?
  description: Select the type of animal you see, if there is none, select "Skip"
```

#### dataset
```
dataset:
  directory: dataset/image_classification/
  data_type: images
  judgements_file: outputs/image_multiclass_classification/labels.csv
```
`judgements_file` defines the file that the labels are saved in.

`data_type` defines what type of model is used. Valid options are `images` and `text`

#### label
```
label:
  type: classification
  classes:
    - Dog
    - Cat
    - Giraffe
    - Dolphin
```

`type` defines the type of label, options are `classification` and `binary`.

#### model
```
model:
  directory: outputs/image_multiclass_classification/models/
```

`directory` defines where the trained model is saved.

#### user
```
user: chris
```
`user` defines who the person labeling is, which may be useful when the label's are used.

### Example Text classification
To run the text classification demo:
```
make run config/text_multiclass_classification.yml
```

### Example Image Classification
To run the Image classification demo:
```
make run config/image_multiclass_classification.yml
```

## How It Works
### Architecture
![Encoder Decoder](https://raw.githubusercontent.com/czhu12/labelling-tool/master/docs/images/encoder-decoder.png)

Most deep learning tasks can be framed as a encoder - decoder architecture. For example, text classification can be framed as an LSTM encoder that outputs into a logistic regression decoder. Object detection can be framed as a ResNet encoder with a regression decoder. All models in Light Bulb are framed as an encoder - decoder architecture, and the encoder are pre-trained on an external dataset (Image Net for images, and Wikitext-103 for text), and then fine-tuned on the target dataset.

### Semi Supervised Text
![Encoder Decoder](https://raw.githubusercontent.com/czhu12/labelling-tool/master/docs/images/language-model.png)

Light Bulb's text encoder is a pretrained language model on [wikitext-103](https://einstein.ai/research/the-wikitext-long-term-dependency-language-modeling-dataset) (inspired by [ULMFiT](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)),  with a vocab limited to the most frequent 100k words in the corpus. The model is fine-tuned on the target dataset as a language model.

### Semi Supervised Image
![Squeeze Net](https://raw.githubusercontent.com/czhu12/labelling-tool/master/docs/images/squeezenet-architecture.png)

Light Bulb uses [Squeeze Net](https://github.com/DeepScale/SqueezeNet) pretrained on the ImageNet dataset to encode image data. The encoder is fine-tuned on the target dataset that is given to be labeled as an auto-encoder. Standard image augmentation techniques are used to expand the labeled training set.

### Active Learning
Light Bulb will train a model as you provide training data through labeling. Light Bulb will sample items to be labeled by scoring the unlabeled items and sample the highest entropy items.

## Coming Soon
- Sequence Tagging
- Object Detection
- Sequence to Sequence Modeling
- Dockerize Application
