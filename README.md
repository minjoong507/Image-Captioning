# Image Captioning

## Intro

- This project is an implementation of the paper ["Show and Tell: A Neural Image Caption Generator"](https://arxiv.org/abs/1411.4555). It may not be completely similar.

- Used Pytorch for the code. ResNet101 is used for extracting the features. You can check pre-trained models [here](https://github.com/pytorch/vision/tree/master/torchvision/models).

- Using [COCO dataset](https://cocodataset.org/#home) 2017 Val images [5K/1GB], annotations [241MB].

- Please check the make_vocab.py and data_loader.py. 
  - **Vocab.pickle** is a pickle file which contains all the words in the annotations. 
  - **coco_ids.npy** stores the image ID to be used. Also, you have to set the path or other settings.

- You can run the source code and try out your own examples. 

## Environment

- Python 3.8.5
- Pytorch 1.7.1
- cuda 11.0

## How to use
- For train

```
cd src
python train.py
```

- For test

```
cd src
python sample.py
```

## TODO List
- [ ] TensorBoard
- [ ] Description of the model and other details
- [ ] Code Refactoring

## License
[MIT License](https://opensource.org/licenses/MIT)

## Reference
[1] [yunjey/pytorch-tutorial](https://github.com/yunjey/pytorch-tutorial)

