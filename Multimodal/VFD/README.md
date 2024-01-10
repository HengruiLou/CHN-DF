# Released code for VFD
This is the release code for CVPR2022 paper ["Voice-Face Homogeneity Tells Deepfake"](https://arxiv.org/abs/2203.02195).

update 2020.3.25: We have rearranged the code and provided the pretrained model. In addition, a sample test set from FakeAVCeleb is provided.

## Fair Comparison
The critical contribution of this paper is to determine the authenticity of videos cross deepfake datasets via the matching view of voices and faces. Except for the Voxceleb2 (which is difficult to access by now), you can employ any generic visual-audio datasets as training sets and test the model in deepfake datasets. We regard it as a fair comparison.

We applied the Transformer as the feature extractor to process the voice and face input. The ablation experiments show that these extractors will achieve SOTA results. However, we welcome any modifications to the feature extractors for efficiency or scalability as long as a clear statement of the model structure in the paper.

We utilized the [DFDC](https://arxiv.org/abs/2006.07397) and [FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb) datasets as test sets.

## Quick Start

1. Download the pretrained model from [this link](https://drive.google.com/drive/folders/1QN8ZES1dS4wDE9bbpDWY4vHEaPoWnPd6?usp=sharing) and put them in ./checkpoints/VFD

2. Download the sample dataset from [this link](https://drive.google.com/drive/folders/1lCUQvIfAoGKY9SkVzp85POccMhkARyMo?usp=sharing) and unzip it to ./Dataset/FakeAVCeleb

3. Run the test.py

   ```
   python test_DF.py --dataroot ./Dataset/FakeAVCeleb --dataset_mode DFDC --model DFD --no_flip --checkpoints_dir ./checkpoints  --name VFD
   ```
   - **[Baidu Link]**
     
     - Model: [this link](https://pan.baidu.com/s/1yJGWk2ZPSg0Q5cXeAFRbcQ?pwd=41yf ), code: 41yf
     
     - Data: [this link](https://pan.baidu.com/s/1Xf5bXySSAzrD3mWmriCa4g?pwd=d9us), code: d9us 
## Train a New Model

#### Data Preprocess

**Notably:** for DFDC, we have cropped the face region from the origin frames via Dlib toolkit.

For boosting the I/O speed, we have preprocessed the videos and audios in the format shown in ./Dataset.
In paticular, for the videos (take voxceleb2 as example), we extract 1 frame in every video to represent the video. These frames are stored in the 

```
VFD/Dataset/Voxceleb2/face/id_number/video_name/frame_id
```
For example,
```
VFD/Dataset/Voxceleb2/face/id00015/JF-4trZP6fE/00182.jpg
```
For the audios, we extract the Melspectrogram as representation with following code,
```
y, sr = librosa.load(audio, duration=3, sr=16000)
mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=160, n_mels=512)
Mel_out = librosa.power_to_db(mel_spect, ref=np.max)
```
and the Mel_out is stored as .npy,
```
VFD/Dataset/Voxceleb2/voice/**id_number**_**video_name**_**frame_id**.npy
```
For example,
```
VFD/Dataset/Voxceleb2/voice/id00015_3X9uaIs66A0_00022.npy
```
#### Train the Model

After data preprocessing, you can apply the following command to train a new model:

```
python train_DF.py --dataroot ./Dataset/Voxceleb2 --dataset_mode Vox_image --model DFD --no_flip --name experiment_name --serial_batches
```

## Note
If you find this paper is somehow helping, please cite our paper,

```
@inproceedings{VFD2022,
  title={Voice-Face Homogeneity Tells Deepfake},
  author={Harry Cheng, Yangyang Guo, Tianyi Wang, Qi Li, Tao Ye, Liqiang Nie},
  booktitle={{IEEE} Conference on Computer Vision and Pattern Recognition, {CVPR}},
  year={2022}
}
```

Part of the framework is borrowed from  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.

If you have any problem when reading the paper or reproducing the code, please feel free to commit issue or contact us (E-mail: xacheng1996@gmail.com).
