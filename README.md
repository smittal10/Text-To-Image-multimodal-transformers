# Text-To-Image-multimodal-transformers
Code for all the experiments performed as part of Directed Study.

### Installation for CogView

```shell
sudo apt-get -y install llvm-9-dev cmake
# pip install googletrans==3.1.0a0

git clone https://github.com/microsoft/DeepSpeed.git Deepspeed
cd Deepspeed
DS_BUILD_SPARSE_ATTN=1 ./install.sh -r

git clone https://github.com/NVIDIA/apex.git apex
export CUDA_HOME=/usr/local/cuda-11.1/ && cd /tmp/apex && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

cd CogView
pip install -r requirements.txt
cd ..
wget "https://www.dropbox.com/s/lk78djywfdw9na1/vqvae_hard_biggerset_011.pt?dl=1" -O CogView/pretrained/vqvae/vqvae_hard_biggerset_011.pt
```
---
### Model and Data Download
```shell

wget "https://the-eye.eu/public/AI/CogView/cogview-base.tar" -O cogview-base.tar
tar -xvf cogview-base.tar -C CogView/pretrained/cogview/

wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip /content/annotations_trainval2014.zip
```
---
### Run CogView Inference
```shell
cd CogView
bash CogView/scripts/text2image.sh --input-source="../annotations/chinese_captions.txt" --batch-size=$images_to_generate --max-inference-batch-size=8 --device 0 --debug
```