### TPU VM and Environment setup for DALL-E Training

## Create Pytorch TPU VM on GCP.
- Follow the steps for Pytorch TPU VM creation and connection [here](https://cloud.google.com/tpu/docs/pytorch-quickstart-tpu-vm). (Optionally, can create the TPU VM using the cloud console interface. Select tpu-vm-pt-1.10 as the architecture while creating)
- Note: Do not forget to set [XRT TPU device configuration](https://cloud.google.com/tpu/docs/pytorch-quickstart-tpu-vm#set_xrt_tpu_device_configuration) as mentioned.
- To add additional disk space, follow : https://cloud.google.com/tpu/docs/setup-persistent-disk
- Install other required packages
```shell
pip install -r requirements.txt
```
### MS COCO data download
- The following link can be used, https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9

### Download Pre-trained checkpoint
- You can download the pretrained models including the tokenizer from this [link](https://arena.kakaocdn.net/brainrepo/models/minDALL-E/57b008f02ceaa02b779c8b7463143315/1.3B.tar.gz). This will require about 5GB space.
- Move the config.yaml file from configs to the extracted model directory from the above link.

### Fine-tune the checkpoint on MS-COCO

```shell
python3 examples/finetune_lightning.py
```
