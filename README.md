# CGM

#Prepare work
git clone

conda create -n CGM python==3.7.12

pip install -r requirements.txt

# 1 For protein embeding 
Firstly, you need to download pytorch_model.bin file from the following URL https://huggingface.co/Rostlab/prot_bert_bfd/blob/main/pytorch_model.bin. And put pytorch_model.bin file into prot_bert_bfd directory.

# 2 run the code
python CGM.py

# 3 other problem
if you have other problem, please contact xueleecs@gmail.com
