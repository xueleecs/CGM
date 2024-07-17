# CGM

#Prepare work
git clone

conda create -n CGM python==3.7.12

pip install -r requirements.txt

# 1 FOR DATASET
unzip fasta.tar.gz in fasta file
# 2 FOR PROTEIN Embedding 
Firstly, you need to download pytorch_model.bin file from the following URL https://huggingface.co/Rostlab/prot_bert_bfd/blob/main/pytorch_model.bin. And put pytorch_model.bin file into prot_bert_bfd directory.

# 3 RUN THE CODE
python CGM.py

# 4 OTHER PROBLEM
if you have other problem, please contact xueleecs@gmail.com
