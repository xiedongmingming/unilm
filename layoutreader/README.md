# layoutreader

layoutreader captures the text and layout information for reading order prediction using the seq2seq model. it significantly improves both open-source and commercial ocr engines in ordering text lines in their results in our experiments.


our paper "[layoutreader: pre-training of text and layout for reading order detection](https://arxiv.org/pdf/2108.11591.pdf)" has been accepted by emnlp 2021.

**readingbank** is a benchmark dataset for reading order detection built with weak supervision from word documents, which contains 500K document images with a wide range of document types as well as the corresponding reading order information. for more details, please refer to [readingbank](https://aka.ms/readingbank).

## installation
~~~
conda create -n LayoutReader python=3.7
conda activate LayoutReader
conda install pytorch==1.7.1 -c pytorch

pip install nltk

python -c "import nltk; nltk.download('punkt')"

git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext
pip install transformers==2.10.0

git clone https://github.com/microsoft/unilm.git

cd unilm/layoutreader

pip install -e .
~~~

## run
1. download the pre-processed data ([`readingbank.zip`](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/Efh3ZWjsA-xFrH2FSjyhSVoBMak6ypmbABWmJEmPwtKhhw?e=tbthMD)). for more details of the dataset, please refer to [readingbank](https://aka.ms/readingbank).
2. (optional) download our pre-trained model ([`layoutreader-base-readingbank.zip`](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ET9XynvgSZFLhPy7p30zbtoBs-T_Yxj6gl_k-b2-N53ChQ?e=gKafBy)) and evaluate it refer to step 4.
3. training
    ~~~
    export CUDA_VISIBLE_DEVICE=0,1,2,3
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    
    python -m torch.distributed.launch --nproc_per_node=4 run_seq2seq.py \
        --model_type layoutlm \
        --model_name_or_path layoutlm-base-uncased \
        --train_folder /path/to/ReadingBank/train \
        --output_dir /path/to/output/LayoutReader/layoutlm \
        --do_lower_case \
        --fp16 \
        --fp16_opt_level O2 \
        --max_source_seq_length 513 \
        --max_target_seq_length 511 \
        --per_gpu_train_batch_size 2 \
        --gradient_accumulation_steps 1 \
        --learning_rate 7e-5 \
        --num_warmup_steps 500 \
        --num_training_steps 75000 \
        --cache_dir /path/to/output/LayoutReader/cache \
        --label_smoothing 0.1 \
        --save_steps 5000 \
        --cached_train_features_file /path/to/ReadingBank/features_train.pt
    ~~~
4. decoding
    ~~~
    export CUDA_VISIBLE_DEVICES=0
    export OMP_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    
    python decode_seq2seq.py --fp16 \
        --model_type layoutlm \
        --tokenizer_name bert-base-uncased \
        --input_folder /path/to/ReadingBank/test \
        --cached_feature_file /path/to/ReadingBank/features_test.pt \
        --output_file /path/to/output/LayoutReader/layoutlm/output.txt \
        --split test \
        --do_lower_case \
        --model_path /path/to/output/LayoutReader/layoutlm/ckpt-75000 \
        --cache_dir /path/to/output/LayoutReader/cache \
        --max_seq_length 1024 \
        --max_tgt_length 511 \
        --batch_size 32 \
        --beam_size 1 \
        --length_penalty 0 \
        --forbid_duplicate_ngrams \
        --mode s2s \
        --forbid_ignore_word "."
    ~~~

## results
our released [pre-trained model](https://mail2sysueducn-my.sharepoint.com/:u:/g/personal/huangyp28_mail2_sysu_edu_cn/ET9XynvgSZFLhPy7p30zbtoBs-T_Yxj6gl_k-b2-N53ChQ?e=gKafBy) achieves 98.2% average page-level bleu score. detailed results are reported as follow:

* evaluation results of the layoutreader on the reading order detection task, where the source-side of training/testing data is in the left-to-right and top-to-bottom order

  | Method                     | Encoder                | Avg. Page-level BLEU ↑ | ARD ↓ |
  | -------------------------- | ---------------------- | ---------------------- | ----- |
  | Heuristic Method           | -                      | 0.6972                 | 8.46  |
  | LayoutReader (text only)   | BERT                   | 0.8510                 | 12.08 |
  | LayoutReader (text only)   | UniLM                  | 0.8765                 | 10.65 |
  | LayoutReader (layout only) | LayoutLM (layout only) | 0.9732                 | 2.31  |
  | LayoutReader               | LayoutLM               | 0.9819                 | 1.75  |

* input order study with left-to-right and top-to-bottom inputs in evaluation, where r is the proportion of shuffled samples in training.

  | Method                          | Avg. Page-level BLEU ↑ | Avg. Page-level BLEU ↑ | Avg. Page-level BLEU ↑ | ARD ↓  | ARD ↓ | ARD ↓ |
  |---------------------------------|------------------------|------------------------|------------------------|--------|-------|-------|
  |                                 | r=100%                 | r=50%                  | r=0%                   | r=100% | r=50% | r=0%  |
  | LayoutReader (text only, BERT)  | 0.3355                 | 0.8397                 | 0.8510                 | 77.97  | 15.62 | 12.08 |
  | LayoutReader (text only, UniLM) | 0.3440                 | 0.8588                 | 0.8765                 | 78.67  | 13.65 | 10.65 |
  | LayoutReader (layout only)      | 0.9701                 | 0.9729                 | 0.9732                 | 2.85   | 2.61  | 2.31  |
  | LayoutReader                    | 0.9765                 | 0.9788                 | 0.9819                 | 2.50   | 2.24  | 1.75  |
  
* input order study with token-shuffled inputs in evaluation, where r is the proportion of shuffled samples in training.

  | Method                          | Avg. Page-level BLEU ↑ | Avg. Page-level BLEU ↑ | Avg. Page-level BLEU ↑ | ARD ↓  | ARD ↓ | ARD ↓  |
  |---------------------------------|------------------------|------------------------|------------------------|--------|-------|--------|
  |                                 | r=100%                 | r=50%                  | r=0%                   | r=100% | r=50% | r=0%   |
  | LayoutReader (text only, BERT)  | 0.3085                 | 0.2730                 | 0.1711                 | 78.69  | 85.44 | 67.96  |
  | LayoutReader (text only, UniLM) | 0.3119                 | 0.2855                 | 0.1728                 | 80.00  | 85.60 | 71.13  |
  | LayoutReader (layout only)      | 0.9718                 | 0.9714                 | 0.1331                 | 2.72   | 2.82  | 105.40 |
  | LayoutReader                    | 0.9772                 | 0.9770                 | 0.1783                 | 2.48   | 2.46  | 72.94  |

## citation

if you find layoutreader helpful, please cite us:
```
@misc{wang2021layoutreader,
      title={LayoutReader: Pre-training of Text and Layout for Reading Order Detection}, 
      author={Zilong Wang and Yiheng Xu and Lei Cui and Jingbo Shang and Furu Wei},
      year={2021},
      eprint={2108.11591},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


## license

this project is licensed under the license found in the license file in the root directory of this source tree. portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) and [s2s-ft](../s2s-ft) projects.
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)

## contact

for help or issues using layoutreader, please submit a github issue.

for other communications related to layoutlm, please contact lei cui (`lecu@microsoft.com`), furu wei (`fuwei@microsoft.com`).
