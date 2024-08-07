# layoutlmft
**multimodal (text + layout/format + image) fine-tuning toolkit for document understanding**

## introduction

## supported models
popular language models: bert, unilm(v2), roberta, infoxlm

layoutlm family: layoutlm, layoutlmv2, layoutxlm

## installation

~~~bash
conda create -n layoutlmft python=3.7
conda activate layoutlmft

git clone https://github.com/microsoft/unilm.git

cd unilm
cd layoutlmft

pip install -r requirements.txt
pip install -e .
~~~

## license

the content of this project itself is licensed under the [attribution-noncommercial-sharealike 4.0 international (cc by-nc-sa 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
portions of the source code are based on the [transformers](https://github.com/huggingface/transformers) project.
[microsoft open source code of conduct](https://opensource.microsoft.com/codeofconduct)

### contact Information

for help or issues using layoutlmft, please submit a github issue.

for other communications related to layoutlmft, please contact lei gui (`lecu@microsoft.com`), furu wei (`fuwei@microsoft.com`).

