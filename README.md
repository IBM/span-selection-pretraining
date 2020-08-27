# span-selection-pretraining
Code to create pre-training data for a span selection pre-training task inspired by reading comprehension and an effort to avoid encoding general knowledge in the transformer network itself.

## Installation
* python setup.py
* build irsimple.jar (or use pre-built com.ibm.research.ai.irsimple/irsimple.jar)
  * cd com.ibm.research.ai.irsimple/
  * mvn clean compile assembly:single
  * (install maven if necessary from https://maven.apache.org/install.html)

## Data Generation
* Download a [Wikipedia dump](https://dumps.wikimedia.org/) and [WikiExtractor](https://github.com/attardi/wikiextractor)
  * IBM is not granting a license to any third-party data set.  You are responsible for complying with all third-party licenses, if any exist.
```bash
python WikiExtractor.py --json --filter_disambig_pages --processes 32 --output wikiextracteddir enwiki-20190801-pages-articles-multistream.xml.bz2
```
* Run create_passages.py (this just splits into passages by double newline)
```bash
python create_passages.py --wikiextracted wikiextracteddir --output wikipassagesdir
```
* Run Lucene indexing
```bash
java -cp irsimple.jar com.ibm.research.ai.irsimple.MakeIndex wikipassagesdir wikipassagesindex
```
* Run sspt_gen.sh
```bash
nohup bash sspt_gen.sh ssptGen wikipassagesdir 2>&1 > querygen.log &
```
* And AsyncWriter
```bash
nohup java -cp irsimple.jar com.ibm.research.ai.irsimple.AsyncWriter \
  ssptGen \
  wikipassagesindex 2>&1 > instgen.log &
```

## Training
**FIXME**: rc_data and span_selection_pretraining require a modified version of [pytorch-transformers](https://github.com/huggingface/pytorch-transformers)
The adaptations needed are in the process of being worked into this repo and a pull request for pytorch-transformers.
Hopefully it is relatively clear how it should work.
```bash
python span_selection_pretraining.py \
  --bert_model bert-base-uncased \
  --train_dir ssptGen \
  --num_instances 1000000 \
  --save_model rc_1M_base.bin

```
