# Katsetused

## installimine

```bash
cd  ~/git/huggingface_github/huggingface_tokenizers_github_fork/tv
python3 -m venv venv
venv/bin/pip3 install -e ~/git/huggingface_github/huggingface_tokenizers_github_fork/bindings/python
```

## kopeerime näidisprogrammi

```bash
cp ../bindings/python/examples/train_bert_wordpiece.py .
```

## jooksutame näidisprogrammmi

```bash
venv/bin/python3 train_bert_wordpiece.py --files='dataset_in/*txt' --out=vocab_out/
```
