
# Minimal TF-IDF baseline models. Train, infer, eval, build Docker container

## Development

Fetch public Russian SuperGLUE data.

```bash
wget https://russiansuperglue.com/tasks/download -O combined.zip
unzip combined.zip
rm combined.zip
mkdir data
mv combined data/tasks
```

Fetch TF-IDF vectorizer weights.

```bash
wget https://russiansuperglue.com/tasks/tf_idf -O tfidf.pkl.zip
unzip tfidf.pkl.zip
rm tfidf.pkl.zip
mv tfidf.pkl data
```

Install deps.

```bash
pip install \
  scikit-learn \
  joblib
```

Train classifiers. No pretrain for MuSeRC and RuCos.

```bash
declare -A titles
titles[danetqa]=DaNetQA
titles[lidirus]=LiDiRus
titles[muserc]=MuSeRC
titles[parus]=PARus
titles[rcb]=RCB
titles[rucos]=RuCoS
titles[russe]=RUSSE
titles[rwsd]=RWSD
titles[terra]=TERRa

mkdir -p data/classifiers
for task in danetqa parus rcb russe rwsd terra
do
  cat data/tasks/${titles[$task]}/train.jsonl | \
    python main.py train $task data/tfidf.pkl data/classifiers/$task.pkl
done
```

Infer. Use TERRa classifiers for LiDiRus. Omit classifier weights for MuSeRC and RuCos.

```bash
mkdir -p data/infer
for task in danetqa parus rcb russe rwsd terra
do
  python main.py infer $task data/tfidf.pkl data/classifiers/$task.pkl \
    < data/tasks/${titles[$task]}/val.jsonl \
    > data/infer/$task.jsonl
done

python main.py infer lidirus data/tfidf.pkl data/classifiers/terra.pkl \
  < data/tasks/LiDiRus/LiDiRus.jsonl \
  > data/infer/lidirus.jsonl

python main.py infer rucos data/tfidf.pkl \
  < data/tasks/RuCoS/val.jsonl \
  > data/infer/rucos.jsonl

python main.py infer muserc data/tfidf.pkl \
  < data/tasks/MuSeRC/val.jsonl \
  > data/infer/muserc.jsonl
```
