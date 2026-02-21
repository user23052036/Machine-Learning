By default, NLTK downloads to:

```
~/nltk_data
```

If you want to change the download location, specify `download_dir`.

---

## Option 1 — Download to Custom Folder

```python
import nltk
nltk.download('stopwords', download_dir='./nltk_data')
```

This will create:

```
./nltk_data/corpora/stopwords/
```

---

## Option 2 — Tell NLTK Where to Look

After downloading to custom path, add:

```python
import nltk
nltk.data.path.append('./nltk_data')
```

Now NLTK will search there.

---

## Option 3 — Set Permanent Environment Variable (Cleanest)

In Linux (since you use Parrot OS):

```bash
export NLTK_DATA=/home/midori/my_custom_nltk_data
```

Add it to `.bashrc` for persistence.

---

## Engineering Advice (For You)

If this is for your ML project repo:

* Create a local `nltk_data/` folder inside project
* Download there
* Append path in code
* Document in README

This prevents deployment failures.

---
