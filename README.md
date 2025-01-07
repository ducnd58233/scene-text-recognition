- export requirements

```bash
pip list --format=freeze > requirements.txt
conda env export > environment.yml --no-builds
```

- start env

```bash
conda activate aio2024-hw
conda install --yes --file requirements.txt
```
