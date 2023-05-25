# Pnemonia x-rays classification

&nbsp;

## Commands

### Activate Miniconda env

```bash
conda activate zoidberg2.0
```

&nbsp;

### Configuring env variables

#### 1. Listing environment variables

```bash
conda env config vars list
```

#### 2. Setting **TF_CPP_MIN_LOG_LEVEL** in order to deactivate *tensorflow* warning logs

```bash
conda env config vars set TF_CPP_MIN_LOG_LEVEL=2
```

#### 3. Reloading the env

```bash
conda activate zoidberg2.0
```

&nbsp;

### Add a library (pip)

```bash
make addPyLib LIB_NAME=my_library
```
