# PROJECT_NAME

A template for developing an ML experimentation framework.

## Setup

The first think you should do is run the `post-install.sh` script to configure the name
of the project; this script will change every instance of `PROJECT_NAME` to be
whatever you pass as an argument. In particular, this will change the name of the
project in the `pyproject.toml` file and the name of the associated Conda environment.

```bash
bash scripts/post-install.sh my_project
```

Next, you should figure out what tensor backend you want. By default, the
`pyproject.toml` won't install any, but it provides optional dependency groups for
`torch` and `flax`, as well as a group for `huggingface`. You can either keep these
all as optional dependencies and just decide which to use, or move the appropriate
block into the main dependencies section.


## Local Installation

All python dependencies are provided in `pyproject.toml`. Install using `uv`:

1. `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. `uv venv`
3. `source .venv/bin/activate`
4. `uv pip install -e .`

To generate a set of locked dependencies, run

```bash
uv pip compile pyproject.toml -o requirements.txt
```

If you need to use Conda instead, you can do so by creating a new environment from
the provided `environment.yml` file, which will just wrap the `pyproject.toml` file with
pip:

```bash
conda env create -f environment.yml
```


## Docker / Devcontainer

There's a built-in Dockerfile and devcontainer configuration to make running
the project in a remote container from VSCode easy. Just install the remote containers
extension and open the project in a container.
