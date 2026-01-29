# pyvalues Development Notes

```shell
poetry install --with development
```

## Running unittests (automatically on push)

```shell
poetry run python -m unittest
```

## Running linter (automatically on push)

```shell
poetry run flake8 src --count --max-complexity=10 --max-line-length=127 --statistics --ignore=C901
```

## Release new version

- Change `version` in [`pyproject.toml`](../pyproject.toml)
- Add a release via [Github web interface](https://github.com/ValueEval/pyvalues/releases/new), tagged `v<VERSION>`
