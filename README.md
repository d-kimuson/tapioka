# tapioka

## 必要ツール

- [Yarn](https://yarnpkg.com/getting-started/install)
  - poetry はタスクランナーすら定義できないのと、開発ツールチェーン周りは Node が最強なので yarn 使うよ
- [Poetry](https://cocoatomo.github.io/poetry-ja/)
  - Python 向けパッケージ管理
- [pyenv](https://github.com/pyenv/pyenv) or `python 3.8.13`
  - MUST ではないけど推奨

リンクを参考にインストールしてください

```bash
$ yarn -v
1.22.15
$ poetry -V
Poetry version 1.1.13
$ pyenv -v
pyenv 2.2.5
```

## 環境構築

```bash
$ pyenv install 3.8.13
$ yarn install
```

## python スクリプトを実行する

```bash
$ poetry run python src/main.py  # 普通に poetry run するか
$ yarn r src/main.py             # 長いから yarn r 使う
yarn run v1.22.15
$ poetry run python src/main.py
Hello World!
✨  Done in 0.77s.
```

## 外部パッケージを追加する

パッケージ管理は poetry でしてるので、必要になったパッケージは poetry で追加してね
開発向けは `-D` だよ

```bash
$ poetry add torch
$ poetry add -D black
```
