# 関数同定基盤

データから候補式を探索し、同じ入力・分割・指標で比較して、式・PNG・CSVを保存するPythonパッケージです。

中心は **PySR / ParFam / TorchSISSO / E2E / PhySO**。mean/linearは比較基準です。

同じ入力でも探索空間・制約の適用段階・保証は手法ごとに異なります。

## 使う

WindowsではPythonのlauncherからLinux Dockerを使います。初回は [導入](docs/user_manual.md) を確認してください。

```powershell

python scripts/run_linux.py -- python examples/basic.py --output results/basic

```

これは小さな合成例の探索・Selection比較です。5手法すべての実行や精度保証ではありません。

自分の問題では [入力](docs/user_manual.md) に沿ってCSV、単位、固定split、条件と予算を設定します。

```powershell

python scripts/run_linux.py --config my_problem.json --run-dir results/my_run/calculation --compare

```

`my_run`は新しい名前にします。`--compare`では`--run-dir`の末尾を`calculation`にします。

既定の出力rootは`results`です。別rootを指定するときは比較出力の親もそのroot内へ置きます。不適合な配置は探索前に拒否します。

探索のみなら`--compare`を省略します。比較は採用判断・再fit・Final採点を行いません。

メモリなどの実行資源は利用可能量と問題に合わせて明示し、上限到達を精度限界と解釈しません。

```text

results/my_run/

  equations.txt  式と係数

  figures/       PNG・SVGと独立した数式画像

  tables/        指標・制約・グラフの元CSV

  calculation/   入力、候補、評価・比較の計算記録

  metadata/      表示用JSONと出力hash

```

## 読む順序

| 文書 | 内容 |

| --- | --- |

| [ユーザーマニュアル](docs/user_manual.md) | 役割、導入、データ・設定、5手法、実行、式・グラフと結果の読み方 |

| [開発者仕様書](docs/developer_spec.md) | 背景、実装の工夫、構成と責務、手法ごとの接続・限界、保存、検証、参考文献 |

用途別の小例は [examples](examples/README.md)、回帰テストは [tests](tests/README.md)。

開発前に [プロジェクト方針](AGENTS.md) を読みます。版の正本は `pkgs/symbolic/pyproject.toml` です。

source、wheel、実行imageが一致するかは版名だけでなくmanifestとhashで確認します。

## 範囲

単位、既知式、応答条件の全併用に全手法が対応するわけではありません。

未対応は実行前に拒否し、無制約探索や別手法へ置換しません。

有限点の制約適合は全域保証ではなく、良い予測は機構・因果・未知法則の同定ではありません。

Trainで探索、Selectionで代表固定、必要な場合だけFinalを採点します。既露出Finalを未知testへ戻しません。

`data/`・`pretrained/`は入力と資産、`runtime/`はローカル環境、`results/`は実行結果、`build/`は開発生成物です。

これらの個別実行や過去結果は第三者向けのsource手順には含めません。

## Public derived snapshot and optional assets

This is a public derived snapshot of the current platform, datasets, saved results and self-fit portable models. See [evaluation scope](results/practical-fixed-20261115-1400/EVALUATION_SCOPE.txt): historical results retain their original meaning; later fixes do not retroactively rescore them. Operational diagnostics containing local user paths are excluded. Git history, local environments and internal build evidence are not included.

The official E2E `model1.pt` is deliberately **not included**. Download and verify it with the existing setup script from a configured environment:

```sh

python scripts/setup_e2e_assets.py --directory pretrained/e2e

```

Official URL: https://dl.fbaipublicfiles.com/symbolicregression/model1.pt

SHA256: `169569a0648ae1a4f1ac2bbb376fa5ebb70ebce4bfcf23d6666323596990ff91`

Code revision: `c4144c99d078dd611795338e24fb3da49a32b9d8`

Code Apache-2.0 does not establish separate weight redistribution rights; those rights remain unconfirmed here.

Optional Flash/RASR model-only ZIPs are separate public Release assets, with exact hashes and exact-revision official model-card/API MIT and weight-hash evidence in `ASSET_MANIFEST.json`. Download them from the public [models-v1 Release](https://github.com/dkawa2523/symbolic-identification-platform-public/releases/tag/models-v1):

```sh

gh release download models-v1 --repo dkawa2523/symbolic-identification-platform-public --pattern flash-model-only.zip --pattern rasr-model-only.zip --dir downloaded-models

python -m zipfile -e downloaded-models/flash-model-only.zip pretrained/flash/model

python -m zipfile -e downloaded-models/rasr-model-only.zip pretrained/rasr/model

```

Before extraction, compare `Get-FileHash downloaded-models/*.zip -Algorithm SHA256` with `ASSET_MANIFEST.json`. These model-only archives exclude simplipy assets and caches. Use the existing explicit `scripts/setup_flash_assets.py` / `scripts/setup_rasr_assets.py` setup procedures to obtain and verify remaining official assets. Extraction alone does not restore the original complete asset manifests. No LFS billing configuration is changed by this snapshot.

This publication is **not the original complete frozen snapshot**. `PUBLICATION_SCOPE.json` records privacy exclusions and contact redaction with original/public hashes. Historical manifests may still reference excluded or modified original files; their original hashes are not claims about these derived files. Numerical results, portable equations/models and Final scores were not recalculated or modified. Original private records remain separate.
