# 探索・制約の追加設定

ここには既存のベンチマークへ重ねる**部分設定**だけがあります。データと分割を含む完全な設定ではないため、`run_linux.py --config` には直接渡しません。`benchmark_symbolic.py --overrides` に指定します。

| 設定 | 用途 |
| --- | --- |
| [physics.json](physics.json) | 振動・活性化反応の制約付き比較 |
| [physo_reference.json](physo_reference.json) | 次元付き探索と明示した既知関数形の対照 |
| [torchsisso_units.json](torchsisso_units.json) | TorchSISSOの次元付き探索 |
| [pretrained.yaml](pretrained.yaml) | 学習済みモデルを使う探索の共通設定 |
| [comparison.json](comparison.json) | PySR・ProGED・事前学習手法の共通比較予算 |
| [pysr.json](pysr.json) | PySRの短時間予算 |
| [physo.json](physo.json)、[proged.json](proged.json) | 各手法の探索設定 |
| [pse.json](pse.json)、[pysips.json](pysips.json) | ニューラル探索・事後分布推定の設定 |
| [torchsisso.json](torchsisso.json)、[parfam.json](parfam.json) | 疎な式・関数族の探索設定 |
| [pretrained_noise.json](pretrained_noise.json) | ノイズ条件を変える事前学習手法比較 |
| [flash_constraints.json](flash_constraints.json) | Flash-ANSRの候補と制約の確認 |

```powershell
py -3.13 scripts/run_linux.py --profile core -- python scripts/benchmark_symbolic.py --prepare-only --output results/physics/calculation --cases damped_resonance activated_kinetics --seeds 0 --engines mean linear dictionary --overrides benchmarks/settings/physics.json
```

このコマンドは入力準備だけです。探索・選択・Final評価の実行は、[ベンチマーク一覧](../README.md)から対象の評価手順を選んで確認してください。設定を置いたことは、その手法の成功や性能検証を意味しません。学習済み資産は `pretrained/` のセットアップ済みローカルファイルだけを参照します。

`--overrides`はprepare-onlyの入力生成に使います。生成後は通常launcherの--config/--compareへ進み、Finalは明示した代表採点で行います。旧collectは過去形式の保存結果だけのreaderです。
