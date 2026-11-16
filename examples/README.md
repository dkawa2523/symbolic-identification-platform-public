# 利用例

軽量な動作確認は [basic.py](basic.py) です。mean/linear/dictionaryの3基準手法を使う例で、通常利用の5手法を評価した証拠ではありません。5手法を探索からSelection比較・PNG出力まで実行する`--compare`と個別設定は[入力ガイド](../docs/user_manual.md#通常利用の5手法と個別設定)を参照してください。合成データの作成、固定Train/Selectionでの探索、手法別の比較、数式・PNG・CSV出力まで実行します。Final採点・運用採用・区間校正は行いません。

```powershell
py -3.13 scripts/run_linux.py --profile core -- python examples/basic.py --output results/basic
```

同じ出力先は上書きしません。再実行時は `--output results/basic-2` のように名前を変えます。数式・PNG・表と計算記録は指定した結果フォルダにまとまります。

| 確認したい内容 | 入口 |
| --- | --- |
| 独立校正・予測区間・Final監査・保存後推論 | [prediction_intervals.py](prediction_intervals.py) |
| CSVの列、単位、制約の入力 | [input/](input/README.md) |
| 既知の式と未知関数を組み合わせる | [partial_equations.py](partial_equations.py) |
| 単位のある複数の未知関数を同時に推定する | [units.py](units.py) |
| 組成などの入力制約と座標変換 | [geometry.py](geometry.py) |
| 入力の独立性・交絡と予測可能な範囲 | [support.py](support.py) |
| 絶対値と変化量の評価を区別する | [response.py](response.py) |
| 手法・評価処理を追加する | [extensions/](extensions/README.md) |

```powershell
py -3.13 scripts/run_linux.py --profile core -- python examples/partial_equations.py --enumerate --output results/partial-equations
py -3.13 scripts/run_linux.py --profile core -- python examples/units.py --output results/units
py -3.13 scripts/run_linux.py --profile core -- python examples/geometry.py --output results/geometry
py -3.13 scripts/run_linux.py --profile core -- python examples/support.py --output results/support
py -3.13 scripts/run_linux.py --profile core -- python examples/response.py --output results/response
```

これらは用途別の合成例です。未知法則の発見や全手法の性能を証明するものではありません。データセットを使った比較は [benchmarks](../benchmarks/README.md) を参照してください。
