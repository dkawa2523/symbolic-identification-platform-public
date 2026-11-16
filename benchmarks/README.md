# 比較実験

普段の探索・Selection比較は [通常入口](../README.md)、実験の分割と評価規則は [評価方針](../docs/developer_spec.md) を使います。各runの設定・データ・seed・実効native予算・source/asset/image hashが正本です。過去の結果や未実施計画を現行仕様へ重ねません。

## 入口

| 目的 | 入力・コード |
| --- | --- |
| 通常5手法の実データ比較 | [Yacht](yacht/README.md)、`yacht/main.json` |
| 既知式と未知関数の小確認 | [partial_equations](partial_equations/README.md) |
| 制約なし/後判定/探索時利用の対照 | [constraints](constraints/README.md) |
| 既知式の係数推定 | [NIST](nist.md) |
| 独立/交絡入力と追加測定の対照 | [identification](identification.md)、[Proton](proton.md) |
| 公開測定条件の外挿 | [Airfoil](airfoil.md) |

## 合成caseと部分設定

`scripts/benchmark_symbolic.py --prepare-only`が固定表・split・完全configを生成します。下表のJSONは追加optionであって、単独の完全configではありません。データseed、split、法則、知識の有無は生成metadataと解決済みconfigに保存します。真式/clean labelsはscorer専用で、探索へ渡しません。任意手法の成功や未使用testは仮定しません。

| generator case | optionの例 | 注意 |
| --- | --- | --- |
| damped_resonance / activated_kinetics | settings/physics.json | known_formには追加構造知識があり、blind法と同じ条件ではない |
| rational_unknown / sine_unknown | settings/comparison.json、proged.json、pse.json | 公開した合成族。新標本でも未知構造の証明ではない |
| ambiguous_quadratic / correlated_interaction | settings/pysips.json | ensemble構造spreadと観測予測区間は別 |
| dimensional_ballistic | settings/physo_reference.json | 既知固定定数と単位を変えず、非対応も残す |
| sparse_nuisance / dimensional_sparse | settings/torchsisso.json、torchsisso_units.json | relevant変数を探索へ教えず、次元付き係数の表現前提を確認 |
| nonlinear_noisy_ood | settings/pretrained_noise.json | noisy/外挿入力。事前学習重複はunknown |

```powershell
python scripts/run_linux.py --profile core -- python scripts/benchmark_symbolic.py --prepare-only --output results/my-cases --cases rational_unknown --engines pysr parfam --seeds 17 --data-seed 123 --seconds 300
python scripts/run_linux.py --config results/my-cases/rational_unknown/17/config.json --run-dir results/my-comparison/calculation --compare
```

予算値は構文例で、品質推奨ではありません。条件を結果を見て変えず、同じ比較内の各手法に同じ観測・既知情報を渡します。既存caseのFinal利用歴は既露出/不明として保持し、再実行を独立確認と呼びません。
`benchmark_symbolic.py --collect-tasks`等は過去形式の保存結果だけのreaderで、現在のlauncher出力の収集入口ではありません。

## 代表固定後のFinal採点

```powershell
python scripts/run_linux.py --profile core -- python benchmarks/score.py --study results/my-comparison/calculation/study --output results/my-final-score
```

実験で必要な全選択を先に固定します。このscorerはSelection最良の適格代表だけを採点し、再fit・Final選択をしません。Final区画がなければ拒否し、Selectionで代用しません。`final_history`はknown_exposed/unknown/declared_unseenを実態に合わせます。外部利用歴を自動認証するものではありません。

## その他の入力生成

`prepare_data.py`は組成chart、入力測定誤差、UCI316/551の表と固定分割を生成します。明示`--fetch`以外はネット取得しません。データ出典/利用条件/固定hashは生成metadataを使います。
`quality/ccpp.py`は [UCI294 CCPP](https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant)（Tfekci & Kaya、DOI10.24432/C5002N、CC BY4.0）の既存source.zipを検査します。5シートは同一観測の並べ替えなので別データとして連結せずSheet1を使います。正確な行/group/単位/hashはdata/ccpp/metadata.jsonが正本です。旧quality画面のSelectionや過去Finalを新しい独立性能へ読み替えません。
