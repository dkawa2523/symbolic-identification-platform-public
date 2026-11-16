# 制約を探索に使う比較

## 現行5手法の対照評価

`settings-main.json` は PySR/ParFam/TorchSISSO/E2E/PhySO の計画設定です。`none` と明示的な `hard` 探索の2対照、同じ固定分割・探索seed17。合成問題は過去に検討した開発問題であり、新しいTest座標を生成しても未知法則発見の証拠ではありません。Concreteは既存group分割652/149/229行、既露出Test、事前学習重複不明です。元単位を共通変換し、圧縮強度の非負条件だけをTrain由来の同じ17点で要求します。実データの因果的単調性などは仮定しません。

5手法のnative目的・探索範囲・計算量は同一ではありません。common hardは有限点の条件を意味しますが、PySR fitness、ParFam族内fit、TorchSISSO subset回帰、E2E生成後の係数fit、PhySO報酬前fitという適用段階は異なります。E2E decoder自体は応答非条件付き。短い一律時間を品質合否にせず、600秒は安全上限、部分結果と終了理由を報告します。native profileは設定ファイルと各job/summaryに記録します。

合成の元座標AST上限64、Concrete320。Concreteの物理単位変換込みで二次ParFam分子271、一次分母付き307nodeとなり、256ではその族を排除するため、結果を見る前の表現コストから320を設定しています。それでも上限内の全式網羅とは呼びません。初期設定は実行前preflightで検証し、Testを用いて調整しません。

合成のprepareは既存入口に `--settings benchmarks/constraints/settings-main.json --engines pysr parfam torchsisso e2e physo` を指定し、`discover --engine <手法>` を適切なnative環境で順次呼びます。従来の3対照/既知式実験は過去手順との互換として残り、下記の旧成績を現行5手法の結果と混同しません。設定ファイルは実行結果ではありません。

同じ少量の観測、演算子、乱数、計算上限で、次の三条件を比べます。

- `none`: 応答制約を与えない。
- `filter`: 同じ応答制約を探索後の候補選別だけに使う。
- `search`: 応答制約を探索中に反映し、最終候補も共通条件で評価する。

全3条件で `objective: "mse"` を明示し、共通のTrain尺度で正規化したMSEを使います。native固有の別の損失との差を制約の効果と取り違えないためです。

単一の合成ケースです。狭い観測範囲の16点にノイズを加え、未知の広い領域で増加・1〜3の応答範囲という事前情報を使います。真の式 `1+2*x/(1+x)` は採点用で、探索へ渡しません。新しい式族の発見や実装置の妥当性を示すものではありません。単位・既知部分式の組合せを網羅するベンチマークでもありません。

## 実行

リポジトリ直下で、変更後のclassic環境が検証済みであることを確認して実行します。

```powershell
py -3.13 scripts/run_linux.py --profile classic -- python benchmarks/constraints/run.py prepare --output results/constraints-001 --seeds 17 --seconds 120
py -3.13 scripts/run_linux.py --profile classic -- python benchmarks/constraints/run.py discover --output results/constraints-001
py -3.13 scripts/run_linux.py --profile classic -- python benchmarks/constraints/run.py select --output results/constraints-001
py -3.13 scripts/run_linux.py --profile classic -- python benchmarks/constraints/run.py score --output results/constraints-001
```

`prepare` は同じデータと分割を各条件に固定します。`discover` は既存のnative workerで実行し、既存の試行を上書きしません。`select` はSelectionだけで各手法の代表式を選び、全選択を保存してから終了します。`score` はその後に独立Testを開きます。採点後の選び直しは禁止し、実装調整は新しいrunとして扱います。

探索中の制約点は17点、独立検査は別の103点です。中央差分は有限の幅の応答であり、大域的な微分符号の証明ではありません。独立Testは補間61点と外挿81点です。Trainのノイズに対しTestは真の平均応答の評価なので、実測ノイズ込み予測精度とは区別します。

## 結果

- `summary.csv`: 全試行の実行状態、選択式、Selection精度、独立Testの補間・外挿精度、独立制約適合。失敗を除いた平均だけで順位付けしません。
- `scores.json`: 同じ結果と制約別の評価範囲・違反記録。
- `<条件>/<手法>/<seed>/output/`: 既存の共通出力で作った式一覧、PNG、CSV。Selectionの証拠でありFinalの図とは区別します。
- `<条件>/<手法>/<seed>/test_predictions.csv`: 代表式を固定した後の独立予測値。

`none` の代表式は制約なしで選び、他の二条件は同じ制約で適格性を判定します。この選択規則の違いも処置の一部です。全条件の最終制約検査は同じ独立点で行います。`filter` と `search` の比較が、最終候補選別だけではなく探索へ反映する効果を調べます。

このファイルの存在やprepareの成功は、native探索成功や性能改善の証拠ではありません。変更後の実際の実行記録から判定してください。

## Historical search-objective context

The earlier PySR `search` runs below used a finite-weight mean-squared violation penalty. The current five-method protocol explicitly requests hard enforcement. ParFam and template_enumeration use explicit finite-point constrained local optimization; `constraint_weight` is rejected for these two paths. Neither mechanism guarantees finding a feasible model or global validity. The initial smoke and aligned runs below used the earlier soft objective for both native engines. A returned candidate can still fail the common constraints; an execution success is not a useful-model success. Current search and common bound checks share the Train-derived domain/geometry points plus the 17 declared probe points. The derivative checks use the 17 declared centers. The initial smoke predates this alignment. Independent scoring uses a separate 103-center grid.

The recorded `results/constraints-20261113-smoke` campaign used seed 17 with concurrent validation load. All six native runs completed, but search returned only a constant eligible PySR model and no eligible ParFam model. This is integration and failure-path evidence, not proof of constraint effectiveness or a method ranking. Its `audit.json` preserves the Train/Selection-only diagnosis.

The separate `results/constraints-20261113-aligned` rerun checks point alignment with unchanged weights. Both engines received 39 bound points. PySR returned three eligible candidates including a nonconstant one; ParFam returned none. Since the same development case and already-inspected test data were reused, this is diagnostic evidence, not independent confirmation of effectiveness.

## 開封前に固定する新しい検証

`--test-seed` は別の固定乱数でTest座標と検証条件点を生成します。観測Train/Selectionは従来と同一です。値は `test.csv` と `test_conditions.json` に隔離し、`plan.json` には条件値を載せずハッシュを保存します。`test_lock.json` は未閲覧から一度だけの採点へ状態を記録します。これは運用上の開封手順であり、ファイルのアクセス権による遮断ではありません。

改善中は `discover` と `select` までに留め、Testファイルや条件点を開かないでください。全候補選択を固定後に一度だけ `score` を実行します。採点後の調整・再選択はこのTestでは評価しません。同じ生成関数なので、新しい物理系への汎化や事前学習汚染を検証するものではありません。

`--known-relation --engines pysr template_enumeration` は共通の仮説 `y = 1 + h(x)` と同じ応答条件を2経路に入力します。未知のhや正解係数は探索に渡しません。列挙の構造順とnative進化探索は異なり、同じ探索量を保証しません。固定構造予算・壁時計上限と実際の完了範囲を記録して比較します。既存のprepare/discover/select/scoreをそのまま使い、専用の評価実装を増やしません。

保存済みの通常18件・既知式6件のholdoutは、PySRのslot別ノード上限を探索loss内で棄却する修正より前の記録です。修正後の仕様と旧計算の結果を混同しません。これらの結果は再計算・再採点していません。


Current scope correction: explicit response probes remain unchanged. Automatic
prediction-domain probes now check only finite prediction/target-domain validity;
they no longer inherit user response bounds. Historical smoke/aligned/holdout
artifacts retain their original combined-probe semantics and are not recalculated
or relabelled as evidence of the corrected implementation.

Standard entry uses one fixed seed17. Historical multi-seed result records below are preserved; they are not the default execution plan.
