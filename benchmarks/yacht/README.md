# Yachtの通常比較

`main.json`は中心5手法/seed42、`baselines.yaml`はcheckpoint不要の小さい基準例です。同じ`data/yacht/table.csv`と固定`split.csv`を使います。データ準備と出典は [評価](../../docs/user_manual.md#5-探索と比較を実行する)、profile/資産は [導入](../../docs/user_manual.md) を参照してください。

```powershell
python scripts/run_linux.py --config benchmarks/yacht/main.json --run-dir results/yacht-new/calculation --compare
```

新しいrun名を使います。mainの各手法3600秒等は開始条件で、精度/収束の保証ではありません。状態、実効native設定、候補/不適格理由、source複雑度、元単位のSelection誤差を比較します。PNG表示代表は採用ではなく、Finalを開きません。
この公開splitは既露出で、未見hull/外部testの保証ではありません。派生geometry groupの越境がなくても実hull IDや類縁形状の独立性は保証しません。

その他JSON/YAMLは手法や文法/予算の異なる明示設定です。`e2e_study.json`だけは探索結果を読むplanで、pathはplan基準です。新runへ使う場合はprepared_runと全sourcesを同じrunへ揃えます。all_methodsや任意profileの設定を標準5法の成功一覧としません。
追加のFinal評価は全代表を固定してから共通`benchmarks/score.py`へ進みます。過去の13法/複数seed専用の集計writerは廃止し、原結果の読込と現在の実行を混同しません。
