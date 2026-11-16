# ベンチマーク入力データ

再利用する入力データはここへまとめます。計算結果や学習済みモデルは置きません。

| ディレクトリ | 内容・手順 |
| --- | --- |
| yacht/ | UCI Yachtの表と固定分割。[手順](../benchmarks/yacht/README.md) |
| airfoil/、concrete/ | UCIの表・固定分割・元アーカイブ。[データ案内](../docs/user_manual.md#5-探索と比較を実行する) |
| ccpp/ | UCI CCPPの原本・Sheet1表・5シート同一性/重複監査。元source/表/監査のmetadataを確認。[入力生成](../benchmarks/README.md) |
| nist/ | 既知関数形の係数推定用データと配布条件。[手順](../benchmarks/nist.md) |
| partial_equations/ | 部分既知式の合成データ、Abalone、分割と出典。[手順](../benchmarks/partial_equations/README.md) |
| external/ | 出典ごとに保存した外部データ。[Proton](../benchmarks/proton.md) |

元データの出典・ライセンス・ハッシュ・固定分割は、各データと一緒に保持します。既知の正解式を含む採点用データは探索へ渡しません。公開データとの事前学習重複が不明なら未知法則の発見とは扱いません。

追加データの準備は `scripts/prepare_validation_data.py`、相補的な合成例・UCI316/551の準備は `benchmarks/prepare_data.py` です。取得を伴う場合は各コマンドの明示的な取得オプションを使います。
