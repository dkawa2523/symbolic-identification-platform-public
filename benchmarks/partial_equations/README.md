# 既知外枠の最小確認

`current_campaign.py`は既存prepare/discover/studyと代表scorerを使う小さい機構確認です。既定はadditive、template_enumeration、search seed17、worker30秒/32構造。既知の2*xだけを指定し、未知関数の真式を探索へ渡しません。短い品質推奨や5手法全部の確認ではありません。

```powershell
python scripts/run_linux.py --profile core -- python benchmarks/partial_equations/current_campaign.py prepare --root results/partial-common
python scripts/run_linux.py --profile core -- python benchmarks/partial_equations/current_campaign.py discover --root results/partial-common
python scripts/run_linux.py --profile core -- python benchmarks/partial_equations/current_campaign.py select --root results/partial-common
python scripts/run_linux.py --profile core -- python benchmarks/partial_equations/current_campaign.py score --root results/partial-common
```

新規rootで各段階を一度実行します。prepareがprotocolを先に保存し、観測[-1,1]^2、予測domain[-1,2]^2、補間と外挿[1.2,2]^2の採点点32ずつを固定します。外挿は観測外でも宣言domain内です。元のFinal行を除外しますが、公開した合成族なのでhistoryはknown_exposedです。selectはSelection study作成、scoreは代表を固定後にFinalを採点し、失敗/不適格も残します。条件や予算の変更は別prepareです。
`--cases/--engines`はprepare時の明示選択です。PySRを選ぶdiscoverにはclassic環境が必要です。

## 固定fixture

`data/partial_equations`のmanifestを`test_data.py`で確認します。Windows Python3.12.8が記録された生成環境で、他環境のmath差を黙って同一CSVへ上書きしません。sourceの数式文字列は説明で、eval設定ではありません。現入口は小さいdefinitions.pyのDATA/template/OPERATORS定義だけを参照します。旧campaign実行・専用集計・単発代数診断は削除しました。旧sealed/hybrid/外挿の結果は現入口へ取り込みません。

Abaloneは [UCI1](https://archive.ics.uci.edu/dataset/1/abalone)、Nashほか(1994)、DOI10.24432/C55C7W、CC BY4.0です。shell weightをgeometryから予測する用途へ再構成し、source.namesと出典・行/group・splitを保持します。元単位の説明にはscaleの曖昧さがあるため数値座標unit1の条件であり、密度/SI量の同定ではありません。zero-height等の難しい行も保持します。volume*hという外枠は仮説であって真理ではありません。

共有slot/成分の良い予測は成分一意性を意味しません。wrong prior、交絡、zero factor、加法的gauge、strict-real domainを別の対照として扱います。未知構造回復や因果の成功件数へ機構テストを足しません。
