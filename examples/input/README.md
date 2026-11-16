# 入力設定

[input_demo.json](input_demo.json)、[input_demo.csv](input_demo.csv)、[input_demo_split.csv](input_demo_split.csv) は、列・単位・制約と固定分割を指定する小さな入力例です。

```powershell
py -3.13 scripts/run_linux.py --profile core -- python -m ml_platform_symbolic validate examples/input/input_demo.json
```

これは入力の確認だけで、探索や精度評価は実行しません。計算からPNG・数式までの例は [basic.py](../basic.py) です。

[入力ガイド](../../docs/user_manual.md) / [利用例一覧](../README.md)
