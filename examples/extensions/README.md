# 明示的な拡張の例

| ファイル | 役割 |
| --- | --- |
| [installed_extension/](installed_extension) | 本体とは別にインストールする教材package。既知次数のNumPy polynomial adapter、非連結domain、領域確認evaluator |
| [runtime.py](runtime.py) | 別workerでの発見・解析と保存後推論を確認。`--output` は新しいdirectory |
| [feasible_response_evaluator.py](feasible_response_evaluator.py) | 実行可能方向の解析を加えるevaluator。単体のCLIではない |
| [residual_quantiles_evaluator.py](residual_quantiles_evaluator.py) | 保存残差の経験分位を加えるevaluator。信頼区間ではない |

インストールと実行は [拡張手順](../../docs/developer_spec.md) を使ってください。別runtimeにも同じmodule/versionを明示導入する必要があります。保存モデルを読むだけで拡張コードが導入されることはありません。

終了時は `verification.json` のnative status・prediction誤差を読みます。この例は既知次数familyの技術確認で、汎用SRや物理的識別の実証ではありません。
