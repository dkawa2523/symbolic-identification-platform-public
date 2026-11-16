# ml-platform-symbolic

同じ観測・単位・成立領域・要求から関係式候補を作り、用途別に比較・採否・保存再利用するpackageです。版は `pyproject.toml` が正本です。探索完了、native忠実性、制約適格性、科学的利用の支持を区別します。

## 利用

source repositoryのルートから、対応Linux Docker環境で実行します。

```sh
python scripts/run_linux.py --profile core --build -- python examples/basic.py --output results/basic
```

これは合成の導入確認であり、未知法則回復の証明ではありません。E2Eはリポジトリの `docs/developer_spec.md`、native環境は `docs/user_manual.md`、全例は `examples/README.md` を参照してください。Windows nativeのprocess-tree回収は対応保証外です。

標準flowは discovery → `create_study` → 必要な明示fit/解析 → `compare_equations` → `choose_hypothesis` → 必要なら独立校正と `assess_prediction_use` → `finalize_hypothesis`。比較・描画はfit・再選択・Final読取を行いません。旧 `run_task` に `task: symbolic_pipeline` の設定を渡す経路 の自動Finalは明示legacyです。

## 保存と依存

scalarモデルの推論はClearML、native探索ライブラリ、checkpointを必要としません。保存列名、公開単位、元domainと入力写像、固定/自由係数、算術意味を維持します。ensembleの粒子・重み・重複とimplicitの残差・明示枝は別形式で保存します。

対応する旧artifactを元の意味で読み、未知version・hash不整合・矛盾する単位は拒否します。新しいtemplateやviewを旧runtime向けに黙って削りません。形式別の対応は `docs/developer_spec.md`、用途と出力は `docs/user_manual.md`、共通契約は `docs/developer_spec.md` が正本です。

`export_result` / `load_result` は保存readoutと明示した同じsubjectの予測器/区間を持ち運びます。live解析環境や元CSVの移転ではありません。校正済み判断は同じ区間hashへ結び付き、異なる校正へ流用できません。

## 導入・配布の条件

社内Gitのclean sourceが主経路です。`docs/user_manual.md` と `docs/user_manual.md` を確認してください。coreは `ml-platform-core` の固定commit `91aa29a235527b9ecd6fc7f6d17101beb0cfe9a0` を使います。社内URL、公開index上のcore配布、権利者を仮定しません。

offlineセットを使う場合は、そのREADMEとhash付きrequirementsに従い、新しいLinux環境へ導入します。モデルweights・実測データ・計算結果はclean sourceへ含めません。既存の第三者LICENSE/NOTICEを保持し、本体/coreの利用・複製・受渡し権限とcheckpoint条件は別に確認します。

実施済み検証は同じsourceの記録を参照してください。有限点の制約確認、条件付きensemble、局所係数識別性は、全域保証・真式・因果・任意外挿の証明ではありません。
