# 未完成的 exact 全网格尝试

- 状态：`ABORTED`，不得作为完整网格结果引用。
- 原因：在前 32 个配置完成后，`m=32, lambda=1, honest=Alice, first=Alice`
  的显式状态已经导出，但 exact 有理数引擎对同一图上的 10 个可达性查询耗时不成比例。
- 处置：保留已经生成的日志和状态文件，不覆盖、不补写为 PASS。后续唯一完整
  run-id 改用显式数值引擎（记录 epsilon），而从完整 `.sta` 以整数计算最大差值；
  小型负例仍用 exact 引擎交叉检查。
- 本目录没有完整 `summary.json`，也不应与后续完整 run-id 合并。
