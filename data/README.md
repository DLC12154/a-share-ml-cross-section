# 数据说明

`raw/csi300_prices_historical.parquet` 来自改进后的动量项目，覆盖历史名单并集481只股票，运行代码会校验SHA-256。

`reference/csi300_historical_membership.csv` 包含291个历史名单查询日、87,300条成员记录、481只股票。

`raw/csi300_prices_2020_2025.parquet` 只作为旧固定300只股票快照保留，当前实验不再使用。

数据文件不提交到Git。缺少文件时运行：

```powershell
python scripts/build_historical_dataset.py --workers 4
```

数据仅用于个人研究复现。再次公开或分发前应核对数据供应商的使用条款。

