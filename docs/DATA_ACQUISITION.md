# 数据获取与冻结

## 当前使用的数据

- `data/reference/csi300_historical_membership.csv`
- `data/raw/csi300_prices_historical.parquet`
- `data/reference/historical_dataset_manifest.json`

数据由Baostock日线与`query_hs300_stocks(date=...)`历史名单构造，研究期2020-01-02至2025-12-31，额外下载2019-07-01以后行情用于特征预热。

## 重新下载

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\build_historical_dataset.py --workers 4
```

脚本会：

- 每5个交易日查询一次沪深300历史成分，共291个快照。
- 校验每个快照有300个唯一代码。
- 下载历史成员并集的481只股票行情。
- 缓存单日和单股票结果，以支持中断续跑。
- 生成行情与名单的SHA-256清单。

Baostock可能限流或修订历史数据。项目代码会校验`config.json`中的行情哈希；供应商修订导致哈希变化时，实验会停止，必须先审阅变更并更新清单。

## 缺少数据时

`tests/test_real_data.py`只在冻结行情存在时运行，否则显示跳过原因。`tests/test_core.py`使用合成数据，不依赖本地行情。

