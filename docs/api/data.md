# Data

The data layer loads, samples, and partitions DataFrames to feed the Runner.

```
DataSource → SamplingStrategy → DataPartitioner → (current, reference)
   load          narrow window          split
```

→ [Full data layer guide](../data-layer.md)

---

## Sources

::: ayn_ml.data.source.DataFrameSource

::: ayn_ml.data.csv.CsvSource

::: ayn_ml.data.excel.ExcelSource

---

## Sampling strategies

::: ayn_ml.data.sampling.SamplingStrategy

::: ayn_ml.data.sampling.LastNRowsSampling

::: ayn_ml.data.sampling.TimeWindowSampling

::: ayn_ml.data.sampling.RandomSampling

---

## Partitioning

::: ayn_ml.data.partitioner.TimeBasedPartitioner
