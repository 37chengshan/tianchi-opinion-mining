# 官方测试数据 README（转录）

1. 提交结果包括 5 个字段：`id，AspectTerm，OpinionTerm，Category，Polarity`，一条结果保存为一行；

2. 预测结果为空的字段以 `_` 表示；

3. id 不可以为空，从 1 开始编号，顺序递增，与 Test_reviews.csv 中的 id 保持一致，不可遗漏 id，也不可多 id，否则将报错。例如测试数据中 id 为 1-10，那么提交结果中的 id 也应该为 1-10。若某 id 对应评论的所有字段的预测结果均为空，也应保留该 id，其他字段用 `_` 表示即可；

4. Result(example).csv 为提交样例文件，参赛者请按照该文件中的格式将预测结果保存为名为 `Result.csv` 的文件，最后提交 `Result.csv` 即可；

5. 提交的文件中不需要表头部分，只要给出预测结果即可；

6. 必须提交无 BOM UTF-8 格式的 `.csv` 文件。
