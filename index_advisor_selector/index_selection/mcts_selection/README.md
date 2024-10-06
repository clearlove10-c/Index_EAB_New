# 数据说明
测试用的工作负载数据集为 /volume_ai/Index_EAB_New/workload_generator/random/tpch_work_multi.json
其中包含多组（100组）工作负载，每组工作负载包含一组（18条） 查询语句，每条查询语句格式：
```sql
[
    1,		# query ID
    "SELECT MIN(mc.note) AS production_note, MIN(t.title) AS movie_title ...",	# query SQL
    666		# query frequency
],
...

```
不同组的工作负载仅有频率以及SQl子句中的具体数值有细微差别

# 使用方法
运行`mcts_advisor.py`文件，对上述的多组工作负载进行测试
>这里只对工作负载的前10组进行测试，可通过`mcts_advisor`中的`work_list = work_list[:10]`修改

```bash
python mcts_advisor.py \
--res_save <path_to_result_save_file> \
--overhead \
--budget 25 \
--storage 100 \
--min_budget 50 \
--work_file /Index_EAB_New/workload_generator/template_based/tpch_work_temp_multi.json \
--db_file /Index_EAB_New/configuration_loader/database/db_con.conf \
--schema_file /ndex_EAB_New/configuration_loader/database/schema_tpch.json
```