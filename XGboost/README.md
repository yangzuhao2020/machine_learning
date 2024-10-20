作业4
**基于XGboost的收入分类预测**
输入数据：

在train/目录下包含一个train.csv文件，其中每行表示数据的一条记录。

在test/目录下包含一个test.csv文件，与train.csv类似，只不过test.csv不包含income列，您需要根据参数给出预测。



输出数据：

您的程序需要生成一个result.csv文件，用于保存您程序的预测结果。第一行固定为income，之后每一行为一个1或0的值表示预测结果。


评价指标：

准确率。



输入样例：

age,workplace,id,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,income
39, State-gov,77516, Bachelors,13, Never-married, Adm-clerical, Not-in-family, White, Male,2174,0,40, United-States,0
50, Self-emp-not-inc,83311, Bachelors,13, Married-civ-spouse, Exec-managerial, Husband, White, Male,0,0,13, United-States,0
38, Private,215646, HS-grad,9, Divorced, Handlers-cleaners, Not-in-family, White, Male,0,0,40, United-States,0


输出样例：

income
1
0
0
0
1