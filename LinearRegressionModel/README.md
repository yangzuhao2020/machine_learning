作业2：金鱼年龄预测
任务描述：

金鱼年龄的预测是现实生活中出现过的一项任务，给出一份金鱼的训练数据集，训练一个线性回归模型并对测试集中金鱼的年龄进行预测。



输入数据：

在train/目录下包含一个train.csv文件，其中每行表示一条金鱼的特征和年龄。文件中每行共有9列，前8列为金鱼的身体状况参数，最后一列表示该金鱼的年龄。

在test/目录下包含一个test.csv文件，与train.csv类似，只不过test.csv不包含金鱼的年龄数据，需要根据金鱼的身体状况参数给出预测。



输出数据：

程序需要生成一个result.csv文件，用于保存程序对金鱼年龄的预测结果。第一行固定为age，之后每一行为一个数值代表预测的金鱼年龄，表示程序对test.csv中对应行的金鱼年龄预测结果。



注：

评测信息中的'rank': '0.10350234624825604'表明测试集的RMSE=0.10350234624825604



输入样例：

n1,n2,n3,n4,n5,n6,n7,n8,age
1,0.455,0.365,0.095,0.514,0.2245,0.10099999999999999,0.15,15
1,0.35,0.265,0.09,0.2255,0.0995,0.0485,0.07,7
-1,0.53,0.42,0.135,0.677,0.2565,0.1415,0.21,9
0,0.33,0.255,0.08,0.205,0.0895,0.0395,0.055,7
-1,0.53,0.415,0.15,0.7775,0.237,0.1415,0.33,20


输出样例：

age
11.419235677462172
11.602508087480594
11.842765817087468
6.261259406734112
7.960520691911781
7.54970269783407


 