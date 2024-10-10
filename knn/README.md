作业1：基于KNN的IRIS分类
任务描述：

请设计一个分类器，根据花朵的萼片长度(sepal length)、萼片宽度sepalwidth)、花瓣长度(petal length)和花瓣宽度(petal width)来预测它属于三种不同的鸢尾属植物setosa、versicolor和virginica中的哪一种。



输入数据：

在train/目录下包含一个train.csv文件，其中每行代表一个已知样例。文件中每行共有6列，第一列为id，2-5列为四个属性值，最后一列表示该花朵属于哪种植物，分别用0，1，2来表示setosa、versicolor和virginica。

在test/目录下包含一个test.csv文件，与train.csv类似，每一行表示一朵花瓣的四个属性参数和ID，不过不包含它的分类值，您需要根据参数给出预测。



输出数据：

你的程序需要生成一个result.csv文件，用于保存你程序对花朵情况的预测结果。输出csv文件格式见下方



输入样例：

Id,Sepal.Length,Sepal.Width,Petal.Length,Petal.Width,Species
1,5.1,3.5,1.4,0.2,0
2,4.9,3.0,1.4,0.2,0
3,4.7,3.2,1.3,0.2,0
4,4.6,3.1,1.5,0.2,0


输出样例：

Id,Species
1,1
2,0
3,2
4,2
5,2
6,2

