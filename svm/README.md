任务描述：

手写数字识别是生活中尤其常见的机器学习任务，给出一份手写数字训练数据集，训练一个SVM模型并对测试集进行手写数字识别。



输入数据：

在train/目录下包含多个txt文件，其中每个文件表示一个用01矩阵表示的手写数字，文件名中下划线前面的数字代表手写数字的值（如2_167.txt表示手写数字为2；3_13.txt表示手写数字为3，训练数据集可在教学资料中下载，文件名为svm_train.tar）。

在test/目录下也包含多个txt文件，只不过test文件夹下面的txt文件无法从文件名得知手写数字的值（文件名：0.txt~945.txt），需要根据训练好的模型进行预测。



输出数据：

程序需要生成一个result.csv文件，用于保存程序对test中各个txt文件中手写数字值的预测结果。第一行固定为num，之后每一行为一个数值，代表预测值，表示程序对test中对应txt文件的预测结果。



评价标准：

测试集上的准确率。



输入样例：

00000000000000000011110000000000
00000000000000001111111100000000
00000000001000111111111100000000
00000000011111111111111110000000
00000000111111111111111110000000
00000000111111111111111110000000
00000000111111111111111110000000
00000000111111111111111111000000
00000001111111111101111111000000
00000000111111000000001111000000
00000001111110000000011111000000
00000001111100000000011111000000
00000001111100000000011111000000
00000001111100000000001111000000
00000001111100000000001111000000
00000001111100000000001111000000
00000001111100000000001111000000
00000001111100000000001111000000
00000001111100000000001111000000
00000001111100000000001111000000
00000001111100000000011111000000
00000000111100000000011111000000
00000000011110000000011111000000
00000000111100000001111110000000
00000000111110000111111000000000
00000000111111111111111000000000
00000000011111111111111000000000
00000000011111111111111000000000
00000000011111111111110000000000
00000000001111111111110000000000
00000000000111111111000000000000
00000000000000111100000000000000


输出样例：

num
0
1
2
3
4