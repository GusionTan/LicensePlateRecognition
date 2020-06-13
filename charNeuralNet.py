import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split

numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphbets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z']
chinese = ['zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin',
                'zh_jing', 'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong',
                'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun',
                'zh_zang', 'zh_zhe']

class char_cnn_net:
    def __init__(self):
        self.dataset = numbers + alphbets + chinese #dataset为字母数字汉字集合
        self.dataset_len = len(self.dataset)
        self.img_size = 20 #输入网络图片尺寸
        self.y_size = len(self.dataset) #输出一维数组大小
        self.batch_size = 100 #每次喂100张

        self.x_place = tf.placeholder(dtype=tf.float32, shape=[None, self.img_size, self.img_size], name='x_place')
        self.y_place = tf.placeholder(dtype=tf.float32, shape=[None, self.y_size], name='y_place')
        self.keep_place = tf.placeholder(dtype=tf.float32, name='keep_place') #dropout调用

    def cnn_construct(self):
        x_input = tf.reshape(self.x_place, shape=[-1, 20, 20, 1]) #图片为20*20*1三维，-1为自动计算的图像个数，1也可称通道数

        #Variable变量初始化，random_normal正态分布,stddev样本标准偏差，生成深度为32的3*3*1卷积核（行列分辨率3,3， 通道数为1，核个数为32）
        cw1 = tf.Variable(tf.random_normal(shape=[3, 3, 1, 32], stddev=0.01), dtype=tf.float32)
        cb1 = tf.Variable(tf.random_normal(shape=[32]), dtype=tf.float32)
        #tf.nn.conv2d(输入描述，卷积核描述，核滑动步长，padding='SAME'尺寸不变)
        #relu激活函数f(x)=max(x,0)
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_input,filter=cw1,strides=[1,1,1,1],padding='SAME'),cb1))
        #最大池化提取图片纹理，池化核2*2，滑动步长2
        #tf.nn.max_pool(输入描述，池化核描述，滑动步长，padding)
        #输出shape(20,20,1,32)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #舍弃，将一部分神经元按照一定概率从网络中暂时舍弃，使用时被舍弃的神经元恢复链接
        #tf.nn.dropout(上层输入，暂时舍弃的概率)
        #输出shape(10,10,1,32)
        conv1 = tf.nn.dropout(conv1, self.keep_place)

        cw2 = tf.Variable(tf.random_normal(shape=[3, 3, 32, 64], stddev=0.01), dtype=tf.float32)
        cb2 = tf.Variable(tf.random_normal(shape=[64]), dtype=tf.float32)
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1,filter=cw2,strides=[1,1,1,1],padding='SAME'),cb2))
        #输出shape(10,10,1,64)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #输出shape(5,5,1,64)
        conv2 = tf.nn.dropout(conv2, self.keep_place)

        cw3 = tf.Variable(tf.random_normal(shape=[3, 3, 64, 128], stddev=0.01), dtype=tf.float32)
        cb3 = tf.Variable(tf.random_normal(shape=[128]), dtype=tf.float32)
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2,filter=cw3,strides=[1,1,1,1],padding='SAME'),cb3))
        #输出shape(5,5,1,128)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        #向上取整 输出shape(3,3,1,128)
        conv3 = tf.nn.dropout(conv3, self.keep_place)

        #卷积层的输出是一个x*y*z的矩阵，而全连接层的输入是一个向量，需要把矩阵拉直成向量
        conv_out = tf.reshape(conv3, shape=[-1, 3 * 3 * 128])

        fw1 = tf.Variable(tf.random_normal(shape=[3 * 3 * 128, 1024], stddev=0.01), dtype=tf.float32)
        fb1 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully1 = tf.nn.relu(tf.add(tf.matmul(conv_out, fw1), fb1))
        fully1 = tf.nn.dropout(fully1, self.keep_place)

        fw2 = tf.Variable(tf.random_normal(shape=[1024, 1024], stddev=0.01), dtype=tf.float32)
        fb2 = tf.Variable(tf.random_normal(shape=[1024]), dtype=tf.float32)
        fully2 = tf.nn.relu(tf.add(tf.matmul(fully1, fw2), fb2))
        fully2 = tf.nn.dropout(fully2, self.keep_place)

        fw3 = tf.Variable(tf.random_normal(shape=[1024, self.dataset_len], stddev=0.01), dtype=tf.float32)
        fb3 = tf.Variable(tf.random_normal(shape=[self.dataset_len]), dtype=tf.float32)
        fully3 = tf.add(tf.matmul(fully2, fw3), fb3, name='out_put')
        #输出dataset_len长度的一位数组
        return fully3


    #训练模型
    def train(self,data_dir,save_model_path):
        print('ready load train dataset')
        X, y = self.init_data(data_dir) #初始化数据，X返回图片像素矩阵，y返回对应矩阵标签
        print('success load' + str(len(y)) + 'datas')
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
        #test_size:test数据占比，random_state随机种子；切出test占0.2，train占0.8

        out_put = self.cnn_construct()
        #归一化输出；把一些输入映射为0-1之间的实数，并且归一化保证和为1，因此多分类的概率之和也刚好为1
        predicts = tf.nn.softmax(out_put) 
        #取预测值，根据axis取值的不同返回每行或者每列最大值的索引;axis=0列计算，1行计算
        predicts = tf.argmax(predicts, axis=1)
        #取标签值
        actual_y = tf.argmax(self.y_place, axis=1)
        #准确率
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predicts, actual_y), dtype=tf.float32))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_put, labels=self.y_place))
        #Adam优化算法：是一个寻找全局最优点的优化算法，引入了二次方梯度校正。
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        train_step = opt.minimize(cost)

        with tf.Session() as sess:
            #初始化
            init = tf.global_variables_initializer()
            sess.run(init)
            step = 0
            saver = tf.train.Saver()
            while True:
                train_index = np.random.choice(len(train_x), self.batch_size, replace=False)
                train_randx = train_x[train_index]
                train_randy = train_y[train_index]
                _, loss = sess.run([train_step, cost],
                                   feed_dict={self.x_place:train_randx,self.y_place:train_randy,self.keep_place:0.75})
                step += 1

                if step % 10 == 0:
                    test_index = np.random.choice(len(test_x), self.batch_size, replace=False)
                    test_randx = test_x[test_index]
                    test_randy = test_y[test_index]
                    acc = sess.run(accuracy,feed_dict={self.x_place : test_randx, self.y_place : test_randy,
                                                       self.keep_place : 1.0})
                    print(step, loss)
                    if step % 50 == 0:
                        print('accuracy:' + str(acc))
                    if step % 500 == 0:
                        saver.save(sess, save_model_path, global_step=step)
                    if acc > 0.99 and step > 500:
                        saver.save(sess, save_model_path, global_step=step)
                        break

    #训练测试集，返回预测值列表
    def test(self,x_images,model_path):
        text_list = []
        out_put = self.cnn_construct()
        predicts = tf.nn.softmax(out_put)
        predicts = tf.argmax(predicts, axis=1)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            #恢复模型
            saver.restore(sess, model_path)
            preds = sess.run(predicts, feed_dict={self.x_place: x_images, self.keep_place: 1.0})
            for i in range(len(preds)):
                pred = preds[i].astype(int)
                text_list.append(self.dataset[pred])
            return text_list

    
    def list_all_files(self,root): #传入数据存储地址
        files = []
        list = os.listdir(root) #os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表
        for i in range(len(list)): #此时list[]存储为各文件名
            element = os.path.join(root, list[i]) #路径拼接，
            if os.path.isdir(element): #os.path.isfile()用于判断对象是否为一个文件
                temp_dir = os.path.split(element)[-1] #得到最后一个/之后的内容，例如第一次得到0， list[i]效果相同？
                if temp_dir in self.dataset: #若temp_dir在数字字母汉字集合中
                    files.extend(self.list_all_files(element))#运用递归再次展开子目录，例如展开0、1、2
            elif os.path.isfile(element): #若最终为后缀jpg的文件则加入files中
                files.append(element)
        return files
    #返回的files集合内容:'E:\\test_1\\carIdentityData/cnn_char_train\\0\\gt_339_2.jpg'
    #E:\\test_1\\carIdentityData/cnn_char_train\\zh_yu1\\debug_chineseMat180.jpg
    

    #初始化训练数据地址，返回X图片，y标签
    def init_data(self,dir):
        X = []
        y = [] #y为标签列表；[1,0,0,0,...]
        if not os.path.exists(data_dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(dir) #files为所有图片地址的集合

        for file in files:
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)#CV中默认的彩图通道排列是BGR（蓝绿红）而不是RGB
            #cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片，灰度图为二维
            if src_img.ndim == 3: #ndim为维度；若维度为3则转换失败不做处理
                continue
            resize_img = cv2.resize(src_img, (20, 20)) #转换为20,20大小
            X.append(resize_img)
            # 获取图片文件全目录
            dir = os.path.dirname(file)
            # 获取图片文件上一级目录名
            dir_name = os.path.split(dir)[-1] #此处获得0,1...
            vector_y = [0 for i in range(len(self.dataset))] #建与dataset等长度的0列表
            index_y = self.dataset.index(dir_name) #index() 函数用于从列表中找出某个值第一个匹配项的索引位置；dataset.index(0)=0
            vector_y[index_y] = 1 #将目前识别的字符在列表中的位置置1；vector_y[0]=1
            y.append(vector_y) #作为对应标签列表存入y
        #X为cv对应20,20灰度BGR图片列表,y为对应数组标签
 
        X = np.array(X)#X由列表转换为np矩阵
        y = np.array(y).reshape(-1, self.dataset_len) #将y由列表转换为np矩阵，行数未知-1，列数为dataset长度
        #y转换后矩阵，每行为标签数组，所以列长度固定，行长度由图片数决定
        #X为3维，第一维为单张图片，二三位为长宽；y为2维，第一维为标签数，第二维为内容
        return X, y

    #初始化测试训练数据，返回X图片
    def init_testData(self,dir):
        test_X = []
        if not os.path.exists(test_dir):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(test_dir)
        for file in files:
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            if src_img.ndim == 3:
                continue
            resize_img = cv2.resize(src_img, (20, 20))
            test_X.append(resize_img)
        test_X = np.array(test_X)
        return test_X


if __name__ == '__main__':
    cur_dir = sys.path[0]
    data_dir = os.path.join(cur_dir, 'carIdentityData/cnn_char_train')
    test_dir = os.path.join(cur_dir, 'carIdentityData/cnn_char_test')
    train_model_path = os.path.join(cur_dir, './carIdentityData/model/char_recongnize/model.ckpt')
    model_path = os.path.join(cur_dir,'./carIdentityData/model/char_recongnize/model.ckpt-600')

    train_flag = 0
    net = char_cnn_net()

    if train_flag == 1:
        # 训练模型
        net.train(data_dir,train_model_path) #输入数据地址和模型存储地址
    else:
        # 测试部分:少了BP过程
        test_X = net.init_testData(test_dir)
        text = net.test(test_X,model_path)
        print(text)
