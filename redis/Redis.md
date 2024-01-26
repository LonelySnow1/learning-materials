
# Redis 入门概述
## Redis是什么：
Remote Dictionary Server（远程字典服务器）
使用ANSIC语言编写遵守BSD协议，**是一个高性能的Key-Value数据库**，提供了丰富的数据结构，例如String、Hash、List、Set、SortedSet等等。数据是存在内存中的，同时Redis支持事务、持久化、LUA脚本、发布/订阅、缓存淘汰、流技术等多种功能，提供了主从模式、Redis Sentinel和Redis Cluster集群架构方案

![img.png](img.png)
## 主流功能与应用：
1. 分布式缓存，挡在MySQL数据库之前的带刀护卫 
![img_1.png](img_1.png)
   * 对比传统数据库（MySQL）
     * Redis是key-value数据库的一种，MySQL是关系型数据库
     * Redis数据操作主要在内存，MySQL主要存储在磁盘
     * Redis在某一场景使用要优于MySQL
     * 两者并不是竞争与相互替换的关系，而是共用和配合使用

2. 内存存储和持久化
   * 支持异步将内存中的数据写到硬盘上，同时不影响继续服务
3. 高可用架构搭配
   * 单机、主从、哨兵、集群
4. 缓存穿透、击穿、雪崩
5. 分布式锁
6. 队列
   * Redis提供list和set操作，使得Redis能作为一个很好的消息队列平台使用
   * 可以应用于秒杀等
7. 排行榜+点赞

## 总体功能概述：
![img_2.png](img_2.png)
### 优势：
* 性能极高-Redis读的速度是110000次/秒，写的速度是81000次/秒
* Redis数据类型丰富，不仅仅支持简单的Key-Value类型的数据，同时还提供list，set，zset，hash等数据结构的存储
* Redis支持数据的持久化，可以将内存中的数据保持在磁盘中，重启的时候可以再次加载进行使用
* Redis支持数据的备份，即master-slave模式的数据备份

### 命令参考：http://doc.redisfans.com/

# Redis安装
## 安装环境要求和准备
* 必须是64位linux系统
* 需要有gcc编译环境
* Redis版本必须为6.0.8及以上
### 操作步骤
1. 查看linux系统操作位数  ``` getconf LONG_BIT```
2. 安装gcc编译环境 ```yum -y install gcc-c++```
3. 将Redis安装包上传至linux中的opt目录下
4. 解压 ```tar -zxvf redis-7.2.4.tar.gz```
5. 进入对应路径 ```cd /opt/redis-7.2.4```
6. 安装插件 ```make && make install```
7. 查看默认安装路径 ```cd /usr/local/bin``` ```ll```
   * redis-benchmark:性能测试工具 
   * redis-check-aof:修复有问题的AOF文件
   * redis-check-dump:修复有问题的dump.rdb文件
   * **redis-cil: 客户端，操作入口**
   * redis-sentinel:redis集群使用
   * **redis-server: Redis服务启动命令**
8. 备份redis.conf ```mkdir /myredis``` ```cp redis.conf /myredis/redis7.conf```
9. 修改原redis.conf ```vim /opt/redis-7.2.4/redis.conf ```
   * ![img_3.png](img_3.png)