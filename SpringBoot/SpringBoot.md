# SpringBoot简介 

* SpringBoot是由Pivotal团队提供的全新框架，其设计目的是用来简化Spring应用的初始搭建以及开发的过程
* 原生开发SpringMVC程序的过程
    1. 导入坐标
    2. Web核心配置类
    3. SpringMvc配置类
    4. Controller类实现功能
## 入门案例
### 创建SpringBoots入门程序步骤
  1. 创建新模块，选择Spring Initializr 并配置相关的基础信息——注意对应的JDK版本 
     * SpringBoot3.X 强制要求JDK版本不低于17
     * JDK8 可使用SpringBoot2.X
  2. 勾选SpringWeb
  3. 开发控制器controller类
  4. 运行自动生成的Application类
     * 可以看见Tomcat的端口号和版本信息
  
![img.png](img.png)
![img_1.png](img_1.png)
![img_3.png](img_3.png)

### Spring程序与SpringBoot程序对比
* pom文件中的坐标
  * Spring ： 手动添加
  * Springboot： 勾选添加
* Web配置类
  * Spring ： 手动添加
  * Springboot： 无
* Spring/SpringMvc配置类
  * Spring ： 手动添加
  * Springboot： 无
* pom文件中的坐标
  * Spring ： 手动添加
  * Springboot： 勾选添加


---
