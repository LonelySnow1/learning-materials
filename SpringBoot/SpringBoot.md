# SpringBoot简介 

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

**注：开发Spring程序需要确保联网且能够加载到陈虚谷框架结构，不使用idea可以直接去官网(start.spring.io)创建项目**

### Spring项目快速启动

1. 对SpringBoot项目进行打包（执行Maven构建指令package）
2. 执行启动指令 ```java -jar 名称.jar```
   * jar包的运行需要maven插件，打包时注意
```xml
    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>
```

---
## SpringBoot概述
* SpringBoot是由Pivotal团队提供的全新框架，其设计目的是用来简化Spring应用的初始搭建以及开发的过程
* Spring程序缺点
  * 配置繁琐
  * 依赖设置繁琐
* SpringBoot程序优点
  * 自动配置
  * 起步依赖（简化依赖配置）
  * 辅助功能（内置服务器，...）
* ### 起步依赖
  * starter
    * SpringBoot中常见的项目名称，定义了当前项目使用的所有坐标，以达到**减少依赖配置**的目的
  * parent
    * 所有Spring项目都要继承的项目，定义了若干个坐标的版本号（依赖管理，而非依赖），以达到**减少依赖冲突**的目的
  * 实际开发
    * 使用任意坐标时，仅书写GAV中的G和A，V由SpringBoot提供
    * 如发生坐标错误，再指定version（小心版本冲突）
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.0</version>
    </parent>
  
    <groupId>com.Lonelysnow</groupId>
    <artifactId>SpringBoot</artifactId>
    <version>0.0.1-SNAPSHOT</version>
  
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>
    </dependencies>
</project>
```

* ### 引导类
```java
@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}
```
* SpringBoot在创建项目时，采用jar的打包方式
* SpringBoot的引导类时项目的入口，运行main方法就可以启动项目
* 变更起步依赖项非常便捷
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.0</version>
    </parent>
    <groupId>com.Lonelysnow</groupId>
    <artifactId>SpringBoot</artifactId>
    <version>0.0.1-SNAPSHOT</version>
  
    <dependencies>
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
            <exclusions>
                <exclusion>
                    <groupId>org.springframework.boot</groupId>
                    <artifactId>spring-boot-starter-tomcat</artifactId>
                </exclusion><!--排除tomcat服务器-->
            </exclusions>
        </dependency>

        <dependency>
            <groupId>org.springframework.boot</groupId><!--改用jetty服务器-->
            <artifactId>spring-boot-starter-jetty</artifactId>
        </dependency>
    </dependencies>
  
</project>
```

---

# 基础配置
## 配置格式
SpringBoot提供了多种属性的配置方式
* application.properties
```properties
server.port=80
```
* application.yml
```yaml
server:
  port: 80
```
* application.yaml
```yaml
server:
  port: 80
```
如果yaml/yml文件没有自动补全，代码提示，可以在项目结构中将文件添加到项目的配置文件中

![img_2.png](img_2.png)

* Springboot配置文件的加载顺序
  * application.properties
  * application.yml
  * application.yaml

## yaml
yaml(YAML Ain't Markup Language),一种数据序列化格式
* ### 优点
  * 容易阅读
  * 容易与脚本语言交互
  * 以数据为核心，重数据轻格式
* ### YAML文件拓展名
  * .yml (主流)
  * .yaml
* ### 语法规则：
  * 大小写敏感
  * 属性层级关系使用多行描述，每行结尾使用冒号结束
  * 使用缩进表示层级关系，同层级左侧对齐，只允许使用空格（不允许使用tab）
  * 属性值前面加空格（属性名与属性值之间使用冒号+空格作为分割）
  * \# 表示注释 

**核心格式：冒号后加空格**
* 数组数据在数据书写位置的下方使用减号作为数据的开始符号，每行书写一个数据，减号与数据间空格分割
```yaml
likes:
  - a
  - b
  - c
  - d
```

### 读取数据的方式
