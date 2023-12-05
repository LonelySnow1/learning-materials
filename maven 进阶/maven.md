# 分模块开发
## 分模块开发的意义
* 将原始模块查分成若干个子模块，方便模块间的相互调用，接口共享

## 分模块开发的步骤
1. 创建Maven工程
   * 将要拆分的功能放入另一个模块中（这里将原来的domain中的Book拆出去了）
2. 书写代码模块
* 目标模块pom文件
```xml
<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>org.example</groupId>
  <artifactId>SSM</artifactId>  
  <packaging>pom</packaging>
  <version>1.0-SNAPSHOT</version>
    <dependencies>
<!--      依赖domain运行-->
      <dependency>
        <groupId>org.example</groupId>
        <artifactId>Maven_1</artifactId>
        <version>1.0-SNAPSHOT</version>
      </dependency>
        ...
    </dependencies>
</project>
```
* 拆分出的pom文件
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>
    <groupId>org.example</groupId>
    <artifactId>Maven_1</artifactId>
    <version>1.0-SNAPSHOT</version>
</project>
```
3. 通过maven指令安装模块到本地仓库
   * 运行拆分出模块的install指令，将其安装到本地
   * 再去运行原项目的compile指令，就可以运行了

注：如果不执行install之指令，在运行原项目的时候就会报错

---
# 依赖管理
## 依赖传递
* ### 依赖具有传递性
  * 直接依赖：在当前项目中通过依赖配置建立的依赖关系
  * 间接依赖：被依赖的资源如果依赖其他资源，当前项目间接依赖其他资源 
* ### 依赖传递冲突问题
  * 路径优先： 当依赖中出现相同的资源时，层级越深，优先级越低；层级越浅，优先级越高
  * 声明优先： 当资源在相同层级被依赖时，配置顺序靠前的覆盖配置顺序靠后的
  * 特殊优先： 当同级配置了相同资源的不同版本，后配置的覆盖先配置的

**如何直观的显示直接/间接依赖**
![img.png](img.png)
![img_1.png](img_1.png)

## 可选依赖
* 可选依赖是指对外隐藏所依赖的资源——不透明,我的依赖不给别人用
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>org.example</groupId>
    <artifactId>Maven_1</artifactId>
    <version>1.0-SNAPSHOT</version>

    <dependencies>
        <dependency>
            <groupId>org.example</groupId>
            <artifactId>Maven_2</artifactId>
            <version>1.0-SNAPSHOT</version>
<!--            可选依赖是隐藏当前工作所依赖的资源，隐藏后的对应资源将不具有依赖传递性 true-隐藏 -->
            <optional>false</optional>
        </dependency>
    </dependencies>
</project>
```
## 排除依赖
* 排除依赖是指主动断开依赖，被排除的资源无需指定版本——别人的依赖我不想用
```xml
<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>org.example</groupId>
  <artifactId>SSM</artifactId>  
  <packaging>pom</packaging>
  <version>1.0-SNAPSHOT</version>
    <dependencies>
<!--      依赖domain运行-->
      <dependency>
        <groupId>org.example</groupId>
        <artifactId>Maven_1</artifactId>
        <version>1.0-SNAPSHOT</version>
<!--          排除依赖是隐藏当前资源对应的依赖关系，无视版本，指定id排除-->
          <exclusions> <!--这里面写要排除的项-->
              <exclusion> 
                  <groupId>org.example</groupId>
                  <artifactId>Maven_2</artifactId>
              </exclusion>
          </exclusions>
      </dependency>
        ...
  </dependencies>
</project>
```
---
# 继承和聚合
## 聚合
* 聚合： 将多个模块组织成一个整体，同时进行项目构建的过程称为聚合
* 聚合工程 ：通常是一个不具有业务功能的“空”工程（有且仅有一个pom）文件
* 作用： 使用聚合工程可以将多个工程编组，通过对聚合工程进行构建，实现对所包含的模块进行同步构建
  * 当工程中某个模块发生更新（变更）时，必须保障工程中与已更新模块的关联模块同步更新，
此时可以使用聚合工程来解决批量模块同步构建的问题
### 聚合开发步骤
1. 创建maven模块，设置打包类型为pom
2. 设置当前聚合工程所包含的子模块名称
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>org.example</groupId>
    <artifactId>Maven</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>pom</packaging>

<!--    设置管理的模块名称 -->
    <modules>
        <module>../SSM</module>
        <module>../Maven_1</module>
        <module>../Maven_2</module>
    </modules>
</project>
```
## 继承
* 概念： 继承描述的是两个工程间的关系，与java中的继承相似，子工程可以继承父工程中的配置信息，常见于依赖关系的继承
* 作用：
  * 简化配置
  * 减少版本冲突

### 继承步骤
1. 创建maven模块，打包方式设为pom（跟聚合相同）
2. 在父工程的pom文件中配置依赖关系（子工程将沿用父工程中的依赖关系）
3. 在父工程中配置可选的依赖
* 父工程
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>org.example</groupId>
    <artifactId>Maven</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>pom</packaging>
    
<!--继承必选依赖-->
<dependencies>
   <dependency>
       <groupId>org.springframework</groupId>
       <artifactId>spring-webmvc</artifactId>
       <version>5.2.10.RELEASE</version>
   </dependency>
</dependencies>


<!-- 可选依赖管理-->
    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>junit</groupId>
                <artifactId>junit</artifactId>
                <version>4.12</version>
                <scope>test</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>
</project>
```
* 子工程
```xml
<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>org.example</groupId>
  <artifactId>SSM</artifactId>  
  <packaging>war</packaging>
  <version>1.0-SNAPSHOT</version>
    
<!--    配置当前工程继承自parent工程-->
    <parent>
        <groupId>org.example</groupId>
        <artifactId>Maven</artifactId>
        <version>1.0-SNAPSHOT</version>
<!--        父工程路径（可不填）-->
        <relativePath>../Maven/pom.xml</relativePath>
    </parent>

    <dependencies>
      <dependency>
        <groupId>org.example</groupId>
        <artifactId>Maven_1</artifactId>
        <version>1.0-SNAPSHOT</version>
      </dependency>
<!--        添加可选依赖（不需要版本号）-->
        <dependency>
            <groupId>junit</groupId>
            <artifactId>junit</artifactId>
            <scope>test</scope>
        </dependency>
  </dependencies>
</project>
```

## 继承和聚合的区别
* 作用：
  * 聚合用于快速构建项目
  * 继承用于快速配置
* 相同点：
  * 聚合与继承的pom.xml文件打包方式均为pom，可以将两种关系只做到同一个pom文件中
  * 聚合与继承均属于设计型模块，并无实际的模块内容
* 不同点：
  * 聚合是在当前模块中配置关系，聚合可以感知到参与聚合的模块有哪些
  * 继承是在子模块中配置关系，父模块无法感知哪写子模块继承了自己

---
# 属性
## pom文件中使用属性
* 定义属性-properties
* 使用属性-${}
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>org.example</groupId>
    <artifactId>Maven</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>pom</packaging>
    <dependencies>
        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-webmvc</artifactId>
            <version>${spring.version}</version>
        </dependency>

        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-jdbc</artifactId>
            <version>${spring.version}</version>
        </dependency>

        <dependency>
            <groupId>org.springframework</groupId>
            <artifactId>spring-test</artifactId>
            <version>${spring.version}</version>
        </dependency>
    </dependencies>

<!--    定义属性-->
    <properties>
        <spring.version>5.2.9.RELEASE</spring.version>
    </properties>
</project>
```
## 资源文件中引用属性
1. 定义属性
2. 在pom中设属性生效范围
3. 在资源文件中调用属性
* pom.xml
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>org.example</groupId>
    <artifactId>Maven</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>pom</packaging>
  
<!--    定义属性-->
    <properties>
        <jdbc.url>jdbc:mysql:///ssm_db?useSSL=false</jdbc.url>
    </properties>
<!--    让指定路径下的文可以使用pom中的属性-->
    <build>
        <resources>
            <resource>
                <directory>${project.basedir}/src/main/resources</directory><!--这里用了一个maven的系统属性-->
                <filtering>true</filtering>
            </resource>
        </resources>
    </build>
</project>
```
* jdbc.properties
  * 可定义多个，这里只定义一个用于举例说明 
```
jdbc.driver=com.mysql.jdbc.Driver
jdbc.url=${jdbc.url}
jdbc.username=root
jdbc.password=123456
```
4. 配置maven打war包时不检查web.xml是否存在
```xml
<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>org.example</groupId>
  <artifactId>SSM</artifactId>  
  <packaging>war</packaging>
  <version>1.0-SNAPSHOT</version>

  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-war-plugin</artifactId>
        <version>3.4.0</version>
        <configuration>
          <failOnMissingWebXml>false</failOnMissingWebXml>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
```
## 其他属性
![img_2.png](img_2.png)

##  版本管理
* 工程版本
  * SNAPSHOT（快照版本）
    * 项目开发过程中临时输出的版本，称为快照版本
    * 快照版本会随着开发的进展不断更新
  * RELEASE（发布版本）
    * 项目开发到进入阶段里程碑后，项团队外发布较为稳定的般，这种版本所对应的构建文件是稳定的，即便进行功能的后续开发，也不会改变当前发布版本内容，
这种版本称为发布版本
* 发布版本
  * alpha版
  * beta版
  * 纯数字版

---
# Maven多环境配置与应用
## 多环境开发
* maven提供配置多种环境的设定，帮助开发者使用过程中快速切换环境

1. 配置多环境
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>org.example</groupId>
    <artifactId>Maven</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>pom</packaging>
<!--    配置多环境-->
    <profiles>
<!--     开发环境-->
        <profile>
            <id>env_dep</id>
            <properties>
                <jdbc.url>jdbc:mysql:///ssm_db?useSSL=false</jdbc.url>
            </properties>
        </profile>
<!--    生产环境-->
        <profile>
            <id>env_pro</id>
            <properties>
                <jdbc.url>jdbc:mysql://127.2.2.2:3306/ssm_db?useSSL=false</jdbc.url>
            </properties>
<!--            设定为默认启用的环境-->
            <activation>
                <activeByDefault>true</activeByDefault>
            </activation>
        </profile>
<!--    测试环境-->
        <profile>
            <id>env_test</id>
            <properties>
                <jdbc.url>jdbc:mysql://127.3.3.3:3306/ssm_db?useSSL=false</jdbc.url>
            </properties>
        </profile>
    </profiles>

<!--    让指定路径下的文可以使用pom中的属性-->
    <build>
        <resources>
            <resource>
                <directory>${project.basedir}/src/main/resources</directory>
                <filtering>true</filtering>
            </resource>
        </resources>
    </build>
</project>
```

2. 使用多环境

mvn 指令 -p 环境定义id  范例：

```mvn install -p pro_env```

## 跳过测试
* 应用场景
  * 功能更新中且没有开发完成
  * 快速打包
  * ......

* 使用方式
  * idea自带的maven功能
  * maven插件
  * maven语句
  
1. idea自带的maven功能

![img_3.png](img_3.png)

2.maven插件（细粒度控制跳过测试）
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>org.example</groupId>
    <artifactId>Maven</artifactId>
    <version>1.0-SNAPSHOT</version>
    <packaging>pom</packaging>

<!--    让指定路径下的文可以使用pom中的属性-->
    <build>
        <resources>
            <resource>
                <directory>${project.basedir}/src/main/resources</directory>
                <filtering>true</filtering>
            </resource>
        </resources>
<!--      跳过测试-->
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>2.22.1</version>
                <configuration>
                    <skipTests>false</skipTests><!--是否完全跳过测试-->
                    <!--排除掉不参与测试的内容-->
                    <excludes>
                        <exclude>**/BookServiceTest.java</exclude>
                    </excludes>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```
3. maven指令

mvn 指令 -D skipTests

```mvn install -D skipTests ```

----
# 私服
## 私服简介
* 私服是一台独立的服务器，用于解决团队内部的资源共享与资源同步问题

* Nexus 
  * Sonatype公司的一款maven私服产品
  * 