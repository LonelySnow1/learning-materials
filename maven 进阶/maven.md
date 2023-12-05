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
* 聚合： 将多个模块组织成一个整体，同时进行项目