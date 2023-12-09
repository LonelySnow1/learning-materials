# 初识MyBatisPlus
## 入门案例
SpringBoot整合MyBatis（复习）
1. 创建SpringBoot工程
2. 勾选使用的到的技术
3. 设置dataSource相关属性（JDBC参数）
4. 定义数据层接口映射配置

SpringBoot整合MyBatisPlus(简称mp)
1. 创建新模块，选择Spring初始化，并配置模块相关基础信息（SQL中只需要勾选MySQL Driver）
2. **手动添加mp起步依赖**
3. 设置jdbc参数
4. 制作实体类和表结构
5. **定义数据接口，继承BaseMapper<>**

* pom.xml
  * 由于MyBaits的起步依赖中mybatis-spring的版本过低，所以在SpringBoot3.X的版本上无法运行
  * 所以需要再配置一个高版本的mybatis-spring来解决maven的依赖传递问题
```xml
<dependencies>
        <dependency>
            <groupId>com.baomidou</groupId>
            <artifactId>mybatis-plus-boot-starter</artifactId>
            <version>3.5.4</version>
        </dependency>

        <dependency>
            <groupId>com.alibaba</groupId>
            <artifactId>druid</artifactId>
            <version>1.2.20</version>
        </dependency>

        <dependency>
            <groupId>org.mybatis</groupId>
            <artifactId>mybatis-spring</artifactId>
            <version>3.0.3</version>
        </dependency>
    </dependencies>
```

* UserDao
```java
@Mapper
public interface UserDao extends BaseMapper<ABC> { }
```

* domain/ABC
  * 如果表名和实体类名不一样，可以加上@TableName("对应表名")的注解
```java
@TableName("user")
public class ABC {
    private Long id;
    private String name;
    private String password;
    private Integer age;
    private String tel;
    //......
    //略去getter setter方法以及toString方法
}
```

* application.yml
```yaml
spring:
  datasource:
    type: com.alibaba.druid.pool.DruidDataSource
    driver-class-name: com.mysql.cj.jdbc.Driver
    url: jdbc:mysql://localhost:3306/mybatisplus_db
    username: root
    password: 123456
```

* 测试类
```java
@SpringBootTest
class MyBatisPlusApplicationTests {

    @Autowired
    private UserDao userDao;

    @Test
    void testGetAll() {
        List<ABC> Users = userDao.selectList(null);
        System.out.println(Users);
    }
}
```

## MyBatisPlus概述
* MyBatisPlus(简称MP)是基于MyBatis框架基础上开发的增强型工具，旨在简化开发，提高效率
* 国内组织开发的技术

![img.png](img.png) 

---
# 标准数据层开发
## mp提供的接口
![img_1.png](img_1.png)
## Lombok
* lombok,一个Java类库，提供了一组注解，简化POJO实体类的开发
* 常用@Data:注解在类上，提供get、set、equals、hashCode、canEqual、toString、无参构造方法，**没有有参构造**

* pom.xml
```xml
<dependency>
   <groupId>org.projectlombok</groupId>
   <artifactId>lombok</artifactId>
   <scope>provided</scope>
</dependency>
```

* 实体类
```java
@Data
//如果表名和实体类名不一样，可以加上@TableName("对应表名")的注解
@TableName("user")
public class User {
    private Long id;
    private String name;
    private String password;
    private Integer age;
    private String tel;
}
```