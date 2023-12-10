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

## MP分页查询功能
1. 设置分页拦截器作为Spring管理的bean
2. 执行分页查询
3. 开启日志（可选）

* config/Mpconfig
```java
@Configuration
public class Mpconfig {
    @Bean
    public MybatisPlusInterceptor mybatisPlusInterceptor(){
        //1.定义mp拦截器
        MybatisPlusInterceptor mybatisPlusInterceptor = new MybatisPlusInterceptor();
        //2.在mp拦截器中添加具体拦截器
        mybatisPlusInterceptor.addInnerInterceptor(new PaginationInnerInterceptor());
        return mybatisPlusInterceptor;
    }
}
```

* 测试类
```java
@SpringBootTest
class MyBatisPlusApplicationTests {

  @Autowired
  private UserDao userDao;
  @Test
  void testGetByPage(){
    IPage page = new Page(1,5);
    userDao.selectPage(page,null);
    System.out.println("当前页码数："+ page.getCurrent());
    System.out.println("每页显示数："+ page.getSize());
    System.out.println("一共多少页："+ page.getPages());
    System.out.println("一共多少条："+ page.getTotal());
    System.out.println("数据："+ page.getRecords());
  }
}
```

* application.yml —— 开启日志（可选）
```yaml
mybatis-plus:
  configuration:
    log-impl: org.apache.ibatis.logging.stdout.StdOutImpl
```
---
# DQL编程控制
## 条件查询
### 四种常见查询方式

* lambda的格式，也就是方法引用，有点类似匿名函数
  * User::getId 也就是 (User) -> user.getId()
  * 作用等同于 new User().getId()
```java
@SpringBootTest
class MyBatisPlusApplicationTests {

  @Autowired
  private UserDao userDao;
  @Test
  void testGetAll() {
    //方式一：按条件查询 
//        QueryWrapper qw = new QueryWrapper<>();
//        qw.lt("age",300);
//        List<User> Users = userDao.selectList(qw);
//        System.out.println(Users);

    //方式二：lambda格式按条件查询
//        QueryWrapper<User> qw = new QueryWrapper<>();
//        qw.lambda().lt(User::getAge,200);
//        List<User> Users = userDao.selectList(qw);
//        System.out.println(Users);


    //方式三：lambda格式按条件查询(推荐)
//        LambdaQueryWrapper<User> qw = new LambdaQueryWrapper<>();
//        qw.lt(User::getAge,200);
//        List<User> Users = userDao.selectList(qw);
//        System.out.println(Users);

    //方式四： lambda格式按条件查询(推荐)
    LambdaQueryWrapper<User> qw = new LambdaQueryWrapper<>();
//        qw.lt(User::getAge,600).gt(User::getAge,300); //可以链式编程
    qw.gt(User::getAge, 600).or().lt(User::getAge, 100); //可以链式编程
    List<User> Users = userDao.selectList(qw);
    System.out.println(Users);
  }
}
```

###  空值处理
* 条件参数控制
```java
@SpringBootTest
class MyBatisPlusApplicationTests {

  @Autowired
  private UserDao userDao;

  @Test
  void testGetAll() {
      UserQuery uq = new UserQuery();
//        uq.setAge(100);
//        uq.setAge2(300);
    //null判定,先判断条件是否为true
    LambdaQueryWrapper<User> qw = new LambdaQueryWrapper<>();
    qw.gt(null != uq.getAge(), User::getAge, uq.getAge());
    qw.lt(null != uq.getAge2(), User::getAge, uq.getAge2());
    List<User> Users = userDao.selectList(qw);
    System.out.println(Users);
  }
}
```

## 查询投影


