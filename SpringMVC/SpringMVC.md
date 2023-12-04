# 初始SpringMVC
## SpringMVC概述
* springMVC是一种表现层框架技术
* SpringMVC是一种基于java实现MVC模型的轻量级web框架
* 优点
  * 使用简单，开发便捷（相比于Servlet）
  * 灵活性强

## 入门案例
### 注解
* ####  @Controller
  * 类型：类注解
  * 位置：SpringMVC控制器类上方
  * 作用：设定核心控制器bean


* #### @RequestMapping
  * 类型:方法注解
  * 位置:SpringMVC控制器方法请求访问路径
  * 作用:设置当前控制器方法请求访问路径
  * 参数:请求访问路径


* #### @ResponseBody
  * 类型：方法注解
  * 位置：SpringMVC控制器方法定义上方
  * 作用：设置当前控制器方法响应内容为当前返回值，无需解析
```java
//2.定义controller
//2.1使用@Controller定义bean
@Controller
public class UserController {
    //2.2设置当前操作的访问路径
    @RequestMapping("/save")
    //2.3设置当前操作的返回值类型
    @ResponseBody
    public String save(){
        System.out.println("user save ...");
        return "{'module':'springmvc'}";
    }
}
```

### 开发总结（1+N）
* 一次性工作：
  * 创建工程，设置服务器，加载工程
  * 导入坐标
  * 创建web容器启动类，设置MVC配置，请求拦截路径
  * 设置核心配置类


* 多次工作：
  * 定义处理的控制器类
  * 定义处理请求的控制器方法，配置映射路径（@RequestMapping）返回json数据（@ResponseBody）

```java
//web容器启动配置类
import org.springframework.web.context.WebApplicationContext;
import org.springframework.web.context.support.AnnotationConfigWebApplicationContext;
import org.springframework.web.servlet.support.AbstractDispatcherServletInitializer;

//4.定义一个servlet容器的启动配置类，在里面添加spring的配置
public class ServletContainersInitConfig extends AbstractDispatcherServletInitializer {
  //加载Springmvc容器配置
  @Override
  protected WebApplicationContext createServletApplicationContext() {
    AnnotationConfigWebApplicationContext ctx = new AnnotationConfigWebApplicationContext();
    ctx.register(SpringMvcConfig.class);
    return ctx;
  }
  //设置哪些请求归属Springmvc处理
  @Override
  protected String[] getServletMappings() {
    return new String[]{"/"};
  }
  //加载Spring容器配置
  @Override
  protected WebApplicationContext createRootApplicationContext() {
     AnnotationConfigWebApplicationContext ctx = new AnnotationConfigWebApplicationContext();
     ctx.register(SpringConfig.class);
     return ctx;
  }
}
```

```java
//MVC配置类

import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;

//3.创建SpringMvc的配置文件，加载controller对应的bean
@Configuration
@ComponentScan("com.LonelySnow.controller")
public class SpringMvcConfig { }
```

### 入门案例工作流程分析
启动服务器初始化流程
1. 服务器启动，执行web容器启动配置类，初始化web容器
2. 执行配置类中的createServletApplicationContext方法，创建WebApplicationContext对象
3. 加载MVC配置类
4. 执行@ComponentScan，加载对应的bean
5. 加载控制器类，映射对应的方法
6. 执行配置类中的getServletMappings方法，定义所有的请求都通过SpringMVC

单次请求过程：
1. 发送请求
2. 交由MVC处理
3. 解析请求路径
4. 匹配对应方法
5. 执行方法
6. 有@ResponseBody就将返回值返回给请求方
---
## controller加载控制与业务bean加载控制
* springMVC相关bean 
* spring控制的bean
  * 业务bean Service
  * 功能bean DataSource等

**功能不同，如何避免Spring错误加载到SpringMVC的bean？**

——加载spring控制的bean的时候排除掉SpringMVC控制的Bean

* SpringMVC相关bean的加载控制
  * 均在对应的controller包内
* Spring相关bean加载控制
  1. 扫描范围排除掉controller包
  2. 扫描范围精确到对应的service包、dao包 
  3. 不区分，都加载到同一个环境中

### 注解——@ComponentScan
* @ComponentScan
  * 类型：类注解
  * 属性：
    * excludeFilters:排除指定的bean【需要指定type与classes】
    * includeFilters:加载指定的bean【需要指定type与classes】
```java
@Configuration
//方法一 只加载指定包
@ComponentScan({"com.LonelySnow.service", "com.LonelySnow.dao"})
//方法二 排除指定包
@ComponentScan(value = "com.LonelySnow",
        excludeFilters = @ComponentScan.Filter(
                type = FilterType.ANNOTATION,//按注解过滤
                classes = Controller.class//过滤@Controller的注解
        )
)
public class SpringConfig { }
```

**注意事项**
在Main方法中使用bean的时候，若使用register，则需要额外refresh()
```java
public class App {
    public static void main(String[] args) {
//        AnnotationConfigApplicationContext ctx  = new AnnotationConfigApplicationContext(SpringConfig.class);
        AnnotationConfigApplicationContext ctx  = new AnnotationConfigApplicationContext();
        ctx.register(SpringConfig.class);
        ctx.refresh();
        System.out.println(ctx.getBean(UserController.class));
    }
}
```
### 简化web配置类
```java
public class ServletContainersInitConfig extends AbstractAnnotationConfigDispatcherServletInitializer {

    @Override
    protected Class<?>[] getRootConfigClasses() {
        return new Class[]{SpringConfig.class};
    }

    @Override
    protected Class<?>[] getServletConfigClasses() {
        return new Class[]{SpringMvcConfig.class};
    }

    @Override
    protected String[] getServletMappings() {
        return new String[]{"/"};
    }
}
```
---
# 请求与响应 
**团队多人开发，每人设置不同的请求路径，冲突问题如何解决？**

—— 设置模块名作为请求路径前缀

## 请求映射路径
### 注解——@RequestMapping
* @RequestMapping
  * 类型：方法注解，类注解
  * 位置：位于SpringMVC控制器方法定义的商法
  * 作用：设置当前控制器方法的请求访问路径，如果设置在类上就是设置路径前缀
```java
@Controller
@RequestMapping("/book") // 设置访问路径前缀
public class BookController {
    //2.2设置当前操作的访问路径
    @RequestMapping("/save")
    //2.3设置当前操作的返回值类型
    @ResponseBody
    public String save(){
        System.out.println("book save ...");
        return "{'module':'book'}";
    }
}  
```

## 请求方式
### get请求
直接链接后缀参数就行
```
http://localhost/user/commonParam?name=lonelysnow&age=12
```
### post请求
form表单post请求传参，表单参数名与形参变量名相同即可

在Body中的x-www-form-urlencoded设置参数

## 处理请求（不区分get，post）
```java
@Controller
@RequestMapping("/user")
public class UserController {
  @RequestMapping("/commonParam")
  @ResponseBody
  public String commonParam(String name,int age){
    System.out.println("参数传递 ===>" + name);
    System.out.println("参数传递 ===>" + age);
    return "{'name':'commonParam'}";
  }
}
```


### 处理post中文请求乱码问题
为web容器添加过滤器并指定字符集，Spring-web包中提供了专用的字符过滤器
```java
public class ServletContainersInitConfig extends AbstractAnnotationConfigDispatcherServletInitializer {
    //在web配置类中添加如下代码
    @Override
    protected Filter[] getServletFilters() {
    CharacterEncodingFilter filter = new CharacterEncodingFilter();
    filter.setEncoding("UTF-8");
    return new Filter[]{filter};
  }
}
```
### 请求参数-普通类型
* 参数种类
  * 普通参数
    * url地址传参，参数名与形参变量名相同，定义形参即可传参
    * 若不匹配，使用@RequestParam直接指定参数名
  * POJO参数：实体类
    * 请求参数名与形参属性名相同，定义pojo类型即可接收参数
  * 嵌套pojo参数
    * 请求参数名与形参对象属性名相同，按照对象层次结构关系即可接收嵌套pojo属性参数
    * ``` http://localhost/user/pojo?name=lonelysnow&age=13&add.name=beijing&add.city=beijing ```
  * 数组参数
    * 请求参数名与形参对象属性名相同起额请求为多个，定义数组类型形参即可接收参数
    * ```http://localhost/user/hobby?likes=1&likes=2&likes=3```
  * 集合
    * 与数组传参时相同，接收时需要在集合前加上@RequestParam绑定参数关系
    * 不绑定会默认将集合当作对象对立，由于集合是个接口，找不到init方法，报错

  ### 注解——@RequestParam
* @RequestParam
  * 类型：形参注解
  * 位置：MVC控制器方法形参定义之前
  * 作用：绑定请求参数与处理器方法间的关系
  * 参数：
    * required： 是否为必传参数
    * defaultValue：参数默认值
```java
@Controller
@RequestMapping("/user")
public class UserController {
  @RequestMapping("/commonParam")
  @ResponseBody
  public String commonParam(@RequestParam("username") String name, int age) {
    System.out.println("参数传递 ===>" + name);
    System.out.println("参数传递 ===>" + age);
    return "{'name':'commonParam'}";
  }
}
```
### 请求参数（JSON）
**发送**

Body->raw->JSON

**接收**

1. 
2. 在pom中导入对应坐标
```xml
      <dependency>
          <groupId>com.fasterxml.jackson.core</groupId>
          <artifactId>jackson-databind</artifactId>
          <version>2.15.0</version>
      </dependency>
```
2. 在SpringMvcConfig配置类中加入注解 @EnableWebMvc ——自动转换功能的支持
3. 在控制器方法形参前加入@RequestBody

### 日期类型参数传递
对于不同格式的日期参数可以使用@DateTimeFormat来指定接收方式

#### 注解——@DateTimeFormat
* @DateTimeFormat
  * 类型：形参注释
  * 位置：SpringMVC控制器方法形参前面
  * 作用：设定日期时间型数据格式
  * 属性：pattern：日期时间格式字符串

```java
@Controller
@RequestMapping("/user")
public class UserController {
  @RequestMapping("/Date")
  @ResponseBody
  public String Date(Date date,
                     @DateTimeFormat(pattern = "yyyy-MM-dd") Date date1,
                     @DateTimeFormat(pattern = "yyyy-MM-dd HH:mm:ss") Date date2){
    System.out.println("参数传递 ===>" +date);
    System.out.println("参数传递\"yyyy-MM-dd\" ===>" +date1);
    System.out.println("参数传递\"yyyy-MM-dd HH:mm:ss \" ===>" +date2);
    return "{'name':'Date'}";
  }
}
```
#### 类型转换器
* Converter接口
  * 请求参数年龄数据
  * 日期格式转换
* EnableWebMvc功能之一：根据类型匹配对应的类型转化器

## 响应
* 响应页面 —— 只有在不加请求路径前缀的时候可以用
* 响应数据
  * 文本数据
  * json数据——直接return 对象就可以，自动转json

类型转换器 

httpMessageConverter

---

# REST风格
## 简介
### REST：表现形式状态转化
  * 传统风格资源描述形式
    * ```https://localhost/user/getById?id=1```
    * ```https://localhost/user/save```
  * REST风格描述形式
    * ```https://localhost/user/1```
    * ```https://localhost/user```
### 优点
  * 隐藏资源的访问行为，无法通过地址得知对资源的何种操作
  * 书写简化
### REST风格简介
  * 按照REST风格访问资源时使用行为动作区分对资源进行了何种操作
    * ```https://localhost/user```     GET（查询）
    * ```https://localhost/user```     PUT（修改）
    * ```https://localhost/user```     POST（新增/保存）
    * ```https://localhost/user```     DELETE（删除）
  * 根据REST风格对资源进行访问称为RESTful

注：
  * 上述行为是约定方式，而不是规范，可以打破
  * 描述的模块名称通常使用复数，也就是加s，表示此类资源

## 入门案例
### 流程
1. 设定http请求动作（动词）
   * 增——post
   * 删——delete
   * 改——put
   * 查——get
2. 设定路径参数（路径变量）

### 注解
* #### @RequestMapping
  * 属性：
    * value：访问路径
    * method：http请求动作，标准动作
* #### @PathVariable
  * 作用：绑定路径参数与处理器方法形参间的关系，要求路径参数名与形参名一一对应

```java
@Controller
public class UserController {

    @RequestMapping(value = "/users",method = RequestMethod.POST)
    @ResponseBody
    public String save(){
        System.out.println("user save ...");
        return "{'module':'user'}";
    }

    @RequestMapping(value = "/users/{id}",method = RequestMethod.DELETE)
    @ResponseBody
    public String delete(@PathVariable Integer id){
        System.out.println("user delete ..."+id);
        return "{'module':'delete'}";
    }
}
```

## 三个注解参数的区分
——@RequestBody  @RequestParam @PathVariable
* 区别
  * @RequestParam 用于接收url地址传参或表单传参
  * @RequestBody 用于接收json数据
  * @PathVariable 用于接收路径参数，使用{参数名称}描述路径参数
* 应用
  * 后期开发中，发送请求超过一个时，以json为主，@RequestBody 
  * 非json数据 @RequestParam
  * 采用RESTful开发，且参数数量较少，使用@PathVariable可传递id值

## RESTful快速开发
### 注解
* #### @RestController
  * 类型：类注解
  * 位置：在RESTful开发控制器类定义上方
  * 作用：等同于@Controller + @ResponseBody
* #### @GetMapping @PostMapping ...
  * 方法注解
  * 作用：代替原有的@RequestMapping
```java
//@Controller
//@ResponseBody
@RestController
@RequestMapping(value = "/books")
public class BookController {

  //    @RequestMapping(method = RequestMethod.POST)
  @PostMapping
  public String save() {
    System.out.println("book save ...");
    return "{'module':'springmvc'}";
  }

  //    @RequestMapping(value = "/{id}",method = RequestMethod.DELETE)
  @DeleteMapping("/{id}")
  public String delete(@PathVariable Integer id) {
    System.out.println("book delete ..." + id);
    return "{'module':'delete'}";
  }
}
```
---
# SSM整合
## 整合流程
  1. 创建工程
  2. SSM整合
     * Spring
       * SpringConfig
     * MyBatis
       * MybatisConfig
       * JdbcConfig
       * jdbc.properties
     * SpringMVC
       * ServletConfig
       * SpringMvcConfig
  3. 功能模块
     * 表与实体类
     * dao（接口+自动代理）
     * service（接口+实现类）
       * 业务层接口测试（整合JUnit）
     * controller
       * 表现层接口测试（PostMan）

### 示例代码
**配置**
* pom.xml
```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>
  <groupId>org.example</groupId>
  <artifactId>SSM</artifactId>
  <packaging>war</packaging>
  <version>1.0-SNAPSHOT</version>
  <dependencies>
    <dependency>
      <groupId>org.springframework</groupId>
      <artifactId>spring-webmvc</artifactId>
      <version>5.2.10.RELEASE</version>
    </dependency>

    <dependency>
      <groupId>org.springframework</groupId>
      <artifactId>spring-jdbc</artifactId>
      <version>5.2.10.RELEASE</version>
    </dependency>

    <dependency>
      <groupId>org.springframework</groupId>
      <artifactId>spring-test</artifactId>
      <version>5.2.10.RELEASE</version>
    </dependency>

    <dependency>
      <groupId>org.mybatis</groupId>
      <artifactId>mybatis</artifactId>
      <version>3.5.6</version>
    </dependency>

    <dependency>
      <groupId>org.mybatis</groupId>
      <artifactId>mybatis-spring</artifactId>
      <version>1.3.0</version>
    </dependency>

    <dependency>
      <groupId>mysql</groupId>
      <artifactId>mysql-connector-java</artifactId>
      <version>5.1.47</version>
    </dependency>

    <dependency>
      <groupId>com.alibaba</groupId>
      <artifactId>druid</artifactId>
      <version>1.1.16</version>
    </dependency>

    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.12</version>
      <scope>test</scope>
    </dependency>

    <dependency>
      <groupId>javax.servlet</groupId>
      <artifactId>javax.servlet-api</artifactId>
      <version>3.1.0</version>
      <scope>provided</scope>
    </dependency>

    <dependency>
      <groupId>com.fasterxml.jackson.core</groupId>
      <artifactId>jackson-databind</artifactId>
      <version>2.9.0</version>
    </dependency>
  </dependencies>

  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.tomcat.maven</groupId>
        <artifactId>tomcat7-maven-plugin</artifactId>
        <version>2.1</version>
        <configuration>
          <port>80</port>
          <path>/</path>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>

```
* SpringConfig
```java
@Configuration
@ComponentScan({"com.lonelysnow.service"})
@PropertySource("classpath:jdbc.properties")
@Import({JdbcConfig.class,MyBatisConfig.class})
public class SpringConfig {
}
```
* MybatisConfig
```java
public class MyBatisConfig {
    @Bean
    public SqlSessionFactoryBean sqlSessionFactoryBean(DataSource dataSource){
        SqlSessionFactoryBean factoryBean = new SqlSessionFactoryBean();
        factoryBean.setDataSource(dataSource);
        factoryBean.setTypeAliasesPackage("com.lonelysnow.domain");
        return factoryBean;
    }
    @Bean
    public MapperScannerConfigurer mapperScannerConfigurer(){
        MapperScannerConfigurer msc = new MapperScannerConfigurer();
        msc.setBasePackage("com.lonelysnow.dao");
        return msc;
    }
}
```
* JdbcConfig
```java
public class JdbcConfig {
  @Value("${jdbc.driver}")
  private String driver;
  @Value("${jdbc.url}")
  private String url;
  @Value("${jdbc.username}")
  private String username;
  @Value("${jdbc.password}")
  private String password;

  @Bean
  public DataSource dataSource(){
    DruidDataSource dataSource = new DruidDataSource();
    dataSource.setDriverClassName(driver);
    dataSource.setUrl(url);
    dataSource.setUsername(username);
    dataSource.setPassword(password);
    return dataSource;
  }
}
```

* jdbc.properties
```properties
jdbc.driver=com.mysql.jdbc.Driver
jdbc.url=jdbc:mysql:///ssm_db?useSSL=false
jdbc.username=root
jdbc.password=123456
```

* ServletConfig
```java
public class ServletConfig extends AbstractAnnotationConfigDispatcherServletInitializer {
  @Override
  protected Class<?>[] getRootConfigClasses() {
    return new Class[]{SpringConfig.class};
  }

  @Override
  protected Class<?>[] getServletConfigClasses() {
    return new Class[]{SpringMvcConfig.class};
  }

  @Override
  protected String[] getServletMappings() {
    return new String[]{"/"};
  }
}
```

* SpringMvcConfig
```java
@Configuration
@ComponentScan("com.lonelysnow.controller")
@EnableWebMvc
public class SpringMvcConfig { }
```

**功能模块**
**模型**
* book
```java
package com.lonelysnow.domain;

public class Book {
    private Integer id;
    private String type;
    private String name;
    private String description;

    public Integer getId() {
        return id;
    }

    public void setId(Integer id) {
        this.id = id;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    @Override
    public String toString() {
        return "Book{" +
                "id=" + id +
                ", type='" + type + '\'' +
                ", name='" + name + '\'' +
                ", description='" + description + '\'' +
                '}';
    }
}
```
**数据层标准开发**
* BookDao
```java
public interface BookDao {
    @Insert("insert into tbl_book values(null,#{type},#{name},#{description})")
    public void save(Book book);
    @Update("update tbl_book set type = #{type},name = #{name}, description = #{description} where id = #{id} ")
    public void update(Book book);
    @Delete("delete from tbl_book where id = #{id}")
    public void delete(Integer id);
    @Select("select * from tbl_book where id = #{id}")
    public Book getById(Integer id);
    @Select("select * from tbl_book")
    public List<Book> getAll();

}
```
**业务层标准开发**
* BookService
```java
@Transactional
public interface BookService {

    /**
     * 保存
     * @param book
     * @return
     */
    public boolean save(Book book);

    /**
     * 修改
     * @param book
     * @return
     */
    public boolean update(Book book);

    /**
     * 按id删除
     * @param id
     * @return
     */
    public boolean delete(Integer id);

    /**
     * 按id查找
     * @param id
     * @return
     */
    public Book getById(Integer id);

    /**
     * 查找全部
     * @return
     */
    public List<Book> getAll();
}
```

* BookServiceImpl
```java
@Service
public class BookServiceImpl implements BookService {
    @Autowired
    private BookDao bookDao;

    @Override
    public boolean save(Book book) {
        bookDao.save(book);
        return true;
    }

    @Override
    public boolean update(Book book) {
        bookDao.update(book);
        return true;
    }

    @Override
    public boolean delete(Integer id) {
        bookDao.delete(id);
        return true;
    }

    @Override
    public Book getById(Integer id) {
        return bookDao.getById(id);
    }

    @Override
    public List<Book> getAll() {
        return bookDao.getAll();
    }
}
```
* BookController
```java
@RestController
@RequestMapping("/books")
public class BookController {

    @Autowired
    private BookService bookService;

    @PostMapping
    public boolean save(@RequestBody Book book) {
        return bookService.save(book);
    }

    @PutMapping
    public boolean update(@RequestBody Book book) {
        return bookService.update(book);
    }

    @DeleteMapping("/{id}")
    public boolean delete(@PathVariable Integer id) {
        return bookService.delete(id);
    }

    @GetMapping("/{id}")
    public Book getById(@PathVariable Integer id) {
        return bookService.getById(id);
    }

    @GetMapping
    public List<Book> getAll() {
        return bookService.getAll();
    }
}
```

**测试接口**
* BookServiceTest
```java
@RunWith(SpringJUnit4ClassRunner.class)
@ContextConfiguration(classes = SpringConfig.class)
public class BookServiceTest {
    @Autowired
    private BookService bookService;

    @Test
    public void testGetById(){
        Book book = bookService.getById(1);
        System.out.println(book);
    }

    @Test
    public void testGetAll(){
        List<Book> ls = bookService.getAll();
        System.out.println(Arrays.toString(ls.toArray()));
    }
}
```
---
## 表现层数据封装
* 前端接收数据格式——创建结果模型类，封装数据到data中
* 前端接收数据格式——封装特殊消息到message（msg）属性中
### 设置统一数据返回结果类
**没有统一的标准，但需要跟前端沟通好**

**因为是表现层的格式类，所以做好放在Controller类中**
* Result_结果类
  * 根据需求添加getter setter方法和构造方法
```java
public class Result {
    private Object data;
    private Integer code;
    private String msg;
}
```

* Code——注明使用的标准码
```java
public class Code {
    public static final Integer SAVE_OK = 20011;
    public static final Integer DELETE_OK = 20021;
    public static final Integer UPDATE_OK = 20031;
    public static final Integer GET_OK = 20041;

    public static final Integer SAVE_ERR = 20010;
    public static final Integer DELETE_ERR = 20020;
    public static final Integer UPDATE_ERR = 20030;
    public static final Integer GET_ERR = 20040;
}
```





