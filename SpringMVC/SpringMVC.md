# 初始SpringMVC
## SpringMVC概述
* springMVC是一种表现层框架技术
* SpringMVC是一种基于java实现MVC模型的轻量级web框架
* 优点
  * 使用简单，开发便捷（相比于Servlet）
  * 灵活性强

## 入门案例
### 注解
* @Controller
  * 类型：类注解
  * 位置：SpringMVC控制器类上方
  * 作用：设定核心控制器bean


* @RequestMapping
  * 类型:方法注解
  * 位置:SpringMVC控制器方法请求访问路径
  * 作用:设置当前控制器方法请求访问路径
  * 参数:请求访问路径


* @ResponseBody
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
    return null;
  }
}
```

```java
//MVC配置类
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;

//3.创建SpringMvc的配置文件，加载controller对应的bean
@Configuration
@ComponentScan("com.itheima.controller")
public class SpringMvcConfig {

}
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


