# PERM元模型
* subject (sub 访问实体)
* object (obj 访问的资源)
* action (act 访问方法)
* eft (eft 策略结果 一般为空 默认指定 allow) 还可以定义为deny [只有这两种]
## Policy 策略 p = {sub,obj,act,eft}
策略一般储存到数据库，因为会有很多
```
[policy_definition]
p = sub,obj,act
```
## Request 请求 r = {sub,obj,act}
内容取决于模型，要和策略匹配，一般和p相同
```go
[request_definition]
r = sub, obj, act
```

## Matchers 匹配规则 Request和Policy的匹配规则
 r 请求 p 策略；
 本质上Marchers就是一条表达式，将r和p进行匹配，从而返回结果（eft）
 ```go
[matchers]
m = r.sub == p.sub && r.obj == p.obj && r.act == p.act
 ```
## Effect 影响
它决定我们是否可以放行，这个规则是定死的

| Policy effect                                                | 意义               | 示例              |
|--------------------------------------------------------------|:-----------------|-----------------|
| some(where (p.eft == allow))                                 | 	allow-override	 | ACL, RBAC, etc. |
| !some(where (p.eft == deny))                                 | 	deny-override   | 	Deny-override  |
| some(where (p.eft == allow)) && !some(where (p.eft == deny)) | 	allow-and-deny  | 	Allow-and-deny |
| priority(p.eft) \|\| deny                                    | priority         | 	Priority       |
| subjectPriority(p.eft)                                       | 	基于角色的优先级        | 	主题优先级          |

## role_definition 角色域
```go
g = _,_ 表示以角色为基础
g = _,_,_ 表示以域为基础（多商户模式）
```
## 例子
```go
[request_definition]
r = sub, obj, act

[policy_definition]
p = sub, obj, act,eft

[role_definition]
g = _, _

[policy_effect]
e = some(where (p.eft == allow))

[matchers]
m = g(r.sub, p.sub) && r.obj == p.obj && r.act == p.act
```
```go
策略
p, alice, data1, read
p, bob, data2, write
p, data2_admin, data2, read,deny
p, data2_admin, data2, write

g, alice, data2_admin
```
```go
请求
alice, data1, read
alice, data1, write
alice,data2,write
alice,data2,read
```
```go
执行结果
true Reason: ["alice","data1","read"]
false
true Reason: ["data2_admin","data2","write"]
fals
```