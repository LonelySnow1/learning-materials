# go mod 依赖管理
1. go.mod 文件内容
2. go mod 命令行管理
3. go install/get/clean

## 指定项目第三方依赖
```go
module test

go 1.21

require(
	dependency latest
	// 依赖     版本
)
```
## 排除第三方依赖
可能是因为第三方依赖有bug
```go
module test

go 1.21

exclude(
	dependency latest
)
```

## 替换第三方依赖
路径或版本号
```go
module test

go 1.21

replace(
	source latest => target latest
)
```
## 撤回有问题的版本
并不是真正意义上撤回发行版本，只是给特定版本号做标记，尽量减少特定版本的使用（对开发者来说）
```go
module test

go 1.21

retract(
	v1.0.0
	v1.1.0
)
```

## 命令行
* 下载依赖
```
go mod dowload github.com/gin-gonic/gin@v1.9.0
```
* 依赖对齐

download下载的时候只会下载当前的包，使用以来对齐后会下载这个包其他依赖
```
go mod tidy
```
* 备份依赖

将所有依赖备份到vender文件夹下
```
go mod vendor
```



# 算法

## 数组

### 1. 克隆数组/切片

```go
//数组 —— 默认深拷贝
arr2 := arr1 //深拷贝
arr2 := &arr1 //浅拷贝
//切片 —— 默认浅拷贝，使用下面方法进行深拷贝
target:=make([]int,len(source))
copy(target,source)
```

### 2. 取整

```go
math.Ceil() //向上取整
math.Floor()//向下取整
math.Round()//四舍五入
```

