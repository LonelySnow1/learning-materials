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
//多维数组需要遍历拷贝
target := make([]int,len(source))
copy(target,source)
```

### 2. 取整


```go
math.Ceil(float64) //向上取整
math.Floor(float64) //向下取整
math.Round(float64) //四舍五入
```

### 3.  搜索某个数在数组内的什么位置
```go
sort.SearchInts(arr []int,target int)
//返回第一个target的位置
//若不存在target，则返回target应插入的位置
//arr切片必须是升序排列
```

## 栈
```go
stack := make([]int,0) //初始化 —— 对应类型
stack = append(stack,k) // 添加元素——push
stack = stack[:len(stack)-1] //弹出元素 —— pop
```


## map
```go
scene := make(map[string]int) // 初始化
scene[a] = b // 添加数据
delete(scene,b) // 删除数据
for k, v := range scene { // 遍历数据
    fmt.Println(k, v)
}
```


## 链表
### 1. 定义
```go
type ListNode struct {
	 Val int
     Next *ListNode
  }
```
### 2. 初始化
```go
MyList := &ListNode{}
```

### 2. 删除链表中的特定元素
```go
//删除current的下一个
//让current的下一个等于下一个的下一个
current.Next = current.Next.Next 
```