# go基础知识点总结
## 一、程序结构
1. go语言中不存在未初始化的变量。如果初始化被省略，则会使用该类型零值初始化该变量
2. 简短变量声明（:=）使用时，左侧必须有一个未定义的变量。此时，未定义的变量作初始化，已定义的变量作赋值
3. 变量逃逸是指在超出了作用域的变量不能被立刻回收，仍可找到，如下例
```go
var global *int

func f() {
    var x int
    x = 1
    global = &x
}

func g() {
    y := new(int)
    *y = 1
}
// 此时，在f的作用域外，仍可通过global访问到x中的值，我们就称x发生了变量逃逸
// 而y就没有出现变量逃逸现象，作用域外不可达，可直接被回收
```
4. go允许元组赋值，在进行该操作时，会先对右侧进行计算，再统一赋值给左侧。
```go
a ,b = b,a
```
5. 使用type 类型名字 底层类型 可以给类型重命名。重命名的类型相当于另一个类型，不能与不同类型比较大小。
```go
// Package tempconv performs Celsius and Fahrenheit temperature computations.
package tempconv

import "fmt"

type Celsius float64    // 摄氏温度
type Fahrenheit float64 // 华氏温度
// Celsius和Fahrenheit分别对应不同的温度单位。它们虽然有着相同的底层类型float64，但是它们是不同的数据类型

const (
    AbsoluteZeroC Celsius = -273.15 // 绝对零度
    FreezingC     Celsius = 0       // 结冰点温度
    BoilingC      Celsius = 100     // 沸水温度
)

func CToF(c Celsius) Fahrenheit { return Fahrenheit(c*9/5 + 32) }

func FToC(f Fahrenheit) Celsius { return Celsius((f - 32) * 5 / 9) }

func main(){
	var c Celsius
	var f Fahrenheit
	fmt.Println(c == 0)          // "true"
	fmt.Println(f >= 0)          // "true"
	fmt.Println(c == f)          // compile error: type mismatch 类型不同，不能直接比较
	fmt.Println(c == Celsius(f)) // "true"!
}
```
## 二、基础数据类型
6. 整形：
* int 是有符号整型 uint是无符号整型 （int可能是int32也可能是int64,取决于编译器和环境）
* 即使int和int32等价，在进行运算的时候仍需要进行类型转换
* rune等价int32，byte等价uint8
* uintptr 是一个无符号整型，没有指定具体的大小，但足以容纳指针，一般用于底层编程

7. 布尔值不能隐式转换为0/1 需要借助显式if 语句
8. 内置的len函数可以返回一个字符串中的字节数目（不是rune字符数目）,一个汉字占三个字节，索引返回的也是字节的对应的整数值
```go
func main() {
	a := "你好,世界"
	b := "hello,world"
	fmt.Println(len(a))
	for i := 0; i < len(a); i++ {
		fmt.Println(a[i], string(a[i]))
	}
	fmt.Println("-------------------------")
	for i := 0; i < len(b); i++ {
		fmt.Println(b[i], string(b[i]))
	}
}
```
```go
13
228 ä
189 ½
160  
229 å
165 ¥
189 ½
44 ,
228 ä
184 ¸
150 
231 ç
149 
140 
-------------------------
104 h
101 e
108 l
108 l
111 o
44 ,
119 w
111 o
114 r
108 l
100 d
```
9. 字符串可以和byte切片相互转换
```go
s := "abc"
b := []byte(s)
s2 := string(b)
```
10. 无类型常量：编译器为这些没有明确基础类型的数字常量提供比基础类型更高精度的算术运算；你可以认为至少有256bit的运算精度。这里有六种未明确类型的常量类型，分别是无类型的布尔型、无类型的整数、无类型的字符、无类型的浮点数、无类型的复数、无类型的字符串。
```go
const (
    _ = 1 << (10 * iota)
    KiB // 1024
    MiB // 1048576
    GiB // 1073741824
    TiB // 1099511627776             (exceeds 1 << 32)
    PiB // 1125899906842624
    EiB // 1152921504606846976
    ZiB // 1180591620717411303424    (exceeds 1 << 64)
    YiB // 1208925819614629174706176
)
//例子中的ZiB和YiB的值已经超出任何Go语言中整数类型能表达的范围，但是它们依然是合法的常量，而且像下面的常量表达式依然有效

fmt.Println(YiB/ZiB) // "1024"
```


---

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

将所有依赖备份到vendor文件夹下
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