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
## 三、复合数据类型
1. 无法对map中的元素进行取址操作
```go
_ = &ages["bob"] // compile error: cannot take address of map element
```
原因：
* 动态扩容：
Go 的 map 底层是通过哈希表实现的，当 map 需要扩容时，元素的位置可能会发生变化。如果允许对元素取地址，那么在扩容后，这些地址可能会变得无效。
* 值语义：
map 中存储的是值而不是指针。当你通过 key 获取到一个元素时，实际上是获取到该值的一个副本，而不是原始值的引用。
* 内存迁移：
由于 map 的动态扩容特性，元素的内存地址可能会发生变化，这使得对元素取地址变得不安全

**为什么切片扩容仍然能取地址呢？**

* 切片的底层实现：
切片在 Go 中是一个结构体，包含指向底层数组的指针、长度和容量。即使切片扩容，底层数组的地址可能会改变，但切片结构体本身的地址不会变。因此，你可以对切片中的元素取地址，因为这个地址是底层数组中的元素地址，而不是切片结构体的地址。
* map 的底层实现：
map 的底层实现是哈希表，元素存储在多个桶中。当 map 扩容时，元素可能会重新分配到不同的桶中，导致元素的内存地址发生变化。因此，无法对 map 中的元素取地址，因为这些地址在扩容后可能会变得无效。
* 值语义 vs 引用语义：
切片是引用类型，存储的是底层数组的引用，因此可以对其元素取地址。而 map 是值类型，存储的是元素的副本，无法直接对其元素取地址。


2. interface()类型的切片可以包含自身

3. 如果结构体成员名字是以大写字母开头的，那么该成员就是导出的；这是Go语言导出规则决定的。一个结构体可能同时包含导出和未导出的成员。

4. 结构体嵌入不止获得了匿名成员嵌套的成员，还获得了该类型导出的全部方法

## 四、函数
1. 函数签名： 函数的类型被成为函数的签名，如果两个函数的参数列表和返回值一一对应，那么则称他们的函数签名相同。（变量名不影响函数签名）
2. 没有函数体的函数名声往往代表该函数不是由Go语言实现的
3. 如果一个函数所有的返回值都有显式的变量名，那么该函数的return语句可以省略操作数。这称之为bare return。
```go
func abc(i int) (a, b, c int) {
    if i == 1 {
        return  
    }
    a = 0
    b = 1
    c = 2
    return
}
func main() {
    fmt.Println(abc(1)) //0 0 0
    fmt.Println(abc(2)) //0 1 2
}
// 此时每个return都会按照返回值次序返回
// 等价于 return a,b,c
```
4. Go异常处理有别于其他语言的地方：
* 没有传统意义上的异常：
Go语言没有像Java或C++那样的异常处理机制，即没有try、catch和finally这样的关键字。
* Go使用控制流机制（如if和return）处理错误，这使得编码人员能更多的关注错误处理。
5. 函数可以进行赋值操作,但是不同签名的函数不能进行赋值
6. 函数零值是nil并且可以和nil比较，但函数值之间不可以比较
```go
func abc(i int) int {
    return i * i
}
func a() {}
func main() {
    f := abc
    fmt.Println(f(1))
    fmt.Println(f(2))
    fmt.Printf("%T\n", f)
    f = a //Cannot use 'a' (type func()) as the type func(i int) int
}
```
7. 函数闭包：
```go
// squares返回一个匿名函数。
// 该匿名函数每次被调用时都会返回下一个数的平方。
func squares() func() int {
    var x int
    return func() int {
        x++
        return x * x
    }
}
func main() {
    f := squares()
    fmt.Println(f()) // "1"
    fmt.Println(f()) // "4"
    fmt.Println(f()) // "9"
    fmt.Println(f()) // "16"
}
```
原因：
* 作用域链：
在 JavaScript 和 Go 等编程语言中，函数在定义时会捕获其所在的词法作用域（即定义时的作用域），并形成一个作用域链。这个作用域链允许函数在其外部函数执行完毕后，仍然能够访问外部函数中的变量。
* 函数嵌套：
闭包通常出现在函数嵌套的情况下。当一个函数内部定义了另一个函数，并且内部函数引用了外部函数的变量时，就形成了闭包。
* 变量的持久化：
闭包使得外部函数的变量在外部函数执行完毕后仍然存在。这是因为内部函数引用了这些变量，导致它们不会被垃圾回收机制回收。
* 返回函数或作为参数传递：
闭包通常通过返回一个内部函数或将其作为参数传递给其他函数来实现。这使得内部函数可以在外部函数的作用域之外执行，但仍然能够访问外部函数的变量。
8. return并不是原子性的，返回值可以被改变
```go
func squares() (result int) {
	i := 1
	defer func() {
		result++
	}()
	return i
}
func main() {
	fmt.Println(squares())
}

/* 执行顺序
1. 先将 i赋值给 result result = i
2. 再执行defer result++
3. 返回 return result 此时值为2
*/
```
9. panic发生后程序执行了哪写操作：
   1. 程序中断运行
   2. 执该goroutine中的defer函数
   3. 程序崩溃，输出日志信息 

##  七、Gorountines和Channels
### CSP:
是一种现代的并发编程模型
强调通过通信而不是共享内存来协调并发实体。
在Go语言中，这一理念通过goroutines和channels得以实现

**1. 并不需要关闭每一个channel，当没有引用的时候会自动被Go语言的垃圾回收器回收，泄漏的goroutines并不会被自动回收**

**1.5 关闭重复的channel会造成panic异常；关闭nil值的channel也会导致panic异常。关闭channel还会触发广播机制**

**2. 双向channel可以隐式转换单向channel，但是并没有反向转换的语法**

**3. 关闭操作只用于断言不再向channel发送新的数据，所以只能在发送者关闭，在接收端close是一个编译错误，在编译器检测**

**4. channel的缓存队列解耦了接收和发送的goroutine。**

**5. 一个进程中至有一个线程，一个goroutine中至少有一个函数**

**6. Goroutine状态流转:**
![img.png](img.png)
![img_1.png](img_1.png)

**7. Goroutines是一个结构体，真正让goroutine运行起来的是调度器，Go自己实现的一个用户态的调度器（GMP）**

**GMP：**
>M (thread) 和 G (goroutine)，又引进了 P (Processor)。
> 也就也说 M是线程 G是协程，P是处理器


1. 调度器作用是把可运行的G分配到M上
2. M想要运行G，必须先获取P，P中包含了可运行的G队列




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


# 进阶
## 零、Go零散知识点：
1. var，new和make的区别
   - new返回的指针，make返回的是引用类型的值
   ![img_2.png](material/img_2.png)

2. for 和 for-range的区别：
   - for 容易导致死循环 ；for-range在进入循环前就设定好了循环次数
   - 1.22版本后，Go修复了for - range中循环变量共享同一地址的bug
```go
//循环次数
	a := []int{1, 2, 3, 4, 5}
	for i := 0; i < len(a); i++ {
		fmt.Println(a[i])
		a = append(a, a[len(a)-1]+1)
	} // 死循环
	
	for _, v := range a {
		fmt.Println(v)
		a = append(a, a[len(a)-1]+1)
	}// 1 2 3 4 5
```
```go
// 1.22 更新：
func main() {
   values := []string{"a", "b", "c"}
   for _, v := range values {
	  //v := v    // 这行代码可以显式的创建新变量，用于旧版本代码输出正确结果
      go func() {
        fmt.Println(v)
      }()
   }//如果Go版本在1.22以下，输出结果为 c c c 
   time.Sleep(1 * time.Second)
}

//或者这样直观观察
func main() {
   values := []string{"a", "b", "c"}
   for _, v := range values {
      fmt.Println(&v)
   }
}
/*
0xc000026070
0xc000026070
0xc000026070
 */
```


## 一、常见数据结构的实现原理：
### 1.Channel 管道
#### 1.1 初始化
可用var声明nil管道；用make初始化管道；

len()： 缓冲区中元素个数， cap()： 缓冲区大小
```go
//变量声明 
var a chan int
//使用make初始化
b := make(chan int)  //不带缓冲区
c := make(chan string,2) // 带缓冲区
```
```go
ch1 := make(chan int) // 0 0
ch2 := make(chan int, 2)// 1 2
ch2 <- 1
fmt.Println(len(ch1), len(ch2), cap(ch1), cap(ch2))
```
#### 1.2 读写操作
用 " <- "来表示数据流向，缓冲区满时写/缓冲区空时读 都会阻塞，直到被其他携程唤醒
```go
a := make(chan int, 3)
a <- 1 //数据写入管道
<-a    //管道读出数据
```
管道默认双向可读写，但也可在创建函数时限制单向读写
```go
func write(ch chan<- int,a int)  {
	ch <- a
	// <- ch  无效运算: <- ch (从仅发送类型 chan<- int 接收)
}

func read(ch <-chan int)  {
	<- ch
	//ch <- 1  无效运算: ch <- 1 (发送到仅接收类型 <-chan int)
}
```
读写值为nil的管道，会永久阻塞，触发死锁
```go
	var ch chan int
	ch <- 1  // fatal error: all goroutines are asleep - deadlock!
	<-ch  	 // fatal error: all goroutines are asleep - deadlock!
```

读写已关闭管道：有缓冲区成功可读缓冲区内容，无缓冲区读零值并返回false；写已关闭管道会触发panic

关闭后，等待队列中的携程全部唤醒，按照上述规则直接返回
```go
ch1 := make(chan int)
ch2 := make(chan int, 2)
go func() {
	ch1 <- 1
}()
ch2 <- 2
close(ch1)
close(ch2)
v1, b1 := <-ch1  //0 false
v2, b2 := <-ch2  //2 true
println(v1, v2, b1, b2)
ch1 <- 1  //panic: send on closed channel
ch2 <- 1  //panic: send on closed channel
```
#### 1.3 实现原理
简单来说，channel底层是通过环形队列来实现其缓冲区的功能。再加上两个等待队列来存储被堵塞的携程。最后加上互斥锁，保证其并发安全
```go
type hchan struct {
qcount   uint           // 队列中数据的总数
dataqsiz uint           // 环形队列的大小
buf      unsafe.Pointer // 指向底层的环形队列
elemsize uint16         // 元素的大小（以字节为单位）
closed   uint32         // 表示通道是否已关闭
elemtype *_type         // 元素的类型（指向类型信息的指针）
sendx    uint           // 写入元素的位置
recvx    uint           // 读取元素的位置
recvq    waitq          // 等待接收的队列（包含等待接收的 goroutine）
sendq    waitq          // 等待发送的队列（包含等待发送的 goroutine）

// lock 保护 hchan 中的所有字段，以及阻塞在这个通道上的 sudogs 中的几个字段。
// 在持有此锁时，不要更改另一个 G 的状态（特别是不要使 G 变为可运行状态），
// 因为这可能会与栈收缩操作发生死锁。
lock mutex //互斥锁
}
```
环形队列是依靠数组实现的（buf指向该数组），实现方法类似双指针：一个指向写入位置（sendx），一个指向读取位置（recvx）
![img.png](material/img.png)
等待队列遵循先进先出，阻塞中的携程会被相反的操作依次唤醒

如果写入时，等待接收队列非空(recvq),那么直接将数据给到等待的携程，不用经过缓冲区

select可以监控单/多个管道内是否有数据，有就将其读出；没有也不会阻塞，直接返回；

select执行顺序是随机的
```go
func main() {
	ch1 := make(chan int)
	ch2 := make(chan int)
	go write(ch1)
	go write(ch2)
	for {
		select {
		case e := <-ch1:
			fmt.Printf("ch1:%d\n", e)
		case e := <-ch2:
			fmt.Printf("ch2:%d\n", e)
		default:
			fmt.Println("none")
			time.Sleep(1 * time.Second)
		}
	}
}
func write(ch chan<- int) {
	for {
		ch <- 1
		time.Sleep(time.Second)
	}
}
```
for-range 读取管道时，管道关闭之后不会继续读取管道内数据；

for 循环读取管道时，管道关闭后，仍会继续读取管道内的数据，返回一堆 零值,false
```go
func main() {
   ch1 := make(chan int)
   go write(ch1)
   for e := range ch1 { // 关闭后不会再从管道读取数据
   	fmt.Print(e)
   }
   //1111
   
   for { // 关闭后仍在从管道读取数据。返回 零值,false
   fmt.Print(<-ch1)
   }
   //11110000000000000000000000000000000000000.....
}
func write(ch chan<- int) {
   for i := 1; i < 5; i++ {
   ch <- 1
   time.Sleep(time.Second)
   }
   close(ch)
}
```

### 2. slice 切片
#### 2.1 初始化
var初始化一个nil切片，不分配内存，（并不是空切片）;make 可以指定长度和容量;字面量可以根据长度自动设定长度

若未指定容量，则容量默认等于长度
```go
func main() { // len cap 值已标注在后面
   var v1 []int				    // 0 0
   v2 := make([]int, 0)		    // 0 0
   v3 := make([]int, 5)		    // 5 5
   v4 := make([]int, 5, 10)	    // 5 10
   v5 := []int{}				// 0 0
   v6 := []int{1, 2, 3, 4, 5}	// 5 5
   v7 := *new([]int)            // 0 0 nil切片，不是空切片
}
```

还可以从数组，切片截取来进行初始化。
-  切片长度根据截取长度定。
-  若从切片截取，容量保持跟原切片一致；从数组截取，容量为数组长度 - 起始截取位置；
- 切片截取时，只考虑容量，不考虑长度；即不超过原数组/切片容量就可以截取
```go
func main() {
   a := [10]int{1, 2, 3, 4, 5}
   s1 := a[1:6]
   s2 := s1[0:9]
   s3 := s1[0:10] //panic: runtime error: slice bounds out of range [:10] with capacity 9
   fmt.Println(len(a), cap(a))
   fmt.Println(len(s1), cap(s1))
   fmt.Println(len(s2), cap(s2))
   /*
   10 10
   5 9
   9 9
    */
}
```
#### 2.2 源代码
切片依托于底层数组实现
```go
type slice struct {
	array unsafe.Pointer // 指向底层数组
	len   int            // 切片长度
	cap   int            // 容量
}
```
由于指向的是底层数组，所以在使用数组/切片创建切片的时候，切片会与原数组/切片共用一部分内存，在对切片进行修改的时候，有可能会将原数据一起修改

但扩容之后，数据会被复制到新切片中，此时底层数组就不一样了，不会发生上述情况
```go
//未发生扩容
func main() {
	a := [10]int{1, 2, 3, 4, 5, 6, 7}
	s1 := a[1:5]
	s2 := s1[0:6]
	s1 = append(s1, 10)
	s2[2] = 100
	fmt.Println(a) 
	fmt.Println(s1)
	fmt.Println(s2)
	/*
	[1 2 3 100 5 10 7 0 0 0]
	[2 3 100 5 10]
	[2 3 100 5 10 7]
	 */
}
```
```go
//发生扩容
func main() {
   a := [10]int{1, 2, 3, 4, 5, 6, 7}
   s1 := a[1:5]
   s2 := s1[0:9]
   s1 = append(s1, 10)
   s2 = append(s2, 10)
   s2[2] = 100
   fmt.Println(a)
   fmt.Println(s1)
   fmt.Println(s2)
}
/*  可以看到修改 100 没有改变原数据
[1 2 3 4 5 10 7 0 0 0]
[2 3 4 5 10]
[2 3 100 5 10 7 0 0 0 10]
 */
```
#### 2.3 拷贝
 - 使用copy(new,old)可以拷贝切片，并且此时两个切片底层数组地址不同。
 - 拷贝时，去两个切片长度的最小值进行拷贝，拷贝过程中不会发生扩容操作。
```go
func main() {
	a := []int{1, 2, 3, 4, 5}
	b := make([]int, 3)
	c := make([]int, 10)
	copy(b, a)
	copy(c, a)
	b[0] = 100
	fmt.Println(a)
	fmt.Println(b)
	fmt.Println(c)
	/*
	[1 2 3 4 5]
	[100 2 3]           //修改不会影响原切片，并且复制不会发生扩容操作
	[1 2 3 4 5 0 0 0 0 0]
	 */
}
```
#### 2.4 扩容
- slice扩容时，会先创建一个大数组，再将原数组数据复制进去，最后再执行append操作。
- 1.18前，大于等于1024,每次扩容25%; 小于1024，每次扩容一倍 
- 1.18后，大于256，扩容后的容量计算公式如下：newcap = oldcap+(oldcap+threshold*3)/4; 小于256，每次扩容一倍
- 过渡更加平滑，避免了2-1.25的突变
- 实际扩容后的容量不严格等于计算结果，还要考虑到内存对齐等问题
```go
//扩容后地址改变
func main() {
	a := make([]int, 1, 2)
	fmt.Println(&a[0])
	a = append(a, 1)        //未发生扩容
	fmt.Println(&a[0])
	a = append(a, 2)        //发生扩容
	fmt.Println(&a[0])
	/*
	0xc00000a0d0
	0xc00000a0d0
	0xc0000101c0        //扩容后的地址发生了改变
	 */
}
```
```go
// 扩容示例
func main() {
	a := make([]int, 256) //第一档
	b := make([]int, 257) //第二档 
	c := make([]int, 512)
	d := make([]int, 1024)
	a = append(a, 1)
	b = append(b, 2)
	c = append(c, 3)
	d = append(d, 4)
	fmt.Println(len(a), cap(a))
	fmt.Println(len(b), cap(b))
	fmt.Println(len(c), cap(c))
	fmt.Println(len(d), cap(d))
}
/*          b扩容后计算结果应为 513.25，向上取整 514 ， 但实际结果为 608 这其中就经历了内存对齐
257 512
258 608
513 848
1025 1536
 */
```
#### 2.5 切片表达式
- 简单表达式[low:high] 表示截取[low,high);low,high均可省略
- 扩展表达式[low:high:capmax] 只可省略low;capmax用来限制新切片容量，避免对high后底层数组的元素进行修改
- 作用于字符串时，生成的结果仍为字符串;扩展表达式不能用于字符串
```go
//扩展表达式
func main() {
	a := [10]int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	b := a[:5]
	b = append(b, 5555)
	c := a[:5:5]        //限制切片容量为5 添加元素就会触发扩容操作，改变底层数组指向
	c = append(c, 6666)
	fmt.Println(a) // [1 2 3 4 5 5555 7 8 9 10]
}
```
```go
//用于字符串
func main() {
	a := "lonelysnow"
	b := a[6:]
	//c := a[0:6:6] // invalid operation: 3-index slice of string
	fmt.Println(b)
	fmt.Println(reflect.TypeOf(b)) // 查看截取的类型
}
/*
   snow
   string
 */
```

### 3. map
#### 3.1 初始化
- var/new声明nil map;make初始化map同时可以指定容量;字面量;
- 向nil map中插入会报panic
```go
func main() {
	var m1 map[int]int 		//panic: assignment to entry in nil map
	m2 := *new(map[int]int)	// panic: assignment to entry in nil map
	m3 := make(map[int]int,10)
	m4 := map[int]int{}
	m1[1] = 2
	m2[1] = 3
	m3[1] = 4
	m4[1] = 5
}
```

#### 3.2 增删改查
基础增删改查如下
```go
func main() {
	m4 := map[int]int{}
	m4[1] = 10			//增
	m4[1] = 20			//改
	delete(m4, 1)	    //删
	v, exist := m4[1]	//查
	if exist {
		fmt.Println(v)
	}
}
```
- 若修改的时候，键不存在，则会新增
- 可以对不存在的键进行删除，不会报错
- 查询的时候，如果键不存在，返回：(零值,false)
```go
func main() {
	m4 := map[int]int{}
	m4[2] = 20              //修改没有的键就是新增
	delete(m4, 0)		    //没有key：0的键也不会报错
	v, exist := m4[1]
	fmt.Println(v, exist)	//0 false
}
```
- map并不是线程安全的，并发读写会触发panic
```go
func main() {
	m4 := map[int]int{}
	go func() {
		for {
			m4[0] = 10
		}
	}()
	go func() {
		for {
			a, b := m4[0]   //fatal error: concurrent map read and map write
			fmt.Println(a, b) // 通常情况下应该是 0 false 但偶尔能在写的空窗期读到 10 true
		}
	}()
	go func() {
		for {
			m4[1] = 10      //fatal error: concurrent map writes
		}
	}()
	time.Sleep(2 * time.Second)
}
```
#### 3.3 源码
- 源码中的B，只影响buckets数组的长度，也就是bucket的个数，跟bucket内部能装多少个键值对无关
```go
//map的数据结构
type hmap struct {
	count     int               // 元素个数
	B         uint8             // buckets数组大小
	buckets    unsafe.Pointer   // bucket数组，长度为2^b
	oldbuckets unsafe.Pointer   // 旧bucket数组，用于扩容
	...
}
```
在bucket内的k-v超过8个时，会在创建一个新bucket，由overflow指向它 [扩容]
```go
//bucket的数据结构
type bmap struct {
    tophash [bucketCnt]uint8    //存储Hash值得高8位
    data []byte                 //k-v数据，先存完k，再存v
    overflow uint8              //溢出bucket的位置
}
```

**为什么要存Hash值的高8位？啥叫高8位，Hash低位在干什么？**

>打个比方，如果一个键k的hash值为113，我们一般会先对113%16(bucket数) = 1。好的，此时这个k就会被放入到buckets[1]中，也就是1号bucket
>
>但是程序取模太慢了，为了加快运算速度，要是能把取模操作换成位运算就快多了
> 
> 诶，在对2的N次方求余的时候，还真能够转化成位操作
> 
> eg：11%4  转化成二进制就是 1011 ;4 =2^2 ;将1011向左移动两位，得到10 就是2 也就是商 ，11 就是3 也就是余数
> 
> 也就是说，如果一个数除以2的N次方求余，那么我们就是要得到最后这个数最后N位二进制的值
> 
> 也就是 hash&(2^b-1) 

通过上述公式得到的就是低位hash，也就是余数，用来确定桶。

但是这样只要低位相同，高位不同也在一个桶里，如11001111与11111111 低位都是1111，无法区别，此时就用到高位tophash来确定具体在桶中的位置了。

>在Java中，为了增加散列程度，减少hash冲突，让bucket中的数据分布更加均匀，HashMap将高16位与低16位做**异或**运算，来确保每一位hash都参与到了桶运算中来
> 
> Go采取的是在hash函数中引入随机种子，来减少hash冲突，并使用高位来定位元素，也算是利用上了每一位hash
> 
> PS：为什么用异或？因为异或可以保证两个数值的特性,"&"运算使得结果向0靠近，"|"运算使得结果向1靠近

#### 3.4 负载因子
```
负载因子 = k数/bucket数  //也就是计算出平均每个bucket包含多少k-v对
```
负载因子过低过高都不好，当负载因子大于6.5时，会进行rehash，增加桶数量并将这些hash均匀分布到其中

每个hash表对负载因子容忍能力不同，redis只能容忍负载因子为1（因为每个bucket只能存储1个k-v对）

#### 3.5 扩容
触发扩容的两个条件，满足任一即可：
- 负载因子大于6.5
- overflow数量达到2^min(15,B) —— 溢出桶过多

扩容都是成倍数扩容，因为扩容本质上是B+=1;渐进式扩容，每次操作map的时候将2个bucket中的数据转移到新buckets中

扩容分为增量扩容和等量扩容：
- 增量：桶不够用了，加桶
- 等量：溢出桶太多了，有的都空了，重新排一下，减少一下溢出桶

如果处于扩容过程中，新增操作会直接在新buckets中进行， 但仍从oldbuckets开始寻找

### 4. struct 结构体
go的结构体类似于其他语言中的class，主要区别就是go的结构体没有继承这一概念，但可以使用类型嵌入来实现相似功能。
#### 4.1 初始化
使用type关键字来定义一个新的类型，struct将新类型限定为结构体类型。

结构体中的字段可以为任何类型，但是包含一些特殊类型 如：接口，管道，函数，指针的时候要格外注意
```go
//type定义一个新类型
type newInt int

//type定义一个简单的结构体
type base struct{
   value int   
}

//type定义一个复杂的结构体
type student struct {
   Name string
   age  int
   c    interface{}
   d    func() int
   e    func()
   base     //将base类型嵌入到了student类型中
}
```
#### 4.2 内嵌字段
内嵌字段大体上有两种方式:显式指定(m1)和隐式指定(m2)
- 显式指定就相当于把目标结构体当作字段，调用时需要先调用这个字段，在调用目标结构体中的信息
- 隐式指定相当于把目标结构体中的所有字段都在新结构体中创建了一次，并且指向嵌入结构体内部。同时创建同名嵌入结构体对象[指与base同名]
- 显式创建同名结构体字段 ≠ 隐式指定
```go
type base struct {
	Value int
}
//显式指定
type m1 struct {
	b base
}

//隐式指定
type m2 struct {
	base
}

//显式指定同名字段
type m3 struct {
   base base
}
```
对上述结构体进行调用:
- 只有隐式指定直接操作被嵌入结构体内的数据；
- 隐式指定后，直接操作嵌入结构体中的数据和通同名结构体操作作用一样
```go
func main() {
	a1 := m1{}
	a2 := m2{}
	a3 := m3{}
	//显式指定只能通过嵌入结构体进行操作
	// a1.Value = 1 //a1.Value undefined (type m2 has no field or method Value)
	a1.b.Value = 2
	//隐式指定两种操作数据方法操作的是同一个变量
	a2.Value = 2
	a2.base.Value = 3
	fmt.Println(a2.Value) //3
	//显式指定同名变量 ≠ 隐式指定 
	// a3.Value = 3 //a3.Value undefined (type m3 has no field or method Value)
	a3.base.Value = 4
}
```
当内嵌字段中的字段与结构体中得字段同名时：
- 直接调用时是指定当前结构体中显式定义的字段，但嵌入结构体中的字段仍可通过嵌入类型进行调用
- 方法同理
```go
//数据
func main() {
	a1 := m1{}
	a1.Value = "hello world"
	a1.base.Value = 1
	fmt.Println(a1)
	//获取a1中的所有字段类型
	t := reflect.TypeOf(a1)
	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		fieldType := field.Type
		fmt.Printf("Field: %s, Type: %s\n", field.Name, fieldType.Name())
	}
}

type base struct {
	Value int
}

type m1 struct {
	Value string
	base
}
/*
{hello world {1}}
Field: Value, Type: string
Field: base, Type: base
 */
```
```go
//方法
func main() {
   a1 := m1{}
   a1.test()
   a1.base.test()
   fmt.Println(a1)
   //获取a1中的所有字段类型
   t := reflect.TypeOf(a1)
   for i := 0; i < t.NumField(); i++ {
      field := t.Field(i)
      fieldType := field.Type
      fmt.Printf("Field: %s, Type: %s\n", field.Name, fieldType.Name())
   }
}

type base struct {
   Value int
}

func (b *base) test() {
   b.Value = 1
}
func (m *m1) test() {
   m.Value = "hello world"
}

type m1 struct {
   Value string
   base
}
/*
{hello world {1}}
Field: Value, Type: string
Field: base, Type: base
 */
```
#### 4.3 可见性
 - 首字母大写表示该字段/方法/结构体为可导出的，反之为不可导出的
 - 在同一个包内，不区分是否可导出，都可访问；包外只能访问可导出的字段/方法/结构体
 - 不可导出的字段不可以进行序列化（转化为json）
 - 可通过可导出的方法去操作不可导出的字段
```go
// test/test1.go
package test

type User struct {
   Name string
   age  int
}

func (u *User) Test() {
   u.Name = "hello"
   u.age = 18
}

func (u *User) test() {
   u.Name = "world"
   u.age = 81
}

type student struct {
   Name string
   age  int
}
```
```go
//main.go
package main

import (
   "fmt"
   "test/test"
)

func main() {
   a := test.User{}
   //b := test.student{} // 不能在包外访问未导出结构体
   a.Name = "123"
   //a.age = 123 // 不能在包外访问未导出字段
   a.Test()
   //a.test() // 不能在包外访问未导出方法
   fmt.Println(a)
   /*
      {hello 18}
    */
}
```
#### 4.4 方法与函数
##### 4.4.1 区别
- 方法定义是必须有一个接收器(receiver)；函数不需要
- 大部分情况下，方法的调用需要有一个对象；函数不需要
- 由于go中的所有传递都是值传递，也就是将数据复制一份再调用，所以如果想要修改原本对象的值，就要传递指针，进行引用传递
```go
func 函数名 (参数) 返回值类型 {函数体} //函数定义
func (接收器) 方法名  (参数) 返回值类型 {函数体} // 方法定义
```
```go
func main() {
	a := User{}
	a.setName() // 方法调用
	fmt.Println(a)
	setName(&a) // 函数调用
	fmt.Println(a)
}
/*
   {world 0}
   {hello 0}
 */

type User struct {
	Name string
	age  int
}
//函数
func setName(u *User) {
	u.Name = "hello"
}
//方法
func (u *User) setName() {
	u.Name = "world"
}
```
值传递时可以通过返回值的方式修改目标对象
```go
func main() {
	a := User{}
	a = a.setName() //需要显式给原变量赋值
	fmt.Println(a)
	a = setName(a)  //需要显式给原变量赋值
	fmt.Println(a)
}

type User struct {
	Name string
	age  int
}

func setName(u User) User {
	u.Name = "hello"
	return u
}
func (u User) setName() User {
	u.Name = "world"
	return u
}
```
##### 4.4.2 闭包
说到函数和方法，就必须说一下闭包

什么是闭包？
>简单来说，就是函数内部引用函数外部变量，导致变量生命周期发生变化。这样的函数就叫做闭包
> 
> 常见于函数返回值为另一个函数时
```go
package main

import "fmt"

func main() {
   b := test()
   fmt.Println(b())
   fmt.Println(b())
}

func test() func() int {
   a := 1
   return func() int {
      a++
      return a
   }
}
```
上面的函数导致变量a无法正常释放，导致变量逃逸
```shell
go build -gcflags="-m" main.go
# command-line-arguments
./main.go:11:6: can inline test
./main.go:13:9: can inline test.func1
./main.go:6:11: inlining call to test
./main.go:13:9: can inline main.test.func1
./main.go:7:15: inlining call to main.test.func1
./main.go:7:13: inlining call to fmt.Println
./main.go:8:15: inlining call to main.test.func1
./main.go:8:13: inlining call to fmt.Println
./main.go:6:11: func literal does not escape
./main.go:7:13: ... argument does not escape
./main.go:7:15: ~R0 escapes to heap
./main.go:8:13: ... argument does not escape
./main.go:8:15: ~R0 escapes to heap
./main.go:12:2: moved to heap: a
./main.go:13:9: func literal escapes to heap
```
#### 4.5 Tag 字段标签
##### 4.5.1定义
在reflect包中提供了获取字段名称、类型、Tag的方法（上文展示过获取名称和类型）

结构体StructField表示结构体的一个字段(reflect/type.go)
```go
// A StructField describes a single field in a struct.
type StructField struct {
	// Name is the field name.
	Name string

	// PkgPath is the package path that qualifies a lower case (unexported)
	// field name. It is empty for upper case (exported) field names.
	// See https://golang.org/ref/spec#Uniqueness_of_identifiers
	PkgPath string

	Type      Type      // field type
	Tag       StructTag // field tag string
	Offset    uintptr   // offset within struct, in bytes
	Index     []int     // index sequence for Type.FieldByIndex
	Anonymous bool      // is an embedded field
}

type StructTag string 
```
##### 4.5.2 Tag规范
StructTag本质上就是字符串，理论上任何形式都符合规范。但通常情况下约定，Tag的格式应该是key:"value"
- key:非空字符串，不能包含控制字符，空格，引号，冒号
- value：双引号包围的字符串
- 冒号前后不能有空格，多个value用逗号隔开，key之间用空格隔开
- key一般表示用途，value表示控制指令；

##### 4.5.3 Tag意义
- Go语言反射机制可以给结构体成员赋值，用Tag可以决定赋值的动作
- 可以使用定义好的Tag规则，参考规则就可以继续不同的操作
```go
//仅对Tag值为true的字段赋值（Tag决定赋值动作）
type Person struct {
	Name string `assign:"true"`
	Age  int    `assign:"false"`
}

func assignValues(v interface{}) {
	val := reflect.ValueOf(v).Elem() // 获取指针指向的值
	typ := val.Type()

	for i := 0; i < val.NumField(); i++ {
		field := val.Field(i)
		tag := typ.Field(i).Tag.Get("assign") // 获取字段的tag

		if tag == "true" {
			// 根据字段类型赋值
			switch field.Kind() {
			case reflect.String:
				field.SetString("Default Name")
			case reflect.Int:
				field.SetInt(25)
			}
		}
	}
}

func main() {
	p := &Person{}
	assignValues(p)
	fmt.Printf("Person: %+v\n", p)
}
/*
Person: &{Name:Default Name Age:0}
 */
```
下方例子使用了json:"kind,omitempty"，这个tag规定了字段为空不进行序列化；
```go
import (
	"encoding/json"
	"fmt"
)

type Person struct {
	Name  string `json:"kind,omitempty"` //为空时不进行序列化
	Value int
	age   int
}

func main() {
	// 创建一个 Person 实例
	p := Person{Name: "", Value: 100, age: 100}

	// 序列化为 JSON
	jsonData, _ := json.Marshal(p)
	fmt.Println(string(jsonData)) 
	/*
	{"Value":100}
	 */
}
```
