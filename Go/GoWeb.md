## socket
1. socket是对TCP 或 UDP的封装，并不存在于网络分层中
2. 按照连接时间可分为长连接和短链接
3. 分为两部分：客户端、服务器端
### Go对Socket的支持
1. TCPAddr表示服务器IP和端口
2. TCPCoon表示连接，封装数据读写操作
3. TCPListener 表示监听服务器特定端口
<br> tcpsock.go
```go
// TCPAddr represents the address of a TCP end point.
type TCPAddr struct {
    IP   IP
    Port int
    Zone string // IPv6 scoped addressing zone
}

// TCPConn is an implementation of the [Conn] interface for TCP network
// connections.
type TCPConn struct {
    conn
}

// TCPListener is a TCP network listener. Clients should typically
// use variables of type [Listener] instead of assuming TCP.
type TCPListener struct {
    fd *netFD
    lc ListenConfig
}
```
<br>简易程序
```go
// 服务端 server/server.go
package main

import (
	"fmt"
	"net"
)

func main() {
	//1. 创建服务器地址
	addr, _ := net.ResolveTCPAddr("tcp4", "localhost:8899")
	//2. 创建监听器
	listener, _ := net.ListenTCP("tcp4", addr)
	//3. 通过监听器获取客户端传递的信息
	//阻塞式
	fmt.Println("服务器已经启动")
	conn, _ := listener.Accept()
	//4. 转换数据
	buffer := make([]byte, 1024)
	size, _ := conn.Read(buffer)
	fmt.Println(string(buffer[:size]))
	//5.关闭连接
	conn.Close()
}
```
```go
// 客户端 client/client.go
package main

import (
	"fmt"
	"net"
)

func main() {
	// 1. 创建服务器地址
	addr, _ := net.ResolveTCPAddr("tcp4", "localhost:8899")
	// 2. 创建连接
	conn, _ := net.DialTCP("tcp4", nil, addr)
	// 3. 发送数据
	conn.Write([]byte("hello world"))
	fmt.Println("服务器已经发送数据")
	// 4. 关闭连接
	conn.Close()
}
```