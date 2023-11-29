# HTML





## 使用Visual Studio Code 来编写HTML代码



VSCODEPROJECT--新建文件--命名后缀为 .html 

文件内编写

```html
<!DOCTYPE html>
<html>			<!-- 完整的HTML页面 -->
<head>			<!-- 头部元素 -->
    <meta charset="UTF-8"> 			<!--定义网页编码格式为 utf-8 -->
    <title>Hello lhw</title>		<!--元素描述了文档的页面标题 -->
</head> 
<body>			<!-- 元素包含了可见的页面内容 -->
    <h1>这是一个标题</h1>
    <p>这是第一段话<p>
</body>    
</html>			<!--完整的HTML页面-->
```



## HTML基础



### 标题、段落、**图像**、换行、加粗、斜体，电脑自动输出、上标和下标、链接

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    
    <title>哈喽</title>
     <!--<body>  ... </body>定义文档的主题  -->
<body>
     
    <h1>这是一个标题</h1>
    <h2>这是一个标题</h2>
    <h3>这是一个标题</h3>
    <h4>这是一个标题</h4>
    <h5>这是一个标题</h5>
    <h6>这是一个标题</h6>

    <!-- 段落的标签 -->
    <p>这是第一段话<p>
    <p>这是第二段话 假如说我这段话很长，换行时避免不<br>了的，那换行后的效果</p>

    <!-- 链接的标签 -->
    <!-- <a href ="网址">链接文本</a> -->
    <!-- "链接文本" 不必一定是文本。图片或其他 HTML 元素都可以成为链接。 -->
    <a href = "https://www.bilibili.com">哔哩哔哩！干杯！</a><br><br>
    <a href="https://www.runoob.com/">访问菜鸟教程</a><br><br>
    <!-- <a href="网址" target="_blank" rel="noopener noreferrer">链接文本</a> -->
    <!-- target 属性，  在新窗口打开 -->
    <a href="https://www.bilibili.com" target="_blank" rel="noopener noreferrer">哔哩哔哩,亁杯!</a> <br><br>

    <!-- id 属性可用于创建一个 HTML 文档书签，书签不会以任何特殊方式显示 -->
    <!-- <a href="#标签名">链接文本</a> -->
    <a id="tips">提示部分</a><br><br>
    <a href="#tips">点击访问提示部分，定位到此处</a><br><br>
    <a href="https://www.runoob.com/html/html-links.html#tips">
        访问有用的提示部分</a>
        <br><br>
        
    <b>加粗文本</b><br><br>
    <i>斜体文本</i><br><br>
    <code>电脑自动输出</code><br><br>
    这是 <sub> 下标</sub> 和 <sup> 上标</sup> <br><br>
    
    那我直接写字能行吗<br><br>
    还真行<br><br>

    <!-- <hr>标签在 HTML 页面中创建水平线 -->
    <hr>
    <br><br>

    <!-- 图片标签img -->
    <!-- 语法<img scr="图片路径" alt="加载失败时显示的替换文本" title="鼠标悬停时的提示信息">  -->
    
    <!-- 相对路径 工作中常用路径 -->
    <!-- 相对路径1：与HTML文件在同一目录下   路径直接写文件名或文件名前加上 ./  -->
    <img src="./bing.jpg" alt="冰冰" title="我老婆" width="900" height="540" ><br>
    <!-- <br>换行的标签 -->
    <img src="bing.jpg" alt="冰冰" width="900" height="540">
    <!-- 相对路径2：在上一级文件中 取上级目录   ../文件名 返回上一级目录 -->
    <img src="../bing2.jpg" alt="冰冰" width="900" height="540">
    <!-- 相对路径3：在下一级目录中  文件夹名称/文件名-->
    <img src="picture/bing3.png"alt="冰冰" width="720" height="540">
    <img src="./picture/bing3.png"alt="冰冰" width="720" height="540">

    <!-- 绝对路径 -->
    <img src="D:\MicrosoftEdgeDownload\VSCode\vscodeProject\.vscode\bing.jpg" >

    <!-- 服务器路径  右击-复制图片地址-->
    <img src="https://www.baidu.com/img/PCtm_d9c8750bed0b3c7d089fa7d55720d6cf.png">


</body>
</html>
```







## HTML元素



HTML 元素以**开始标签**起始以**结束标签**终止



~~~html
链接元素:
<a href = "https://www.bilibili.com">哔哩哔哩！干杯！</a>
~~~

~~~html
段落元素:
<p>这是第一段话<p>  
~~~

~~~html
标题元素:
<h1>这是一个标题</h1>
~~~

~~~ht
换行元素:
<br>
~~~







## HTML属性



HTML 元素可以设置**属性**

属性可以在元素中添加**附加信息**

属性一般描述于**开始标签**

属性总是以名称/值对的形式出现，**比如：name="value"**。





## HTML head

~~~HTML
<head>标签用于定义文档的头部，它是所有头部元素的容器。
~~~

~~~html
<!DOCTYPE html>
<html>
<!-- head里的标签的效果对全局生效 -->
<head>
    <meta charset="UTF-8">

    <!-- <base>元素 -->
    <!-- href="http:www.baidu.com" 即：该标签作为HTML文档中所有的链接标签的默认链接，若链接为空，则转到默认地址 -->
    <!-- target="_blank"  在新页面打开 -->
    <base> 标签描述了基本的链接地址/链接目标，该标签作为HTML文档中所有的链接标签的默认链接:
    <base href="http:www.baidu.com" target="_blank">
    <!-- <link>元素 -->
	
 <link> 标签定义了文档与外部资源之间的关系。
 <link> 标签通常用于链接到样式表:
    <!-- <style>元素 -->
    <!-- 设置颜色;body:背景色;p:段落字体色 -->
    <style type="text/css">
        body {background-color:rgb(125, 89, 133)}
        h1{color: rgb(192, 104, 104);}
        p {color:red}
    </style>

    <!-- <meta>元素 -->
    <!-- meta标签描述了一些基本的元数据 -->
    <!-- <meta> 标签提供了元数据.元数据也不显示在页面上，但会被浏览器解析 -->
    <!-- META 元素通常用于指定网页的描述，关键词，文件的最后修改时间，作者，和其他元数据。 -->
    <!-- 元数据可以使用于浏览器（如何显示内容或重新加载页面），搜索引擎（关键词），或其他Web服务 -->
    <!-- <meta> 一般放置于 <head> 区域 -->
    <!-- 1.为搜索引擎提供关键词 -->
    <meta name="keywords" content="HTML,CSS">
    <!-- 2.为网页丁页描述 -->
    <meta name="description" content="笔记">
    <!-- 3.定义网页作者 -->
    <meta name="author"content="LiHuawei">
    <!-- 4.每30秒钟刷新当前页面 -->
    <meta http-equiv="refresh" content="30">

    <!-- <script>元素 -->
    
    <!-- <title>定义了HTML文档的标题 -->
    <title>Document</title>
</head>

<body>
    <h1>欢迎</h1><br><br><hr>
    <p>进入次元世界</p>

    <a href="https://www.bilibili.com">哔哩哔哩</a>
    <!-- 链接地址为空，将设置的默认地址打开 -->
    <a href>哔哩哔哩</a>

</body>
</html>
~~~







## HTML CSS

CSS 是在 HTML 4 开始使用的,是为了更好的渲染HTML元素而引入的.

CSS 可以通过以下方式添加到HTML中:

- 内联样式- 在HTML元素中使用"style" **属性**
- 内部样式表 -在HTML文档头部 <head> 区域使用<style> **元素** 来包含CSS
- 外部引用 - 使用外部 CSS **文件**

最好的方式是通过外部引用CSS文件.





当特殊样式需要应用到**个别元素**时，可以使用**内联样式**；

~~~html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>使用CSS</title>
</head>
<body>
    <!-- CSS (Cascading Style Sheets) 用于渲染HTML元素标签的样式。 -->

    <!-- 内联样式 -->
    <!-- 当特殊的样式需要应用到 个别元素 时，就可以使用内联样式 -->
    <!-- 设置颜色，左外边距 -->
    <p style="color:blue;margin-left:20px;">这是一个段落。</p>
    <!-- 背景颜色 -->
    <body style="background-color:pink;">
    <h2 style="background-color: yellow;">标<br>题</h2>
    <p style="background-color: green;">段落</p>
    <!-- 字体颜色大小 -->
    <h1 style="font-family: Verdana;">一个标题</h1>
    <p style="font-family: Arial; color:red;font-size: 60px;">一个段落</p>
    <!-- 对齐方式 -->
    <h3 style="text-align: center;">居中对齐的标题</h3>
    <p>后面的段落</p>

    </body>
</body>
</html>
~~~



当**单个文件**需要特别样式时，就可以使用内部样式表。你可以在<head> 部分通过 <style>标签定义内部样式表

~~~html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Document</title>
</head>
<body>
    <!-- 内部样式表 -->
    <!-- 当单个文件需要特别样式时，就可以使用内部样式表 -->
    <style type="text/css">
        body {
            background-color:pink;
        }
        
        p {
            color:blue;    
        }
    </style>
    <h1>内部样式表的联系</h1>
    <p>我就直接在段落里写一下对上面的说明:两个style里面是给页面设置格式<br>
        的，里面的东西在浏览器页面中不可见</p>

</body>
</html>
~~~





当样式需要被应用到**很多页面**的时候，**外部样式**表将是理想的选择。使用外部样式表，你就可以通过更改一个文件来改变**整个站点**的外观。

~~~html
<!DOCTYPE html>
<html >
<head>
    <meta charset="UTF-8">
    <title>随便写写</title>
    <!-- 使用<link>标签导入外部样式表文件 -->
    <!-- href属型设置外部样式表文件的地址 -->
    <!-- rel属性定义关联文档,这里表式关联的是样式表 -->
    <!-- type属性定义导入文件的类型,同style元素一样,text/css表明为 CSS 文本文件。 -->
    <link rel = "stylesheet" type="text/css" href="mystyle.css">
</head>
<body>
    <!-- 外部样式表 -->
    <!-- 样式应用到很多页面的时候,使用外部样式表 -->
    <!-- 外部样式表必须导入到网页文档中，才能够被浏览器识别和解析 -->
    <!-- 使用在头部<head>内 -->
        <h1>hello</h1><br><hr>
        <p>ohhhhhhhhhhh</p>        
</body>
</html>
~~~

* 存在疑问







## HTML表格

~~~html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>练习表格</title>
</head>
<body>
    <h3>没有表头的表格</h3>
    <!-- 表格由标签<table>来定义 -->
    每个表格均有若干行（由 <tr> 标签定义），每行被分割为若干单元格（由 <td> 标签定义）。
    <!-- border定义边框的宽度,等于零时无边框 -->
    <!-- <tr> 标签定义表格中的行 -->
    <!-- <td>标签用于表示一个表格中的单元格 -->
    <!-- <th>表头标签 -->
    <!-- 表格的标题可以用<caption>标签 -->
    <table border="1">
        <caption>表格标签</caption>
        <tr>
            <td>阿珍</td>
            <td>阿强</td>
            <td>阿伟</td>
            <td>彬彬</td>
        </tr>
        <tr>
            <td>lihuawei</td>
            <td>王冰冰</td>
            <td>胖头鱼</td>
            <td>王咕咕</td>
        </tr>
    </table>

    <h3>水平表头</h3>
    <table border="1">
        <tr>
            <th>姓名</th>
            <th>性别</th>
            <th>身高</th>
            <th>职业</th>
        </tr>
        <tr>
            <td>李华威</td>
            <td>男</td>
            <td>183cm</td>
            <td>学生</td>
        </tr>
        <tr>
            <td>王冰冰</td>
            <td>女</td>
            <td>170cm</td>
            <td>记者</td>
        </tr>
    </table>

    <h3>竖直表头</h3>
    <table >
        <tr>
            <th>姓名</th>
            <td>王冰冰</td>
        </tr>
        <tr>
            <th>性别</th>
            <td>女</td>
        </tr>
        <tr>
            <th>身高</th>
            <td>170cm</td>
        </tr>
    </table>
</body>
</html>
~~~







## HTML列表

~~~html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Document</title>
</head>
<body>
    <!-- 无序列表 标签<ul> -->
    <!-- 每个列表项始于 <li> 标签 -->
        <h3>无序列表</h3>
        <ul>
            <li>无序列表第一条</li>
            <li>无序列表第二条</li>
        </ul>

    <!-- 有序列表始于 <ol> 标签; 每个列表项始于 <li> 标签 -->
    <!-- 标签<dd>可以为列表增加注释 -->
        <h3>有序列表</h3>
        <ol>
            <li>有序列表第一行</li>
            <dd>--写下注释</dd>
            <li>有序列表第二行</li>
        </ol>
        <!-- 还可以更改有序列表的格式 -->
        <!-- <ol type="a"> -->
        <h3>有序列表++</h3>
        <ol type="a">
            <li>有序列表第一行</li>
            <dd>--写下注释</dd>
            <li>有序列表第二行</li>
        </ol>


    <!-- 自定义列表 -->
    <!-- 自定义列表以 <dl> 标签开始 -->
    <!-- 每个自定义列表项以 <dt> 开始 -->
    <!-- 每个自定义列表项的定义以 <dd> 开始 -->
        <h3>自定义列表</h3>
        <dl>
            <dt>自定义列表第一行</dt>
            <dd>--写下注释</dd>
            <dt>自定义列表第二行</dt>
            <dd>--写下第二行注释</dd>
        </dl>
</body>
</html>
~~~







##　HTML区块



### html区块元素

大多数html元素被定义为**块级元素**或**内联元素**

**块级元素**在浏览器显示时，通常会以新行来开始（和结束）。

实例:

~~~html
<h1>, <p>, <ul>, <table>
~~~

**内联元素**在显示时通常不会以新行开始

实例：

~~~html
<b>, <td>, <a>, <img>
~~~



### **&lt;div元素&gt;**

~~~html
HTML <div> 元素是块级元素，它可用于组合其他 HTML 元素的容器。
~~~

### &lt;span&gt;元素

~~~html
HTML<span>元素对元素部分进行格式的设置
~~~



~~~html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Document</title>
</head>
<body>
    <!-- <div>中的元素可以单独设置样式 -->
    <!-- <div>定义了文档的区域，块级 (block-level) -->
    <div style="text-align: center;color: pink;">
        <p>what's wrong </p>
        <p>有什么区别</p>
    </div>
    <p>ohhhhhhhhhhhhhhhhhhhhhhhhhhh</p>
    <p><span style="color: blueviolet;">哎呦哥哥</span>嗨你好</p>
</body>
</html>
~~~







## HTML布局



使用div元素：

~~~html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HTML布局</title>
</head>
<body>
    <!-- 用<div>实现布局 -->
    <div id="container" style="width:1000px">
    <div id="header" style="background-color:pink;">

        <h1 style="margin-bottom:0;">主要的网页标题</h1>
    </div>

    <div id="menu" style="background-color:#FFD700; height:500px; width:100px; float: left;">
        <b>菜单</b><br>
        <h3>windows</h3>
        <h3>macOS</h3>
        <h3>linux</h3>
    </div>
    <div id="content" style="background-color: rgb(230, 211, 255);height: 500px;width: 900px; float:left;">
        <h3 style="text-align: center;">helloWorld</h3>
    </div>
    <div id="footer" style="background: darkgray;clear: both;text-align: center;">
        <p>拜拜</p>
    </div>
</body>
</html>
~~~



使用表格：

~~~html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HTML布局(用表格实现)</title>
</head>
<body>
    <!-- <tr> 标签定义表格中的行 -->
    <!-- <td>标签用于表示一个表格中的单元格 -->
    <table width="1000" height="80" border="0">
        <!-- 第一行 -->
        <tr>
            <!-- colspan="3":横跨单元格数量 -->
            <td colspan="3"style="background-color:lemonchiffon;text-align:center">
                <a href="https://www.bilibili.com" target="_blank" rel="noopener noreferrer">哔哩哔哩,亁杯!</a> <br><br>
            </td>
        </tr>
        <!-- 第二行 -->
        <tr>
            <td style="background: darkgreen;width: 100px;">
                <p>windows</p>
                <p>MacOS</p>
                <p>linux</p>
            </td>
            <td style="background:rgb(192, 170, 212);height: 200px;width: 300px;">
                我是什么
            </td>
            <td style="background:rgb(224, 148, 148);height: 200px;width: 100px;">
                <a href="https://www.mi.com/" target="_blank" rel="noopener noreferrer">小米官方</a> <br><br>
            </td>
        </tr>
        <!-- 第三行 -->
        <tr>
            <td colspan="2" style="background: lemonchiffon;">
                <p>我占两个单元格</p>
            </td>
            <td colspan="1" style="background: rgb(109, 105, 74);">
                <p>真的是屌爆了</p>
            </td>
        </tr>
    </table>
</body>
</html>
~~~





## HTML表单

表单是一个包含表单元素的区域。

表单元素是允许用户在表单中输入内容,比如：文本域(textarea)、下拉列表、单选框(radio-buttons)、复选框(checkboxes)等等。

~~~html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>html表单</title>
</head>
<body>
    <!-- 表单使用表单标签<form> -->
    <!-- 表单输入元素使用input -->

    
    
    <!-- 表单本身并不可见。同时，在大多数浏览器中，文本域的默认宽度是 20 个字符。 -->
    <form action="">
        <!-- 文本域通过<input type="text"> 标签来设定 -->

        First name : <input type ="text" name ="firstname"><br>
        Last name  : <input type ="text" name ="lastname"><br>
        E-mail: <input type = "text" name="e-mail"><br>

        <!-- 密码字段通过标签<input type="password"> 来定义 -->
        Password   : <input type = "password" name="pwd"><br><br>

        <!--  -->

        <!-- <input type="radio"> 标签定义了单选框选项 -->
        <input type="radio" name="sex" value="male">Male
        <input type="radio" name="sex" value="female">Female
        <input type="radio" name="sex" value="secret">Secret <br><br>
        
    </form>

    <form>
        <!-- <input type="checkbox"> 定义了复选框选项 -->
        <input type="checkbox" name="vehicle" value="Bike">I have a bike<br>
        <input type="checkbox" name="vehicle" value="Car">I have a car<br><br>
    </form>

    <!-- <input type="submit"> 定义了提交按钮 -->
        <!-- 当用户单击确认按钮时，表单的内容会被传送到里另一个文件，表单
            的动作属性定义了文件的文件名，由动作属性定义的这个文件通
            常会对接受到的输入数据进行相关处理
         -->
    <form name = "input" action="html_form_action.php" method="get">
        Username:<input type="text" name="user">
        <input type="submit" value="Submit">
    </form>

    <br><br>
    <!-- 下拉列表 -->
    请选择你的玩家:
    <form action="">
        <select name = "name">
            <option value="阿伟">阿伟</option>
            <option value="彬彬">彬彬</option>
            <option value="杰哥">杰哥</option>
        </select>
    </form>
    <br><br>
    <!-- 预选下拉列表 -->
    请选择你的英雄:
    <form action="">
        <select name = "name">
            <option value="守约">百里守约</option>
            <option value="韩信" selected>韩信</option>
            <option value="程咬金">程咬金</option>
            <option value="王昭君">王昭君</option>
            <option value="李白">李白</option>
            <option value="公孙离">公孙离</option>
            <option value="老夫子">老夫子</option>
        </select>
    </form>
    <br><br>
    <!-- 文本域 -->
    <textarea name="评论区" id="10086" cols="40" rows="5">请输入你想说的话:
    </textarea>
    <br><br>
    <!-- 创建按钮 -->
    <form action="">
        <input type="button" value="开始游戏">
    </form>

</body>
</html>
~~~

~~~html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>还是HTML表单</title>
</head>
<body>
    <!-- 带边框的表单 -->
    <!-- <fieldset>标签用于显示边框 -->

    <!-- size 标签设置文本域宽度是" "个字符。-->
    <form action="">
        <fieldset>
            <legend>Peronal information:</legend>
            name:
            <input type="text"size="30"><br>
            e-mail:
            <input type="text" size="30"><br>
            date of birth:
            <input type="text" size="10"><br>
        </fieldset>
    </form>
</body>
</html>
~~~







## HTML框架

通过使用框架，你可以在同一个浏览器窗口中显示不止一个页面



~~~html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>这里是HTML框架</title>
</head>
<body>
    <h3>html框架可以在浏览器中显示不止一个页面</h3>
    <!-- <iframe src="URL"></iframe> -->
    <!-- URL指向不同的网页 -->
    <!-- frameborder="0" 移除边框 -->

    
    <iframe  src="../.vscode/helloH5.html" width="600"height="400" ></iframe><br>
    <iframe  src="App012.html" width="600"height="400" ></iframe><br>
    <iframe  src="https://space.bilibili.com/2026561407?from=search&seid=1297
    5886227506424399&spm_id_from=333.337.0.0" width="800"height="400" frameborder="0"></iframe>
</body>
</html>
~~~







## HTML脚本

~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>

<body>
    <!-- <script>标签用于定义客户端脚本,比如JavaScript -->
    <!-- <script>元素既可包含脚本语句，也可以通过scr属性指向外部脚本文件 -->

    <!-- 下面的脚本会向浏览器输出"Hello World!" -->
    <!-- <noscript>只有在浏览器不支持脚本或禁用脚本时，才会显示 -->

    <!-- Javacript可以直接在HTML输出: -->
    <script>
        document.write("Hello World");
    </script>
    <noscript>抱歉,你的浏览器不支持JavaScript!</noscript>

    
    <script>
        document.write("<p>这是一个段落</p>");
    </script>

    <!-- JavaScrip事件响应 -->
    <p id="demo01">
        JavaScript 可以触发事件，就像按钮点击
    </p>
    <script>
        function myFunction() {
            document.getElementById("demo01").innerHTML = "Hello JavaScript!";
        }
    </script>
    <button type="button" onclick="myFunction()">点我！</button>

</body>
</html>
~~~

~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">

    <title>Document</title>
</head>

<body>
    <!-- JavaScript处理 HTML 样式 -->
    <p id="demo">
        JavaScript 能改变 HTML 元素的样式。
    </p>
    <script>
        function myFunction() {
            x = document.getElementById("demo") // 找到元素
            x.style.color = "#ff9999"; // 改变样式
        }
    </script>
    <button type="button" onclick="myFunction()">点击这里</button>

</body>
    
</html>
~~~







## HTML实体符号

~~~html
 &lt;noscript&gt;  
~~~

 &lt;noscript&gt; 



| 显示结果 | 描述        | 实体名称          | 实体编号 |
| :------- | :---------- | :---------------- | :------- |
|          | 空格        | &nbsp;            | &#160;   |
| <        | 小于号      | &lt;              | &#60;    |
| >        | 大于号      | &gt;              | &#62;    |
| &        | 和号        | &amp;             | &#38;    |
| "        | 引号        | &quot;            | &#34;    |
| '        | 撇号        | &apos; (IE不支持) | &#39;    |
| ￠       | 分          | &cent;            | &#162;   |
| £        | 镑          | &pound;           | &#163;   |
| ¥        | 人民币/日元 | &yen;             | &#165;   |
| €        | 欧元        | &euro;            | &#8364;  |
| §        | 小节        | &sect;            | &#167;   |
| ©        | 版权        | &copy;            | &#169;   |
| ®        | 注册商标    | &reg;             | &#174;   |
| ™        | 商标        | &trade;           | &#8482;  |
| ×        | 乘号        | &times;           | &#215;   |
| ÷        | 除号        | &divide;          | &#247;   |

虽然 html 不区分大小写，但实体字符对大小写敏感。







## HTML URL

html统一资源定位器(Uniform Resource Locators)





## HTML5



HTML5 的改进

- 新元素

- 新属性

- 完全支持 CSS3

- Video 和 Audio

- 2D/3D 制图

- 本地存储

- 本地 SQL 数据

- Web 应用

  

[html5](https://www.runoob.com/html/html5-intro.html)



~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>为html添加新元素</title>


    <script>
        document.createElement("mySet")
    </script>
    <style>
        mySet{
        display:block;
        background-color:rgb(203, 204, 145);
        padding:50px;
        font-size:30px;
    }
    </style>
</head>
<body>
    <h3>html添加新元素</h3>
    <p>自己定义格式</p>
    <mySet>我的第一个新元素</mySet><br>
    <mySet>helloWorld</mySet>
</body>
</html>
~~~







## HTML5 Canvas





#### &lt;canvas&gt;元素

&lt;canvas&gt;只有两个可以选的属性，默认值为width为300px，height为150px；



#### 渲染上下文( Thre Rending Context)

&lt;canvas&gt;会创建一个固定大小的画布，会公开一个或多个**渲染上下文**（画笔），使用**渲染上下文** 来绘制和处理要展示的内容。重点研究2d渲染上下文，

~~~javascript
var canvas = document.getElementById('tutorial');
//获得2d上下文对象
var ctx =canvas.getContext('2d');
~~~

~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>

<body>
    <!-- 创建一个画布(Canvas) -->
    <canvas id="myCanvas" width="200" height="100" style="border: 1px solid #000">
    </canvas>
	
    <!-- 绘制工作必须在JavaScript内部完成 -->
    <!-- 绘制的图像在画布上-->
    <script>
        // 找到<canvas>元素
        var c = document.getElementById("myCanvas");
        // 创建context对象
        var ctx = c.getContext("2d");
        // 绘制一个红色矩形
        ctx.fillStyle = "#FF0000";
        ctx.fillRect(0, 0, 150, 75);

    </script>
</body>

</html>
~~~





#### 检测支持性

~~~javascript
var canvas = document.getElementById('tutorial');

if (canvas.getContext){
  var ctx = canvas.getContext('2d');
  // drawing code here
} else {
  // canvas-unsupported code here
}
~~~





#### 代码模板



~~~html
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>菜鸟教程(runoob.com)</title>
<style type="text/css">
canvas {
    border: 1px solid black;
}
</style>
</head>
<body>
<canvas id="tutorial" width="300" height="300"></canvas>
<script type="text/javascript">
function draw(){
    var canvas = document.getElementById('tutorial');
    if(!canvas.getContext) return;
      var ctx = canvas.getContext("2d");
      //开始代码
    //ctx.fillStyle = "#FF0000";
    //ctx.fillRect(0, 0, 150, 75);
}
draw();
</script>
</body>
</html>
~~~





### 绘制形状



#### 栅格(grid)和坐标空间

![img](https://www.runoob.com/wp-content/uploads/2018/12/Canvas_default_grid.png)

canvas 元素默认被网格所覆盖，一个网格中的一个单元相当于canvas元素中的一像素



#### 绘制矩形

&lt;canvas&gt;只支持一种原生的图形绘制：矩形

所有的其他图形都至少需要生成一种路径(path),不过拥有众多路径生成的方法让复杂图形的绘制成为可能：

canvas提供了三种方法绘制矩形：

~~~javascript
fillRect(x,y,width,height)//绘制一个填充的矩形
~~~

~~~JavaScript
strokeRect(x,y,width,height)//绘制一个矩形边框
~~~

~~~JavaScript
clearRect(x,y,width,height)//清除指定的区域矩形，然后这块区域会变得完全透明
~~~

 x, y：指的是矩形的左上角的坐标。(相对于canvas的坐标原点)





### 绘制路径

图形的基本元素是路径，路径是通过不同颜色和宽度的线段或曲线相连而成的不同形状的点的集合，一个路径，甚至一个子路径，都是闭合的



使用路径绘制图形的一些额外的步骤：

创建路径的起始点——调用绘制方法去绘制出路径——把路径封闭——一旦路径生成，通过描边或填充路径区域渲染图形



方法

~~~javascript
beginPath()//新建一条路径，路径一旦创建成功，图像绘制命令被指向到路径上生成路径

moveTo(x,y)//把画笔移动到指定的坐标(x,y).相当于设置路径的起始点坐标

closePath()//闭合路径之后，图形绘制命令又重新指向到上下文中

stroke()//通过线条来绘制图形轮廓

fill()//通过填充路径的内容区域生实心的图形
~~~





#### 绘制线段



~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>绘制线段</title>
    
</head>
<body>
    <!-- 创建一个画布(Canvas) -->
    <canvas id="tutorial" width="500" height="500" style="border: 1px solid #000">

    </canvas>
    <!-- 绘制线段 -->
    <script>
        function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    ctx.beginPath(); //新建一条path
    ctx.moveTo(50, 50); //把画笔移动到指定的坐标
    ctx.lineTo(200, 50);  //绘制一条从当前位置到指定坐标(200, 50)的直线.
    //闭合路径。会拉一条从当前点到path起始点的直线。如果当前点与起始点重合，则什么都不做
    ctx.closePath();
    ctx.stroke(); //绘制路径。
}
draw();
    </script>
</body>
</html>
~~~





#### 绘制三角形边框



~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>绘制三角形边框</title>
</head>
<body>
    <!-- 创建一个画布(Canvas) -->
    <canvas id="tutorial" width="500" height="500" style="border: 1px solid #000">

    </canvas>
    <script>
        function draw(){
    var canvas = document.getElementById('tutorial');
    if (!canvas.getContext) return;
    var ctx = canvas.getContext("2d");
    ctx.beginPath();
    ctx.moveTo(50, 50);
    ctx.lineTo(200, 50);
    ctx.lineTo(200, 200);
      ctx.closePath(); //虽然我们只绘制了两条线段，但是closePath会closePath，仍然是一个3角形
    ctx.stroke(); //描边。stroke不会自动closePath()
}
draw();
    </script>
</body>
</html>
~~~





### 添加样式和颜色

~~~javascript
fillStyle=color设置图形的填充颜色

strokeStyle=color设置图形轮廓的颜色

~~~







## HTML Video



~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title> 视频 </title>
</head>
<body>
    <!-- <video> 元素提供了 播放、暂停和音量控件来控制视频 -->
    <!-- 提供了 width 和 height 属性控制视频的尺寸 -->
    <video width="320" height="240" controls>
        <source src ="images/movie.mp4" type="video/mp4">
    </video>
</body>
</html>
~~~





## HTML Audio



~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>音频</title>
</head>
<body>
    <audio controls>
        <source src="images/晴天.wav" type="audio/wav">
    </audio>
</body>
</html>
~~~





音频格式的MIME类型

| Format | MIME-type  |
| :----- | :--------- |
| MP3    | audio/mpeg |
| Ogg    | audio/ogg  |
| Wav    | audio/wav  |





## HTML Input



~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>html input</title>
</head>
<body>
    <!-- html5新的Input类型 -->

    <!-- color类型用在input字段主要用于选取颜色 -->
    <form action="demo-form.php">
        选择你喜欢的颜色:<input type="color" name="favcolor"><br>
        <input type="submit"><br><br>
    </form>

    <!-- date类型允许你从一个日期选择器里选择一个日期 -->
    <form action="demo-form.php">
        生日:<input type="date" name="bday"><br>
        <input type="submit"><br><br>
    </form>

    <!-- datetime类型允许你选择一个日期 -->
    <form action="demo-form.php">
        生日(日期和时间):<input type="datetime" name="bdaytime"><br>
        <input type="submit"><br><br>
    </form>

    <!-- datetime-local类型允许你选择一个日期和时间 -->
    <form action="demo-form.php">
        生日(日期和时间):<input type ="datetime-local" name="bdaytime">
        <input type="submit"><br><br>
    </form>

    <!-- email类型用于应该包含e-mail地址的输入域 -->
    <form action="demo-form.php">
        E-mail: <input type="email" name="usremail">
        <input type="submit"><br><br>
    </form>

    <!-- month类型允许你选择一个月份 -->
    <form action="demo-form.php">
        生日 ( 月和年 ): <input type="month" name="bdaymonth">
        <input type="submit"><br><br>
    </form>

    <!-- number类型用于应该包含数值 -->
    <form action="demo-form.php">
        数量(1到5直接):<input type="number" name="quantity" min="1" max="5">
        <input type="submit"><br><br>
    </form>

    <!-- range类型用于应该包含一定范围内数字值的输入域 显示为滑动条 -->
    <!-- max - 规定允许的最大值 -->
    <!-- min - 规定允许的最小值 -->
    <!-- step - 规定合法的数字间隔 -->
    <!-- value - 规定默认值 -->
    <form action="demo-form.php" method="GET">
        Points:<input type="range" name="points" min="1" max="10">
        <input type="submit"><br><br>
    </form>

    <!-- search类型用于搜索域，比如站点搜索或Google搜索 -->
    <form action="demo-form.php">
        Search Google:<input type="search" name="googlesearch">
        <input type="submit"><br><br>
    </form>

    <!-- tel定义输入电话号码字段 -->
    <form action="demo-form.php">
        电话号码: <input type="tel" name="usrtel"><br>
        <input type="submit"><br><br>
    </form>

      <!-- time类型允许你选择一个时间 -->
      <form action="demo-form.php">
        选择时间: <input type="time" name="usr_time">
        <input type="submit"><br><br>
    </form>

    <!-- url类型用于应该包含URL地址的输入域 -->
    <form action="demo-form.php">
        添加你的主页: <input type="url" name="homepage">
        <input type="submit"><br><br>
    </form>

    <!-- week类型允许你选择周和年 -->
    <form action="demo-form.php">
        选择周: <input type="week" name="year_week">
        <input type="submit"><br><br>
    </form>
</body>
</html>
~~~





## HTML 表单元素



~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>表单元素</title>
</head>
<body>
    <!-- <datalist>元素规定输入域的选项列表 -->
    <form action="demo-form.php" method="GET">
        <input list="browsers" name="browser">
        <datalist id="browsers">
            <option value="Internet Explorer">
            <option value="Firefox">
            <option value="Chrome">
            <option value="Opera">
            <option value="Safari">
          </datalist>
          <input type="submit"><br><br>
    </form>

    <!-- <output> 元素用于不同类型的输出，比如计算或脚本输出 -->
    <form oninput="x.value=parseInt(a.value)+parseInt(b.value)">0
        <input type="range" id="a" value="50">100
        +<input type="number" id="b" value="50">
        =<output name="x" for="a b"></output>
    </form>
    
</body>
</html>
~~~







# CSS



## 例子+语法



~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS</title>
    <!-- css内容写在头部 -->
    <!-- css内容写在<style>里面 -->
    <style>
        /* 选择器body */
        body{
            /* 声明 */
            background-color: antiquewhite;
        }
        /* 选择器h1 */
        h1{
            /* 声明 */
            color: blueviolet;
            text-align: center;
        }
        /* 选择器p */
        p{
            /* 声明 */
            font-family: "Times New Roman";
            font-size: 20px;
        }
    </style>
</head>
<body>
    
    <h1>CSS!</h1>
    <P>这是一个段落。</P>
</body>
</html>
~~~





## CSS id和class



id

~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>id</title>

    <!-- id 选择器可以为标有特定 id 的 HTML 元素指定特定的样式 -->
    <!-- HTML元素以id属性来设置id选择器,CSS 中 id 选择器以"#"来定义 -->
    <style>
        #para1{
            text-align: center;
            color: red;
        }
    </style>

</head>

<body>
    <p id="para1">Hello world!(有格式)</p>
    <p>Hello world!(无格式)</p>
</body>
</html>
~~~





css

~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>class选择器</title>

    <!-- class 选择器用于描述一组元素的样式 -->
    <!-- class 选择器有别于id选择器，class可以在多个元素中使用 -->
    <style>
        /* 只是设置段落的格式 */
        p.center{
            text-align: center;
        }
    </style>
</head>
<body>
    <h2 class="center">hello world!</h2>
    <p class="center">hello world!</p>
</body>
</html>
~~~







## CSS 创建





### 外部样式表

~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS创建</title>
    <!-- 外部样式表 -->
    <!-- 当样式需要应用于很多页面时，外部样式表将是理想的选择 -->
    <!-- 在使用外部样式表的情况下，可以通过改变一个文件来改变整个站点的外观 -->

    <link rel="stylesheet" type="text/css" href="mystyle.css">
</head>
<body>
    <h1>hello</h1>
</body>
</html>
~~~



~~~css
hr {color:sienna;}
p {margin-left:20px;}
body {background-image:url("../vscode/bing.jpg");}
~~~





### 内部样式表



~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS创建 内部样式表</title>
    <!-- 当单个文档需要特殊的样式时，就应该使用内部样式表 -->

    <style>
        h1{color: blueviolet;}
        p{margin-left: 20px;}
        body{background-image: url("../bing2.jpg");}
    </style>
</head>
<body>
    <h1>hello</h1>
    <p>老婆王冰冰</p>
    <hr><hr>
    <p>显示什么颜色</p>
</body>
</html>
~~~





### 内联样式



~~~html
<!DOCTYPE html>
<html lang="en">
    
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>内联样式</title>
</head>
<!-- 当样式仅需要在一个元素上应用一次时,要使用内联样式 --> 
<!-- 要使用内联样式，你需要在相关的标签内使用样式（style）属性 -->
<!-- Style 属性可以包含任何 CSS 属性 -->

<body>
    <p style="color:sienna; margin-left:20px ;background-image: url(../vscode/bing.jpg);">这是一个段落</p>
    <p style="background-image:url(../vscode/bing.jpg);">hello
        <br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>hello
    </p>
</body>

</html>
~~~





### 多重样式



~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>多重样式</title>
    <!-- 如果某些属性在不同的样式表中被同样的选择器定义，那么属性值将从更具体的样式表中被继承过来 -->
    <!-- 优先级:内联样式 > 内部样式 > 外部样式 > 浏览器默认样式-->
    <!-- 如果外部样式放在内部样式的后面，则外部样式将覆盖内部样式 -->
    
    <link rel="stylesheet" type="text/css" href="mystyle02.css">
    
    <style>
        h3{
        text-align: right;
        font-size: 20pt;
    }
    </style>

</head>
<body>
    <h3>what's wrong?</h3>
    <!-- 格式为
    color:red;left;
    text-align: right;
    font-size: 20pt;
    -->
</body>
</html>
~~~







## CSS Backgrounds





### 背景颜色

页面的背景颜色使用在body的选择器中:

body {background-color:#b0c4de;}



### 背景图像

background-image 属性描述了元素的背景图像.

默认情况下，背景图像进行平铺重复显示，以覆盖整个元素实体.

~~~html
body {background-image:url('paper.gif');}
~~~





### 背景图像-水平或垂直平铺

默认情况下 background-image 属性会在页面的水平或者垂直方向平铺。

如果图像只在**水平方向平铺** (repeat-x), 页面背景会更好些:

~~~html
body
{
	background-image:url('gradient2.png');
	background-repeat:repeat-x;
}
~~~



### 背景图像- 设置定位与不平铺

让背景图像不影响文本的排版

如果你不想让图像平铺，你可以使用 background-repeat 属性:

~~~html
body
{
	background-image:url('img_tree.png');
	background-repeat:no-repeat;
	background-position:right top;
}
~~~





## CSS Text(文本)



~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>文本对齐方式</title>

    <style>
        /* 标题居中 */
        h1 {
            text-align: center;
        }
        /* 时间内容靠右 */
        p.date {
            text-align: right;
        }
        /* 让整行的文字实现左右对齐，不留空白 */
        p.main {
            text-align: justify;
        }
    </style>
</head>

<body>
    <h1>CSS text-align 实例</h1>
    <p class="date">2015 年 3 月 14 号</p>
    <p class="main">
        “当我年轻的时候，我梦想改变这个世界；当我成熟以后，我发现我不能够改变这个世界，我将目光缩短了些，决定只改变我的国家；当我进入暮年以后，我发现我不能够改变我们的国家，我的最后愿望仅仅是改变一下我的家庭，但是，这也不可能。当我现在躺在床上，行将就木时，我突然意识到：如果一开始我仅仅去改变我自己，然后，我可能改变我的家庭；在家人的帮助和鼓励下，我可能为国家做一些事情；然后，谁知道呢?我甚至可能改变这个世界。”
    </p>
    <p><b>注意：</b> 重置浏览器窗口大小查看 &quot;justify&quot; 是如何工作的。</p>
</body>

</html>
~~~



~~~HTML

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>文本修饰</title>
    <style>
        /* 上划线 */
        h1 {
            text-decoration: overline;
        }

        /* 删除线 */
        h2 {
            text-decoration: line-through;
        }

        /* 下划线 */
        h3 {
            text-decoration: underline;
        }

        /* 文本缩进 */
        p {
            text-indent: 50px;
        }
    </style>
</head>

<body>
    <h1>This is heading 1</h1>
    <h2>This is heading 2</h2>
    <h3>This is heading 3</h3>
    <p>文本缩进</p>
</body>

</html>
~~~





## CSS 字体



~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>字体</title>
    <style>
        p {
            font-family: 'Times New Roman', Times, serif;
            font-size: 50px;
        }
    </style>
</head>

<body>
    <p>hello </p>
    hello
    <!-- 1em和当前字体大小相等。在浏览器中默认的文字大小是16px因此，1em的默认大小是16px。可以通面这个公式将像素转换为em：px/16=em -->
    <p style="font-size: 2.5en;">2.5en大小</p>
    <p style="font-size: 3en;">3en大小</p>
</body>

</html>
~~~





## CSS link(链接)

~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS链接</title>
    <style>
        a:link {color:#000000;}      /* 未访问链接*/
        a:visited {color:#6433a5;}  /* 已访问链接 */
        a:hover {color:#FF00FF;}  /* 鼠标移动到链接上 */
        a:active {color:#0000FF;}  /* 鼠标点击时 */
        /* text-decoration 属性主要用于删除链接中的下划线： */
        a:link {text-decoration:none;}
        a:visited {text-decoration:none;}
        a:hover {text-decoration:underline;}
        a:active {text-decoration:underline;}
        /* 背景颜色 */
        a:link {background-color:#B2FF99;}

    </style>
</head>
<body>
    <p><a href="http://www.bilibili.com" target="_black">哔哩哔哩！</a></p>

    注意: hover必须在:link和 a:visited之后定义才有效. <br>
    注意:active必须在hover之后定义是有效的.
</body>
</html>
~~~







## CSS 列表

~~~CSS
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS列表</title>
    <!-- 无序列表 ul -->
    <!-- 有序列表 ol -->
    <style>
        ul.a {list-style-type: circle;}
        ul.b {list-style-type: square;}
        ol.c {list-style-type: upper-roman;}
        ol.d {list-style-type: lower-alpha;}
        /*  将图像作为列表标记  */
        ul.e {list-style-image:url("true.png");}
    </style>
</head>

<body>
    <p>无序列表实例:</p>
    <ul class="a">
        <li>Coffee</li>
        <li>Tea</li>
        <li>Coca Cola</li>
    </ul>

    <ul class="b">
        <li>Coffee</li>
        <li>Tea</li>
        <li>Coca Cola</li>
    </ul>

    <p>有序列表实例:</p>

    <ol class="c">
        <li>Coffee</li>
        <li>Tea</li>
        <li>Coca Cola</li>
    </ol>

    <ol class="d">
        <li>Coffee</li>
        <li>Tea</li>
        <li>Coca Cola</li>
    </ol>
    <ul class="e">
        <li>bing</li>
        <li>冰冰</li>
    </ul>
</body>

</html>
~~~





## CSS Table(表格)



### 表格边框

~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS 表格</title>
    <!-- 表格边框 -->
    <style>
        table,
        th,
        td {
            border: 1px solid rgb(153, 64, 64);
        }
    </style>

</head>

<body>
    <table>
        <tr>
            <th>Firstname</th>
            <th>Lastname</th>
        </tr>
        <tr>
            <td>Peter</td>
            <td>Griffin</td>
        </tr>
        <tr>
            <td>Lois</td>
            <td>Griffin</td>
        </tr>
    </table>
</body>

</html>
~~~

![image-20210925171828267](C:\Users\LIHUAWEI\AppData\Roaming\Typora\typora-user-images\image-20210925171828267.png)

### 折叠边框

~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS折叠边框</title>
    <!-- border-collapse 属性设置表格的边框是否被折叠成一个单一的边框或隔开 -->
    <style>
        table {
            border-collapse: collapse;
        }

        table,
        th,
        td {
            border: 1px solid rgb(82, 60, 161);
        }
    </style>
</head>

<body>
    <table>
        <tr>
            <th>Firstname</th>
            <th>Lastname</th>
        </tr>
        <tr>
            <td>Peter</td>
            <td>Griffin</td>
        </tr>
        <tr>
            <td>Lois</td>
            <td>Griffin</td>
        </tr>
    </table>
</body>

</html>
~~~





### 表格宽度和高度



~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS表格的宽度和高度</title>
    <style>
        /* 表格的宽 */
        table {
            width: 100%;
            
        }
        /* 表格的高 */
        th {
            height: 50px;
        }
        /* 表格的边框 */
        table,
        th,
        td {
            border: 1px solid rgb(82, 60, 161);
        }
    </style>
</head>

<body>
    <table>
        <tr>
            <th>Firstname</th>
            <th>Lastname</th>
        </tr>
        <tr>
            <td>Peter</td>
            <td>Griffin</td>
        </tr>
        <tr>
            <td>Lois</td>
            <td>Griffin</td>
        </tr>
    </table>
</body>

</html>
~~~





## 表格填充和颜色



~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS表格填充</title>
    <!-- 如需控制边框和表格内容之间的间距，应使用td和th元素的填充属 -->
    <style>
        table,
        td,
        th {
            border: 1px solid black;
        }
        /* 在表格的每个格子里内容的上下左右填充 */
        td {
            padding: 15px;
            /* 文字颜色 */
            color:rosybrown;
        }
        /* 设计表头的属性 */
        th{
            /* 背景颜色 */
            background-color: seagreen;
            /* 文字颜色 */
            color: white;
        }
    </style>
</head>

<body>
    <table>
        <tr>
            <th>Firstname</th>
            <th>Lastname</th>
            <th>Savings</th>
        </tr>
        <tr>
            <td>Peter</td>
            <td>Griffin</td>
            <td>$100</td>
        </tr>
        <tr>
            <td>Lois</td>
            <td>Griffin</td>
            <td>$150</td>
        </tr>
        <tr>
            <td>Joe</td>
            <td>Swanson</td>
            <td>$300</td>
        </tr>
        <tr>
            <td>Cleveland</td>
            <td>Brown</td>
            <td>$250</td>
        </tr>
    </table>
</body>

</html>
~~~







## CSS 盒子模型



所有的HTML元素可以看作盒子，在CSS中，box model 这一术语是用来设计和布局时使用

CSS盒模型本质上是一个盒子，封装周围的HTML元素，它包括：边距，边框，填充，和实际内容

盒模型允许我们在其它元素和周围元素边框之间的空间放置元素。



<img src="https://www.runoob.com/images/box-model.gif" alt="CSS box-model" style="zoom: 80%;" />





- **Margin(外边距)** - 清除边框外的区域，外边距是透明的。
- **Border(边框)** - 围绕在内边距和内容外的边框。
- **Padding(内边距)** - 清除内容周围的区域，内边距是透明的。
- **Content(内容)** - 盒子的内容，显示文本和图像。



~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>盒子模型</title>
    <style>
        /* div 实现布局 */
        div{
            background-color: lightgrey;
            width: 300px;
            margin: 25px;
            border: 25px solid green;
            padding: 25px;
            
        }
    </style>
</head>
<body>
    <h2>盒子模型演示</h2>
    <p>CSS盒模型本质上是一个盒子，封装周围的HTML元素，它包括：边距，边框，填充，和实际内容</p>
    <div>这里是盒子内的实际内容。有 25px 内间距，25px 外间距、25px 绿色边框。</div>

</body>
</html>
~~~



![image-20210925204728169](C:\Users\LIHUAWEI\AppData\Roaming\Typora\typora-user-images\image-20210925204728169.png)



## CSS Border边框





~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>边框</title>
    <style>
        p.one {
            /* 边框样式 */
            border-style: solid;
            /* 边框宽度 */
            border-width: 4px;
            /* 边框颜色 */
            border-color: red;
        }

        p.two {
            border-style: dotted solid;
            border-width: medium;
            border-color: #0000ff;
        }
    </style>
</head>

<body>
    <p class="one">一些文本。</p>
    <p class="two">一些文本。</p>
</body>

</html>
~~~

~~~css
/* 四个边框分别设置边框类型 */
p {
    border-top-style:dotted;

    border-right-style:solid;

    border-bottom-style:dotted;

    border-left-style:solid; 

}
~~~





## CSS margin(外边距)

<img src="https://www.runoob.com/wp-content/uploads/2013/08/VlwVi.png" alt="img" style="zoom: 150%;" />

~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>margin外边距</title>
    <!-- margin 清除周围的（外边框）元素区域。margin 没有背景颜色，是完全透明的 -->
    <style>
        p {
            background-color: tan;
        }

        p.margin {
            margin-top: 100px;
            margin-bottom: 100px;
            margin-right: 50px;
            margin-left: 50px;
        }
    </style>
</head>

<body>
    <p>这是一个没有指定边距大小的段落。</p>
    <p class="margin">这是一个指定边距大小的段落。</p>
</body>

</html>
~~~

~~~css
/* 简写 */
p.margin {
    margin:100px 50px;
}
~~~







## CSS padding(填充)



~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <!-- <meta name="viewport" content="width=device-width, initial-scale=1.0"> -->
    <title>CSS padding(填充) </title>
    <!-- CSS padding（填充）是一个简写属性，定义元素边框与元素内容之间的空间，即上下左右的内边距 -->
    <style>
        p {
            background-color: tan;
        }

        p.padding {
            padding-top: 25px;
            padding-bottom: 25px;
            padding-right: 50px;
            padding-left: 50px;
        }
    </style>
</head>

<body>
    <p>这是一个没有指定填充边距的段落。</p>
    <p class="padding">这是一个指定填充边距的段落。</p>
</body>

</html>
~~~





## CSS分组和嵌套



它可能适用于选择器内部的选择器的样式。在下面的例子设置了四个样式：

- **p{ }**: 为所有 **p** 元素指定一个样式。
- **.marked{ }**: 为所有 **class="marked"** 的元素指定一个样式。
- **.marked p{ }**: 为所有 **class="marked"** 元素内的 **p** 元素指定一个样式。
- **p.marked{ }**: 为所有 **class="marked"** 的 **p** 元素指定一个样式。



~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS分组嵌套</title>
    <!--  -->
    <style>
        p {
            color: blue;
            text-align: center;
        }

        .marked {
            background-color: red;
        }

        .marked p {
            color: white;
        }

        p.marked {
            text-decoration: underline;
        }
    </style>
</head>

<body>
    <p>这个段落是蓝色文本，居中对齐。</p>
    <div class="marked">
        <p>这个段落不是蓝色文本。</p>
    </div>
    <p>所有 class="marked"元素内的 p 元素指定一个样式，但有不同的文本颜色。</p>

    <p class="marked">带下划线的 p 段落。</p>
</body>

</html>
~~~





## CSS尺寸



~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS尺寸</title>

    <style>
        img.big{
            height: 400px;
            width: 640px;
        }
    </style>
</head>
<body>
    <img class="big" src="../vscode/bing.jpg" >
</body>
</html>
~~~





## CSS Display



~~~html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS Display</title>
    
    <!-- visibility:hidden可以隐藏某个元素，但隐藏的元素仍需占用与未隐藏之前一样的空间 -->

    <!-- display:none可以隐藏某个元素，且隐藏的元素不会占用任何空间。也就是说，该元素不但被隐藏了，而且该元素原本占用的空间也会从页面布局中消失 -->

    <style>
        h2.hidden{
            visibility: hidden;
        }
        h3.none{
            display: none;
        }
    </style>

</head>
<body>
    <h2>这是一个可见的标题</h1>
    <h2 class="hidden">这是被隐藏的标题</h2>
    <h2>试一试</h2>

    <h3>这仍然时一个可见的标题</h3>
    <h3 class="none">被隐藏而且不占空间</h3>
    <h3>试一试</h3>
</body>
</html>
~~~





## CSS定位



fixed定位

~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>fixed定位</title>
    <!-- 元素的位置相对于浏览器窗口是固定位置 即使窗口是滚动的它也不会移动 -->
    <style>
        p.pos_fixed {
            position: fixed;
            top: 30px;
            right: 5px;
        }
    </style>
</head>

<body>

    <p class="pos_fixed">Some more text</p>


    <p>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text
        <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text
        <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text
        <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text
        <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text <br>text
        <br>text <br>text <br>text <br></p>
</body>

</html>
~~~





relative定位

~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>relative定位</title>
    <!-- 相对定位元素的定位是相对其正常位置 -->
    <style>
        h2.pos_left {
            position: relative;
            left: -20px;
        }

        ​ h2.pos_right {
            position: relative;
            left: 20px;
        }
    </style>
</head>

<body>
    <h2>这是位于正常位置的标题</h2>
    <h2 class="pos_left">这个标题相对于其正常位置向左移动</h2>
    <h2 class="pos_right">这个标题相对于其正常位置向右移动</h2>
    <p>相对定位会按照元素的原始位置对该元素进行移动。</p>
    <p>样式 "left:-20px" 从元素的原始左侧位置减去 20 像素。</p>
    <p>样式 "left:20px" 向元素的原始左侧位置增加 20 像素。</p>
</body>

</html>
~~~







absolute定位

~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>absolute定位</title>
    <style>
        h2 {
            position: absolute;
            left: 100px;
            top: 150px;
        }
    </style>
</head>

<body>
    <h2>这是一个绝对定位了的标题</h2>
    <p>用绝对定位,一个元素可以放在页面上的任何位置。标题下面放置距离左边的页面100 px和距离页面的顶部150 px的元素</p>
</body>

</html>
~~~





sticky定位

~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>sticky定位</title>
    <style>
        div.sticky {
            position: -webkit-sticky;
            position: sticky;
            top: 0;
            padding: 5px;
            background-color: #cae8ca;
            border: 2px solid #4CAF50;
        }
    </style>
</head>

<body>
    <p>尝试滚动页面。</p>
    <p>注意: IE/Edge 15 及更早 IE 版本不支持 sticky 属性。</p>

    <div class="sticky">我是粘性定位!</div>

    <div style="padding-bottom:2000px">
        <p>滚动我</p>
        <p>来回滚动我</p>
        <p>滚动我</p>
        <p>来回滚动我</p>
        <p>滚动我</p>
        <p>来回滚动我</p>
    </div>

</body>

</html>
~~~







重叠的元素

~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>重叠的元素</title>
    <style>
        img {
            position: absolute;
            left: 0px;
            top: 0px;
            z-index: -1;
        }
    </style>
</head>

<body>
    <h1>This is a heading</h1>
    <img src="../vscode/bing.jpg" width="200" height="140" />
    <p>因为图像元素设置了 z-index 属性值为 -1, 所以它会显示在文字之后。</p>
</body>

</html>
~~~







## CSS Overflow



~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CSS overflow</title>
    <!-- CSS overflow 属性可以控制内容溢出元素框时在对应的元素区间内添加滚动条 -->
    <style>
        div {
            background-color: #fffeee;
            width: 200px;
            height: 50px;
            border: 1px dotted black;
            overflow: visible;
        }
    </style>
</head>

<body>
    <div id="overflowTest">
        <p>这里的文本内容会溢出元素框。</p>
        <p>这里的文本内容会溢出元素框。</p>
        <p>这里的文本内容会溢出元素框。</p>
    </div>
</body>

</html>
~~~



| 值      | 描述                                                     |
| :------ | :------------------------------------------------------- |
| visible | 默认值。内容不会被修剪，会呈现在元素框之外。             |
| hidden  | 内容会被修剪，并且其余内容是不可见的。                   |
| scroll  | 内容会被修剪，但是浏览器会显示滚动条以便查看其余的内容。 |
| auto    | 如果内容被修剪，则浏览器会显示滚动条以便查看其余的内容。 |
| inherit | 规定应该从父元素继承 overflow 属性的值。                 |





彼此相邻的浮动元素

~~~html
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>彼此相邻的浮动元素</title>
    <style>
        .thumbnail {
            float: left;
            width: 110px;
            height: 90px;
            margin: 5px;
        }
        /* 元素浮动之后，周围的元素会重新排列，为了避免这种情况，使用 clear 属性。clear 属性指定元素两侧不能出现浮动元素。 */
        .text_line{
            clear:both;
	        margin-bottom:2px;
        }
    </style>
</head>

<body>
    <h3 class="text_line">hdjkfaljkwerfgbapw;rfghbva;wsrfgbva;srbga;rtg</h3>
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
    <h3 class="text_line">fhpawrfuiodgrwghpfqghbwrugv;bawerujtgb;awrugb</h3>
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
    <img class="thumbnail" src="../bing2.jpg" width="107" height="90">
</body>

</html>
~~~





## CSS对齐





~~~css
/*
元素居中对齐:
margin:auto;

文本居中对齐:(文本在元素内居中对齐)
text-align: center;

左右对齐 - 使用定位方式:
position: absolute;
right: 0px;

*/
~~~





