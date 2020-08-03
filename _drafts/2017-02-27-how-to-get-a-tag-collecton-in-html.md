---
layout: post
title:  "获取HTML中标签所有元素的方法"
tags:   [html, javascript]
date:   2017-02-27
---

## 获取HTML中标签所有元素的方法

测试HMTL代码如下，获取此HTML页面中的`p`标签集合。

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DOM Demo</title>
    <!-- your jquery.js -->
</head>
<body>
    <p>paragraph 1</p>
    <p>paragraph 2</p>
    <p>paragraph 3</p>
</body>
</html>
```

可以通过如下三种方式获取此HTML页面中的`p`标签集合，其中前两种是HTML自带的，最后一种借用了jQuery库实现。

+ `var p0 = document.getElementsByTagName("p");`
+ `var p1 = document.querySelectorAll("p");`
+ `var $p2 = $("p");`

这三种方式获取静态HTML页面中的标签集合时没有差别，但获取动态HTML页面时会出现差异。在测试HTML中加入一段脚本进行说明。

```javascript
// p0, p1, $p2如三种方式获取中定义
$("<p>new paragraph</p>").insertAfter($p2.get(2));
console.log(p0.length);    // => 4
console.log(p1.length);    // => 3
console.log($p2.length);   // => 3
```

通过代码可以明显地看到，只有`getElementsByTagName()`方式获取的标签集合会随着HTML页面的动态变化而更新。

## 原因分析

最简单的方法就是通过`console.log()`方法输出三个变量，查看具体是什么对象。输出结果如下所示。

![Chorome浏览器输出结果]({{"/assets/images/2017-02-27-how-to-get-a-tag-collection-in-html-1.png" | relative_url}})

可以看到三种获取方式返回的对象都不相同，下面对这些对象进行分析。

### HTMLCollection (GetElementsByTagName)

[MDN文档](https://developer.mozilla.org/en-US/docs/Web/API/HTMLCollection/)表明HTMLCollection在HTML的DOM树中是活的集合，当底层的文档改变的时HTMLCollection会自动改变。因而随着HTML页面的动态改变，`p`标签所有元素的数目也会进行相应的改变。

### NodeList (querySelectorAll)

[MDN文档](https://developer.mozilla.org/en-US/docs/Web/API/NodeList/)表明在某些情况下，NodeList是活的集合，例如获取`Node.childNodes`集合，但是在其余情况下，NodeList都是一个静态的集合，也就意味着一旦获取了一个NodeList之后，后续的DOM变化都不会影响到此NodeList，querySelectorAll()返回的就是一个静态的NodeList，因而HTML改变之后集合中`p`标签的数目没有改变。

```javascript
var parent = document.getElementById('parent');  // parent's type is Node
var child_nodes = parent.childNodes;  // child_nodes's type is NodeList
console.log(child_nodes.length);  // => let's assume 2
parent.appendChild(document.createElement('div'));
console.log(child_nodes.length);  // => should output 3
```

### jQuery

jQuery返回的是jQuery.fn.init对象(?)，这部分还有待研究，之后在更新吧。但是[jQuery文档](http://learn.jquery.com/using-jquery-core/jquery-object/#jquery-objects-are-not-quot-live-quot)中对$("p")返回集合不会更新做了说明。

## 小结

MDN是个好东西，我们要珍惜它。