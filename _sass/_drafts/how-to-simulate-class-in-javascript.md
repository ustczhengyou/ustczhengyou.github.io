---
layout: post
title:  "如何在JavaScript中模拟类"
tags:   [javascript]
date:   2017-03-10
---

## JavaScript中模拟类

ES6中引入了`class`与`extend`关键字，简化了JavaScript中类的定义与继承。但我对如何在ES5中模拟类的实现与继承比较感兴趣，同时很多资料与库中都基于原型链实现了这套机制，特此在本文中做一个汇总，以便日后翻阅。

## 《JavaScript Web Application》

[此书](https://book.douban.com/subject/6805476/)中定义了一个`Class`用于完成类的继承。`Class`类需要一个参数`parent`，若此参数为空时，表明返回的类是一个基类，否则是从`Parent`继承的子类。

### Definition

默认使用`init()`函数作为类的构造函数。同时可以通过`include`和`extend`两个函数进行类功能的拓展，其中`include`在原型链上进行扩展，`extend`是作为类的静态函数。

```javascript
var Class = function (parent) {
    var klass = function () {
        this.init.apply(this, arguments);
    };

    // If parent exist, inherits from parent
    if (parent) {
        // We use F instead of subClass sometimes
        var subClass = function () {};
        subclass.prototype = parent.prototype;
        klass.prototype = new subClass;
    }

    // In case a class do not define the init() function
    klass.prototype.init = function () {};
    
    // Same as jQuery
    klass.fn = klass.prototype;

    klass.fn.parent = klass;
    klass._super = klass.__proto__; // ??

    // Add funcitons to the instance
    klass.include = function () { /*  */}

    // Add function to the class, as a static function
    klass.extend = function () { /* */}
    
    return klass;
}
```

### Demo

```javascript
// Base Class
var Animal = new Class;

Animal.include({
    breath: function () {
        console.log('breath');
    }
});

// Cat inherits from Animal
var Cat = new Class(Animal);
var tommy = new Cat;
tommy.breath();  // => breah
```

## 《JavaScript Patterns》

[此书](https://book.douban.com/subject/5252901/)中定义了一个`Klass`用于进行类之间的继承，此函数接受两个参数，其中`parent`为父类，当不需要父类的时候可以传入`null`，`props`为类的方法，会添加到类的原型链上。

### Definition

此处的类定义借鉴了`PHP`中的`__construct`方法，同时使用一个`uber`变量指向父类的原型，从而可以调用父类的同名方法。

```javascript
var klass = function (Parent, props) {
    
    var Child, F, i;

    // Build new Constructor
    Child = function () {

        /* Have Parent and Parent have a construct function,
           then uses parent's construct function first */
        if (Child.uber && Child.uber.hasOwnProperty("__construct")) {
            Child.uber.__construct.apply(this, arguments);
        }
        if (Child.prototype.hasOwnProperty("__construct")) {
            Child.prototype.__construct.apply(this, arguments);
        }
    };

    // Inherits
    Parent = Parent || Object;
    F = funtion () {};
    F.prototype = Parent.prototype;
    Child.prototype = new F();
    Child.uber = Parent.prototype;
    Child.prototype.constructor = Child;

    // Add props
    for (i in props) {
        if (props.hasOwnProperty(i)) {
            Child.prototype[i] = props[i];
        }
    }

    return Child;
};
```

### Demo

```javascript
var Man = klass(null, {
    __construct: function (name) {
        console.log("Man's constructor");
        this.name = name;
    },
    getName: function () {
        return this.name;
    }
});

var SuperMan = klass(Man, {
    __construct: function (name) {
        console.log("SuperMan's constructor");
    },
    getName: function () {
        var name = SuperMan.uber.getName.call(this);
        return "I am " + name;
    }
});
```

## Others

to be continued...