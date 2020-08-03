---
layout: post 
title:  "Awesome窗体管理器简介"
tags:   [tool, awesome, ubuntu]
date:   2017-06-10
---

## Awesome介绍

[Awesome](https://awesomewm.org/)是Linux上一款平铺式桌面管理器。所谓平铺式大概就是如下图所示，各个窗口之间没有重叠（感谢Awesome官网提供的截图^v^）。

![Awesome screen]({{"/assets/images/2017-06-10-awesome-screen.png" | relative_url}})

Awesome中单个可视桌面称为一个Tag，类似GNOME桌面中Workspace的概念，即一个虚拟桌面。可以在每个Tag上打开不同的应用，同时也可以在不同Tag之间切换，后文对于Tag就简称为桌面，有助于理解。

在一个Tag上的运行的窗口之间默认采用平铺的布局，但是也可以手动切换布局，Awesome默认提供了多种布局。

## Awesome安装

在Ubuntu上安装Awesome有两种方式。

1. 直接通过`sudo apt-get install awesome`安装。在我的Ubuntu 14.04中使用atp-get安装的Awesome版本为3.4.15；
2. 下载源码自行编译。可以从github上自行选择3.x或者4.x版本进行安装。

## Awesome使用

Awesome中默认的前缀键为`Mod4Key`，即键盘中的`Win`键，下文均以`Win`键指代。

#### 窗口快捷键

+ `Win + Shift + c`
    * 关闭当前窗口
+ `Win + Shift + r`
    * 重绘当前窗口
+ `Win + m/n`
    * 最大/最小化当前窗口
+ `Win + h/l`
    * 把主区域的宽度增大/减少5%
+ `Win + j/k`
    * 切换到下一个/前一个窗口
+ `Win + Shift + j/k`
    * 当前窗口和前一个/后一个窗口互换位置

#### 桌面快捷键

+ `Win + 1~9`
    * 可以在多个桌面之间切换，Awesome默认有9个桌面
+ `Win + Left/Right`
    * 切换到下一个/前一个桌面

#### 其他快捷键

+ `Win + Enter`
    * 打开一个终端
+ `Win + Space`
    * 切换到下一个布局
+ `Win + Shift + i`
    * 显示当前窗口的Class和Instance
+ `Win + Ctrl + r`
    * 重启Awesome
+ `Win + Shift + q`
    * 退出Awesome

## Awesome配置

Awesome使用的配置语言是Lua，这点深得我心。对于Awesome的配置内容太多了，感兴趣的同学可以自行去网上或者github上搜索相关内容。

这里我只说一点，利用apt-get安装的默认配置文件存放在`/etc/xdg/awesome.rc.lua`，如果需要手动修改可以复制一份到`~/.config/awesome/`中（目录不存在可自行创建），在默认配置上进行修改。

## Reference

1. [https://awesomewm.org/](https://awesomewm.org/)
2. [http://wiki.ubuntu.org.cn/Awesome](http://wiki.ubuntu.org.cn/Awesome)

