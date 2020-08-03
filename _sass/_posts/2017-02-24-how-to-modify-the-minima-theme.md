---
layout: post
title:  "minima主题定制"
tags:   [jekyll, minima]
date:   2017-02-24
---

## 前言

在搭建这个Jekyll博客的时候选择了最为简单的minima主题，考虑到可以通过阅读minima主题的源码进行一些自己的定制。在此对此博客中对minima主题的修改进行一个简单的说明。

## minima主题定制

主要对minima主题进行了如下修改:

+ 删除了minima主题的footer部分；
+ 添加了Tags页面用于显示对应标签所有文章列表；
+ 在文章内容(post.html页面)上添加了文章所属的标签。

现对以上各部分修改进行一个简单的说明。

## 删除minima主题的footer部分

1. 在博客的_layouts目录中添加一个default.html文件，可参考原始minima主题此文件代码；
2. 去除`include footer.html`部分。

## 添加Tags页面用于显示对应标签下的文章

1. 在博客根目录下添加一个tags.md，minima主题会自动读取此文件作为Tags页面；
2. 对Tags页面进行编写，可参考Liquid语法；
3. 通过添加一个自定义的css对Tags页面进行样式的设计，若博客目录下没有_assets目录可以自行创建，然后在其下创建自定义的css样式文件；
4. 创建之后修改_includes中的head.html文件，通过`<link>`标签引入自定义的样式。

## 在文章内容页面上添加文章所属的标签

1. 文章页面是通过_layout文件夹中的post.html模板呈现；
2. 修改此处的代码，可以参考Liquid语法

## 后记

立个Flag，以后写个Jekyll主题。