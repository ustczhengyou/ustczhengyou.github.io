var tagList = [];

function getArticleList(roc) {
    
    var child = roc.childNodes,
        list = document.getElementById("list"),
        ul = document.createElement("ul"),
        index,
        counter = {};

    list.style.display = "none";
    list.appendChild(ul);
    
    // initial counter
    for (index = 1; index <=6; index++) {
        counter['H'+index] = 0;
    }

    for (var i = 0, len = child.length; i < len; i++) {
    
        if (child[i].nodeName.match(/^H/)) {

            if (counter[child[i].nodeName] == 0) {
                counter[child[i].nodeName] = 1;
            } else {
                counter[child[i].nodeName]++;
            }
            var start = parseInt(child[i].nodeName.slice(1));
            for (index = start + 1; index <= 6; index++) {
                counter['H'+index] = 0; 
            }

            var str = '';
            for (index = 1; index <= start; index++) {
                if (counter['H'+index] == 0) continue;
                str += counter['H'+index] + '.';
            }
            
            // to create a <li><a>text</a></li>
            var li = document.createElement("li"),
                a = document.createElement("a"),
                text = document.createTextNode(str + ' ' + child[i].textContent);
            
            a.setAttribute("href", '#' + child[i].getAttribute("id"));
            a.setAttribute("class", child[i].nodeName);
            a.appendChild(text);
            li.appendChild(a);
            ul.appendChild(li);
        }
    }
}

function addEvent() {

    var toggler = document.getElementById("triangle");

    toggler.addEventListener('click', function() {
        var nav = document.getElementById("list");
        if (nav.style.display === "block") {
            nav.style.display = "none";
            toggler.style.borderLeft = "5px solid black";
            toggler.style.borderBottom = "5px solid transparent";
            toggler.style.borderTop = "5px solid transparent";
            toggler.style.borderRight = "0";
        } else {
            nav.style.display = "block";
            toggler.style.borderTop = "5px solid black";
            toggler.style.borderLeft = "5px solid transparent";
            toggler.style.borderRight = "5px solid transparent";
            toggler.style.borderBottom = "0";
        }
    });
}

function addTagEvent() {

    var posts = document.querySelectorAll(".post-list > li"),
        tagLists = document.querySelector(".tag-list");

    for (var i = 0, leni = posts.length; i < leni; i++) {

        var post = posts[i].querySelector(".post-tag"),
            tags = post.querySelectorAll("span");

        for (var j = 0, lenj = tags.length; j < lenj; j++) {

            posts[i].classList.add(tags[j].textContent);
            
            tags[j].addEventListener("click", function(ev) {

                if (tagList.indexOf(ev.target.textContent) == -1) {

                    tagList.push(ev.target.textContent);

                    var posts = document.querySelectorAll(".post-list > li");

                    for (var i = 0, len = posts.length; i < len; i++) {

                        if (posts[i].style.display == "block") {

                            var tagList = ' ' + posts[i].getAttribute("class") + ' ',
                                tag = ' ' + ev.target.textContent + ' ';

                            if (tagList.indexOf(tag) == -1) {
                                posts[i].style.display = "none";
                                tagLists.innerHTML += '<span class="tag">' + ev.target.textContent + "</span>";
                            }
                        }
                    }
                }
                ev.stopPropagation();

            });
        }
    }
}

window.onload = function() {

    var roc = document.getElementById("post-content"),
        postList = document.querySelector(".post-list");
    
    if (roc) {
        getArticleList(roc);
        addEvent();
    }

    if (postList) {
        addTagEvent();
    }
}