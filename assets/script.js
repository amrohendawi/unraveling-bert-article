// an event listener that triggers the function getVisibleElements()
window.addEventListener('scroll', checkScroll, false);
var isScrolling;

function checkScroll() {
    window.clearTimeout(isScrolling);
    // Set a timeout to run after scrolling ends
    isScrolling = setTimeout(function () {
        // Run the callback
        if (window.requestAnimationFrame) {
            requestAnimationFrame(function () {
                var elements = getVisibleElements();
                if (elements.length > 0) {
                    disableScrolling();
                    var target = '';
                    if (elements.includes(document.querySelector('#tldr'))) {
                        target = 'tldr-button';
                    } else if (elements.includes(document.querySelector('#references'))) {
                        target = 'references-button';
                    } else {
                        target = elements[elements.length - 1].id + '-button';
                    }
                    // get the current url without the hash
                    var url = window.location.href.split('#');
                    var url_without_hash = url[url.length - 1];
                    if(url_without_hash + '-button' !== target) {
                        document.getElementById(target).click();
                    }
                    enableScrolling();
                }
            });
        }
    }, 66);
}

// a function that returns all the elements that are visible in the screen
function getVisibleElements() {
    var headlines_ids = ['tldr', 'introduction', 'factors', 'directions', 'conclusion', 'references', 'fine-tuning', 'layer-epoch', 'dataset', 'task-similarity', 'embeddings-quality'];
    var elements = [];
    for (var i = 0; i < headlines_ids.length; i++) {
        var element = document.querySelector('#' + headlines_ids[i]);
        var position = element.getBoundingClientRect();
        // checking whether fully visible
        if(position.top < window.innerHeight && position.bottom >= 0) {
            elements.push(element);
        }
    }
    return elements;
}

document.addEventListener('click', function (e) {
    // remove the scroll event listener when a button is clicked
    window.removeEventListener('scroll', checkScroll, false);
    if (e.target.classList.contains('button')) {
        e.preventDefault();
    }
    //    add the scroll event listener back after 1 second
    setTimeout(function () {
        window.addEventListener('scroll', checkScroll, false);
    }, 1000);
});

function disableScrolling() {
    document.body.style.overflow = 'hidden';
}

function enableScrolling() {
    document.body.style.overflow = 'auto';
}