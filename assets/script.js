// an event listener that triggers the function getVisibleElements()
window.addEventListener('scroll', checkScroll, false);

function checkScroll() {
    // first wait until the scolling is finished
    if (window.requestAnimationFrame) {
        requestAnimationFrame(function () {
            var elements = getVisibleElements();
            if (elements.length > 0) {
                document.getElementById(elements[elements.length - 1].id + '-button').click();
            }
        });
    }
}

// a function that returns all the elements that are visible in the screen
function getVisibleElements() {
    var headlines_ids = ['tldr', 'introduction', 'factors', 'directions', 'conclusion', 'references', 'fine-tuning', 'layer-epoch', 'dataset', 'task-similarity', 'embeddings-quality'];
    var elements = [];
    for (var i = 0; i < headlines_ids.length; i++) {
        var element = document.querySelector('#' + headlines_ids[i]);
        var position = element.getBoundingClientRect();
        // checking whether fully visible
        if (position.top >= 0 && position.bottom <= window.innerHeight) {
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