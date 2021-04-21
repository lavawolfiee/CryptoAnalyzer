window.onload = function() {
    var btn = document.getElementById("predict_btn");
    var img = document.getElementById("result_img");
    var window = document.getElementById("window");
    var steps = document.getElementById("steps");
    var symbol = document.getElementById("symbol");
    var interval = document.getElementById("interval");

    btn.addEventListener("click", function() {
        img.src = 'static/loading.gif';
        var xhr = new XMLHttpRequest();
        var url = new URL("http://127.0.0.1:5000/predict");
        url.searchParams.set('window', window.value);
        url.searchParams.set('steps', steps.value);
        url.searchParams.set('symbol', symbol.value);
        url.searchParams.set('interval', interval.value);
        xhr.open("GET", url);
        xhr.send();

        xhr.onload = function() {
            var responseObj = xhr.response;
            img.src = responseObj;
        };
    });
};