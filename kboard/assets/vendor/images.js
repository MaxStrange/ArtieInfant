var frequency = 500;//ms
var interval = 0;

function show_image() {
  var img = document.createElement("img");
  img.src = "/images/blah.png";
  img.width = 1000;
  img.height = 700;
  img.alt = "blah";
  document.body.appendChild(img);
}

function startLoop() {
    if (interval > 0)
        clearInterval(interval);
    interval = setInterval("updateImages()", frequency);
}

function updateImages() {
    // Get all the images in the image div
    var images = document.getElementById("images_div").getElementsByTagName("img");

    // Update all image srcs
    for (i = 0; i < images.length; i++) {
        images[i].src = images[i].src;
    }
}

