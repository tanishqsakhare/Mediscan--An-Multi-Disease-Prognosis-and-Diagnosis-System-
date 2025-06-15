function uploadImage() {
    const input = document.getElementById('upload-input');
    const resultDiv = document.getElementById('result');

    const file = input.files[0];

    if (!file) {
        resultDiv.innerHTML = "<p>Please select an image</p>";
    } 
}

function previewImage(event) {
    const input = event.target;
    const preview = document.getElementById('preview-image');
    const resultDiv = document.getElementById('result');


    const reader = new FileReader();
    reader.onload = function () {
        preview.src = reader.result;
        preview.style.display = 'block';
        resultDiv.innerHTML = "<p>Preview of an image</p>";
    };

    reader.readAsDataURL(input.files[0]);
}
function goBack() {
    window.history.back();
}
