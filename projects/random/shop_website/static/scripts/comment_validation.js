window.onload = function () {

    titleField = document.getElementById("id_title");
    contentField = document.getElementById("id_content");
    ratingField = document.getElementById("id_rating");
    addcomField = document.getElementById("addcom");

    titlecorr = false;
    contentcorr = false;
    ratingcorr = false;

    titleField.onblur = validateTitle;
    contentField.onblur = validateContent;
    ratingField.oninput = validateRating;

    canComment();

}

function canComment() {
    if (titlecorr && contentcorr && ratingcorr) {
        addcomField.disabled = false;
    } else {
        addcomField.disabled = "disabled";
    }
}


function validateTitle() {
    titleFielderr = document.getElementById("Title");
    if (titleField.value.length > 5 && titleField.value.length < 128) {
        titleField.style.backgroundColor = "lightgreen";
        titleFielderr.innerHTML = "";
        titlecorr = true;
    } else {
        titleField.style.backgroundColor = "red";
        titleFielderr.innerHTML = "Title must be longer than 5 characters and shorter than 128 characters";
        titlecorr = false;
    }
    canComment()
}

function validateContent() {
    contentFielderr = document.getElementById("Content");
    if (contentField.value.length > 10 && contentField.value.length < 2000) {
        contentField.style.backgroundColor = "lightgreen";
        contentFielderr.innerHTML = "";
        contentcorr = true;
    } else {
        contentField.style.backgroundColor = "red";
        contentFielderr.innerHTML = "Content must be longer than 10 characters and shorter that 2000 characters";
        contentcorr = false;
    }
    canComment()
}

function validateRating() {
    ratingFielderr = document.getElementById("Rating")
    if (ratingField.selectedIndex > 0) {
        ratingField.style.backgroundColor = "lightgreen";
        ratingFielderr.innerHTML = "";
        ratingcorr = true;
    } else {
        ratingField.style.backgroundColor = "red";
        ratingFielderr.innerHTML = "You must select a rating";
        ratingcorr = false;
    }
    canComment()
}