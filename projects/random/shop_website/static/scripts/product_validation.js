window.onload = function () {
    quantityField = document.getElementById("id_quantity");
    totpriceField = document.getElementById("id_total_price_0");
    addorderField = document.getElementById("addtoorder");
    priceField = document.getElementById("price");
    titleField = document.getElementById("id_title");
    contentField = document.getElementById("id_content");
    ratingField = document.getElementById("id_rating");
    addrevField = document.getElementById("addrev");

    totpriceField.readOnly = true;
    quantityField.step = 1;

    titlecorr = false;
    contentcorr = false;
    ratingcorr = false;
    quantitycorr = false;

    titleField.onblur = validateTitle;
    contentField.onblur = validateContent;
    ratingField.oninput = validateRating;
    quantityField.onblur = validateQuantity;

    canAddorder()
    canReview();
}

function canReview() {
    if (titlecorr && contentcorr && ratingcorr) {
        addrevField.disabled = false;
    } else {
        addrevField.disabled = "disabled";
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
        titleFielderr.innerHTML = "Title must be longer than 5 characters";
        titlecorr = false;
    }
    canReview()
}

function validateContent() {
    contentFielderr = document.getElementById("Content");
    if (contentField.value.length > 10 && contentField.value.length < 2000) {
        contentField.style.backgroundColor = "lightgreen";
        contentFielderr.innerHTML = "";
        contentcorr = true;
    } else {
        contentField.style.backgroundColor = "red";
        contentFielderr.innerHTML = "Content must be longer than 10 characters";
        contentcorr = false;
    }
    canReview()
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
    canReview()
}


function canAddorder() {
    if (quantitycorr) {
        addorderField.disabled = false;
    } else {
        addorderField.disabled = "disabled";
    }
}

function validateQuantity() {
    quantityFielderr = document.getElementById("Quantity");
    const price = parseFloat(priceField.innerHTML.match(/[+-]?\d+(,\d+)*(\.\d+)?/)[0].replace(',', ''))
    totpriceField.value = Number(quantityField.value * price);
    if (quantityField.value > 0 && Number.isInteger(Number(quantityField.value))) {
        quantityField.style.backgroundColor = "lightgreen";
        quantityFielderr.innerHTML = "";
        quantitycorr = true;
    } else {
        quantityField.style.backgroundColor = "red";
        quantityFielderr.innerHTML = "Quantity must be a whole number larger than 0";
        quantitycorr = false;
    }
    canAddorder()
}

