window.onload = function () {
    nameField = document.getElementById("id_name");
    minpriceField = document.getElementById("id_min_price_0");
    maxpriceField = document.getElementById("id_max_price_0");
    searchField = document.getElementById("search");

    namecorr = false;
    minpricecorr = false;
    maxpricecorr = false;

    if (!sessionStorage.getItem('pageLoaded')) {
        sessionStorage.setItem('pageLoaded', true);
    } else {
        validateName();
        validateMinprice();
        validateMaxprice();
    }


    nameField.onblur = validateName;
    minpriceField.onblur = validateMinprice;
    maxpriceField.onblur = validateMaxprice;


    canSearch()

}

function canSearch() {
    if (namecorr && minpricecorr && maxpricecorr) {
        searchField.disabled = false;
    } else {
        searchField.disabled = "disabled";
    }
}


function validateName() {
    nameFielderr = document.getElementById("Name");
    if (nameField.value.length > 2 && nameField.value.length < 128) {
        nameField.style.backgroundColor = "lightgreen";
        nameFielderr.innerHTML = "";
        namecorr = true;
    } else {
        nameField.style.backgroundColor = "red";
        nameFielderr.innerHTML = "Name must be longer than 2 characters and shorter than 128"
        namecorr = false;
    }
    canSearch()
}

function validateMinprice() {
    console.log(Number(minpriceField.value))
    console.log()
    minpriceFielderr = document.getElementById("Min price");
    if (Number(minpriceField.value) >= 0 && minpriceField.value.length < 8 && minpriceField.value.length > 0 && Number(maxpriceField.value) >= Number(minpriceField.value)) {
        minpriceField.style.backgroundColor = "lightgreen";
        minpriceFielderr.innerHTML = "";
        minpricecorr = true;
    } else {
        minpriceField.style.backgroundColor = "red";
        minpriceFielderr.innerHTML = "Min price must be greater than 0 and smaller than max price"
        minpricecorr = false;
    }
    canSearch()
}

function validateMaxprice() {
    console.log(Number(maxpriceField.value))
    maxpriceFielderr = document.getElementById("Max price");
    if (maxpriceField.value > 0 && maxpriceField.value.length < 8 && Number(maxpriceField.value) >= Number(minpriceField.value)) {
        maxpriceField.style.backgroundColor = "lightgreen";
        maxpriceFielderr.innerHTML = "";
        maxpricecorr = true;
    } else {
        maxpriceField.style.backgroundColor = "red";
        maxpriceFielderr.innerHTML = "Max price must be greater than 0 and larger than min price"
        maxpricecorr = false;
    }
    canSearch()
}