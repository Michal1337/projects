window.onload = function (){
nameField = document.getElementById("id_name");
typeField = document.getElementById("id_type");
priceField = document.getElementById("id_price_0");
descField = document.getElementById("id_description");
weightField = document.getElementById("id_weight_0");
condField = document.getElementById("id_condition");
avlbField = document.getElementById("id_available_num");
addprod = document.getElementById("addprod");

namecorr = false;
typecorr = false;
pricecorr = false;
desccorr = false;
weightcorr = false;
condcorr = false;
avlbcorr = false;

if (!sessionStorage.getItem('pageLoaded')) {
    sessionStorage.setItem('pageLoaded', true);
} else {
    validateName();
    validateType();
    validatePrice();
    validateDesc();
    validateWeight();
    validateCond();
    validateAvlb();

}

nameField.onblur = validateName;
typeField.oninput = validateType;
priceField.onblur = validatePrice;
descField.onblur = validateDesc;
weightField.onblur = validateWeight;
condField.oninput = validateCond;
avlbField.onblur = validateAvlb;

canSubmit();

}

function canSubmit(){
    if(namecorr && typecorr && pricecorr && desccorr && weightcorr && condcorr && avlbcorr){
        addprod.disabled = false;
    }else{
        addprod.disabled = "disabled";
    }
}

function validateName(){
    nameFielderr = document.getElementById("Name");
    if(nameField.value.length>6 && nameField.value.length < 128){
        nameField.style.backgroundColor = "lightgreen";
        nameFielderr.innerHTML = "";
        namecorr = true;
    }else{
        nameField.style.backgroundColor = "red";
        nameFielderr.innerHTML = "Name must be longer than 6 characters and shorter than 128 characters"
        namecorr = false;
    }
    canSubmit()
}

function validateType(){
    typeFielderr = document.getElementById("Type");
    if(typeField.selectedIndex > 0){
        typeField.style.backgroundColor = "lightgreen";
        typeFielderr.innerHTML = "";
        typecorr = true;
    }else{
        typeField.style.backgroundColor = "red";
        typeFielderr.innerHTML = "You must select a type"
        typecorr = false;
    }
    canSubmit()
}

function validatePrice(){
    currencyField = document.getElementById("id_price_1");
    priceFielderr = document.getElementById("Price");
    if(priceField.value > 0){
        priceField.style.backgroundColor = "lightgreen";
        currencyField.style.backgroundColor = "lightgreen";
        priceFielderr.innerHTML = "";
        pricecorr = true;
    }else{
        priceField.style.backgroundColor = "red";
        currencyField.style.backgroundColor = "red";
        priceFielderr.innerHTML = "Price must be greater than 0";
        pricecorr = false;
    }
    canSubmit()
}

function validateDesc(){
    descFielderr = document.getElementById("Description");
    if(descField.value.length>20 && descField.value.length < 2000){
        descField.style.backgroundColor = "lightgreen";
        descFielderr.innerHTML = "";
        desccorr = true;
    }else{
        descField.style.backgroundColor = "red";
        descFielderr.innerHTML = "Description must be longer than 20 characters and shorter than 2000 characters";
        desccorr = false;
    }
    canSubmit()
}

function validateWeight(){
    unitField = document.getElementById("id_weight_1");
    weightFielderr = document.getElementById("Weight");
    if(weightField.value > 0){
        weightField.style.backgroundColor = "lightgreen";
        unitField.style.backgroundColor = "lightgreen";
        weightFielderr.innerHTML = "";
        weightcorr = true;
    }else{
        weightField.style.backgroundColor = "red";
        unitField.style.backgroundColor = "red";
        weightFielderr.innerHTML = "Weight must be greater than 0";
        weightcorr = false;
    }
    canSubmit()
}

function validateCond(){
    condFielderr = document.getElementById("Condition");
    if(condField.selectedIndex > 0){
        condField.style.backgroundColor = "lightgreen";
        condFielderr.innerHTML = "";
        condcorr = true;
    }else{
        condField.style.backgroundColor = "red";
        condFielderr.innerHTML = "You must select a condition";
        condcorr = false;
    }
    canSubmit()
}

function validateAvlb(){
    avlbFielderr = document.getElementById("Available num");
    if(avlbField.value > 0){
        avlbField.style.backgroundColor = "lightgreen";
        avlbFielderr.innerHTML = "";
        avlbcorr = true;
    }else{
        avlbField.style.backgroundColor = "red";
        avlbFielderr.innerHTML = "Number of avilable pieces must be greater than 0";
        avlbcorr = false;
    }
    canSubmit()
}