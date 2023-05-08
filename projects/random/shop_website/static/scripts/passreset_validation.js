window.onblur = function (){

emailField = document.getElementById("id_email");
rstpassField = document.getElementById("reset");

emailcorr = false;

emailField.onblur = validateEmail;

canReset();

}

function canReset() {
    if (emailcorr) {
        rstpassField.disabled = false;
    } else {
        rstpassField.disabled = "disabled";
    }
}

function validateEmail() {
    console.log("XD")
    emailFielderr = document.getElementById("Email");
    if (/^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$/.test(emailField.value)) {
        emailField.style.backgroundColor = "lightgreen";
        emailFielderr.innerHTML = "";
        emailcorr = true;
    } else {
        emailField.style.backgroundColor = "red";
        emailFielderr.innerHTML = "This field should contain a valid e-mail address";
        emailcorr = false;
    }
    canReset()
}