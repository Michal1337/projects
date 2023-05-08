window.onload = function () {
    nickField = document.getElementById("username");
    passField = document.getElementById("password");
    loginField = document.getElementById("login");

    nickcorr = false;
    passcorr = false;

    nickField.onblur = validateNick;
    passField.onblur = validatePass;

    canLogin()

}

function canLogin() {
    if (nickcorr && passcorr) {
        loginField.disabled = false;
    } else {
        loginField.disabled = "disabled";
    }
}


function validateNick() {
    nickFielderr = document.getElementById("Nickerr");
    if (nickField.value.length > 0 && /^[A-Za-z0-9]*$/.test(nickField.value)) {
        nickField.style.backgroundColor = "lightgreen";
        nickFielderr.innerHTML = "";
        nickcorr = true;
    } else {
        nickField.style.backgroundColor = "red";
        nickFielderr.innerHTML = "This is not a valid username";
        nickcorr = false;
    }
    canLogin()
}

function validatePass() {
    passFielderr = document.getElementById("Passerr");
    var hasUpper = /[A-Z]/.test(passField.value);
    var hasSpecial = /[!@#$%^&*]/.test(passField.value);
    var hasNum = /[0-9]/.test(passField.value);
    var hasLower = /[a-z]/.test(passField.value);

    if (passField.value.length > 7 && hasUpper && hasSpecial && hasNum && hasNum && hasLower) {
        passField.style.backgroundColor = "lightgreen";
        passFielderr.innerHTML = "";
        passcorr = true;
    } else {
        passField.style.backgroundColor = "red";
        passFielderr.innerHTML = "This is not a valid password"
        passcorr = false;
    }
    canLogin()
}