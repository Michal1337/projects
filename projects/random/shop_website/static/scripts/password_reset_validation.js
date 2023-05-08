window.onload = function () {
    pass1Field = document.getElementById("id_new_password1");
    pass2Field = document.getElementById("id_new_password2");
    rstpassField = document.getElementById("rstpass");

    pass1corr = false;
    pass2corr = false;


    pass1Field.onblur = validatePass1;
    pass2Field.onblur = validatePass2;

    canReset();


}

function canReset() {
    if (pass1corr && pass2corr) {
        rstpassField.disabled = false;
    } else {
        rstpassField.disabled = "disabled";
    }
}


function validatePass1() {
    pass1Fielderr = document.getElementById("New password");
    var hasUpper = /[A-Z]/.test(pass1Field.value);
    var hasSpecial = /[!@#$%^&*]/.test(pass1Field.value);
    var hasNum = /[0-9]/.test(pass1Field.value);
    var hasLower = /[a-z]/.test(pass1Field.value);

    if (pass1Field.value.length > 7 && hasUpper && hasSpecial && hasNum && hasNum && hasLower) {
        pass1Field.style.backgroundColor = "lightgreen";
        pass1Fielderr.innerHTML = "";
        pass2Field.disabled = false;
        pass1corr = true;
    } else {
        pass1Field.style.backgroundColor = "red";
        pass1Fielderr.innerHTML = "Password must be longer than 7 characters, should contain a capital letter, lowercase letter, a special character and a number";
        pass1corr = false;
    }
    canReset();
    validatePass2();
}

function validatePass2() {
    pass2Fielderr = document.getElementById("New password confirmation")
    if (pass1Field.value === pass2Field.value && pass1corr) {
        pass2Field.style.backgroundColor = "lightgreen";
        pass2Fielderr.innerHTML = "";
        pass2corr = true;
    } else {
        if (!pass1corr) {
            pass2Field.style.backgroundColor = "red";
            pass2Fielderr.innerHTML = "First enter password above"
            pass2Field.disabled = true;
            pass2corr = false;
        } else {
            pass2Field.style.backgroundColor = "red";
            pass2Fielderr.innerHTML = "Passwords don't match"
            pass2corr = false;
        }
    }
    canReset();
}