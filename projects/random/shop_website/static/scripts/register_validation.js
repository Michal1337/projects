window.onload = function () {
    nickField = document.getElementById("id_username");
    fnameField = document.getElementById("id_first_name");
    lnameField = document.getElementById("id_last_name");
    emailField = document.getElementById("id_email");
    pass1Field = document.getElementById("id_password1");
    pass2Field = document.getElementById("id_password2");
    registerField = document.getElementById("register");

    nickcorr = false;
    fnamecorr = false;
    lnamecorr = false;
    emailcorr = false;
    pass1corr = false;
    pass2corr = false;


    if (!sessionStorage.getItem('pageLoaded')) {
        sessionStorage.setItem('pageLoaded', true);
    } else {
        validateNick();
        validateFname();
        validateLname();
        validateEmail();
        validatePass1();
        validatePass2();
    }

    nickField.onblur = validateNick;
    fnameField.onblur = validateFname;
    lnameField.onblur = validateLname;
    emailField.onblur = validateEmail;
    pass1Field.onblur = validatePass1;
    pass2Field.onblur = validatePass2;

    canRegister();

}

function canRegister() {
    if (nickcorr && fnamecorr && lnamecorr && emailcorr && pass1corr && pass2corr) {
        registerField.disabled = false;
    } else {
        registerField.disabled = "disabled";
    }
}

function validateNick() {
    nickFielderr = document.getElementById("Username");
    if (nickField.value.length > 4 && /^[A-Za-z0-9]*$/.test(nickField.value)) {
        nickField.style.backgroundColor = "lightgreen";
        nickFielderr.innerHTML = "";
        nickcorr = true;
    } else {
        nickField.style.backgroundColor = "red";
        nickFielderr.innerHTML = "Username must be longer than 4 characters and should contain only letters and numbers";
        nickcorr = false;
    }
    canRegister()
}

function validateFname() {
    fnameFielderr = document.getElementById("First name");
    if (fnameField.value.length > 2 && /[A-Z]/.test(fnameField.value.charAt(0)) && /^[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]*$/.test(fnameField.value)) {
        fnameField.style.backgroundColor = "lightgreen";
        fnameFielderr.innerHTML = "";
        fnamecorr = true;
    } else {
        fnameField.style.backgroundColor = "red";
        fnameFielderr.innerHTML = "First name must be longer than 2 characters, should start with a capital letter and contain only letters";
        fnamecorr = false;
    }
    canRegister()
}

function validateLname() {
    lnameFielderr = document.getElementById("Last name");
    if (lnameField.value.length > 2 && /[A-Z]/.test(lnameField.value.charAt(0)) && /^[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż]*$/.test(lnameField.value)) {
        lnameField.style.backgroundColor = "lightgreen";
        lnameFielderr.innerHTML = "";
        lnamecorr = true;
    } else {
        lnameField.style.backgroundColor = "red";
        lnameFielderr.innerHTML = "Last name must be longer than 2 characters, should start with a capital letter and contain only letters";
        lnamecorr = false;
    }
    canRegister()
}

function validateEmail() {
    emailFielderr = document.getElementById("Email address");
    if (/^[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}$/.test(emailField.value)) {
        emailField.style.backgroundColor = "lightgreen";
        emailFielderr.innerHTML = "";
        emailcorr = true;
    } else {
        emailField.style.backgroundColor = "red";
        emailFielderr.innerHTML = "This field should contain a valid e-mail address";
        emailcorr = false;
    }
    canRegister()
}

function validatePass1() {
    pass1Fielderr = document.getElementById("Password");
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
    canRegister()
    validatePass2()
}

function validatePass2() {
    pass2Fielderr = document.getElementById("Password confirmation")
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
    canRegister()
}