// function redirectYourPage() {
//     let email = document.getElementById('email').value;
//     let password = document.getElementById('password').value;
//     let userExist = false;
//     fetch('http://localhost:5500/signin',
//         {
//             headers: {
//                 'Content-type': 'application/json'
//             },
//             method: 'POST',
//             body: JSON.stringify({ email: email, password: password })
//         }
//     )
//         .then((response) => {
//             if (response.status == 200) {
//                 userExist = true;
//                 location.href = "/home";
//             }
//             else {
//                 userExist = false;
//                 document.getElementById('lb3').style.display = 'inline';
//             }
//             return response.json();
//         })
//         .then((data) => {
//         });
// }

function checkName() {
    var name = document.getElementById("username");
    if (name.value.trim() == "") {
        document.getElementById("lb1").innerHTML = "Invalid";
        document.getElementById("lb1").style.display = "inline";
        name.style.border = "2px solid red";
        return false;
    }
    else
    {
        name.style.border = "1px solid #e1e1e1";
        document.getElementById("lb1").style.display = "none";
        return true;
    }
}

function checkPassword() {
    var password = document.getElementById("password");
    if (password.value.trim() == "") {
        password.style.border = "2px solid red";
        document.getElementById("lb3").style.display = "inline";
        return false;
    }
    else {
        password.style.border = "1px solid #e1e1e1";
        document.getElementById("lb3").style.display = "none";
        return true;
    }
}

function checkAll(event) {
    var username = checkName();
    var password = checkPassword();
    let name = document.getElementById('username').value;
    let userPassword = document.getElementById('password').value;
    if (username == true && password == true) {
        location.href = "http://localhost:8000/dla/login/";
    }
    else
    {
        event.preventDefault()
    }
}