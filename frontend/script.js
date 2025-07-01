function sendQuery() {
    const query = document.getElementById('userQuery').value;
    fetch('http://localhost:5005/webhooks/rest/webhook', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ sender: "user", message: query })
    })
    .then(response => response.json())
    .then(data => {
        const resDiv = document.getElementById('response');
        if (data && data.length > 0 && data[0].text) {
            resDiv.innerText = data[0].text;
        } else {
            resDiv.innerText = "No response from Rasa.";
        }
    })
    .catch(err => {
        document.getElementById('response').innerText = "Error: " + err;
        console.error(err);
    });
}
