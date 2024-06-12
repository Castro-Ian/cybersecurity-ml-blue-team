document.getElementById('prediction-form').addEventListener('submit', async function(event) {
    event.preventDefault();
    const features = document.getElementById('features').value.split(',').map(Number);
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ input: features })
    });
    const result = await response.json();
    document.getElementById('prediction-result').textContent = `Prediction: ${result.prediction}`;
});

document.getElementById('send-button').addEventListener('click', async function() {
    const userInput = document.getElementById('user-input').value;
    const response = await fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userInput })
    });
    const result = await response.json();
    const messageContainer = document.createElement('div');
    messageContainer.textContent = `Bot: ${result.response}`;
    document.getElementById('messages').appendChild(messageContainer);
});
