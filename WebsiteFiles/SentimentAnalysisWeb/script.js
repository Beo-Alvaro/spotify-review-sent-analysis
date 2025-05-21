function analyzeSentiment() {
    const review = document.getElementById("review").value;
    const result = document.getElementById("result");

    if (!review) {
        alert("Please enter some text.");
        return;
    }

    fetch("http://127.0.0.1:7423/predict", {
        method: 'POST',
        mode: 'cors',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ review: review })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);  // Log the response for debugging
        
        // Convert sentiment string to number
        const sentiment = parseInt(data.sentiment);

        // Map numeric sentiment to string labels
        let sentimentLabel = '';
        if (sentiment === 0) {
            sentimentLabel = 'Negative';
        } else if (sentiment === 1) {
            sentimentLabel = 'Neutral';
        } else if (sentiment === 2) {
            sentimentLabel = 'Positive';
        }

        // Display the sentiment
        result.innerHTML = `<h3>Sentiment:</h3><p>${sentimentLabel}</p>`;
        result.className = `sentiment-result ${sentimentLabel.toLowerCase()}`;
    })
    .catch(error => {
        console.error('Error: ', error);
        result.textContent = 'Please, Try again';
    });
}