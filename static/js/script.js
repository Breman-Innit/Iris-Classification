
const form = document.getElementById("prediction-form");
const predictButton = document.getElementById("predict-button");

const result = document.getElementById("result");
const species = document.getElementById("species");
const confidence = document.getElementById("confidence");
const errorMessage = document.getElementById("error-message");
const resetButton = document.getElementById("reset-button");

resetButton.addEventListener("click", function () {
    form.reset();
    result.classList.add("hidden");
    errorMessage.classList.add("hidden");
    form.scrollIntoView({ behavior: "smooth" });
});

form.addEventListener("submit", async function (event) {
    event.preventDefault();

    // Hide previous results and errors
    result.classList.add("hidden");
    errorMessage.classList.add("hidden");

    // Collect flower measurements
    const measurements = {
        sepal_length: Number(document.getElementById("sepal_length").value),
        sepal_width: Number(document.getElementById("sepal_width").value),
        petal_length: Number(document.getElementById("petal_length").value),
        petal_width: Number(document.getElementById("petal_width").value)
    };

    try {
        predictButton.disabled = true;
        predictButton.textContent = "Predicting...";

        // Send measurements to the Flask backend
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(measurements)
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        // Display prediction results
        species.textContent = data.prediction;
        confidence.textContent = `Confidence: ${data.confidence}%`;

        result.classList.remove("hidden");

    } catch (error) {
        errorMessage.textContent = error.message;
        errorMessage.classList.remove("hidden");

    } finally {
        predictButton.disabled = false;
        predictButton.textContent = "Predict Species";
    }
});
