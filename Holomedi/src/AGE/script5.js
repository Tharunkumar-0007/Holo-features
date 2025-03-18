let currentQuestionIndex = 0;
let answers = [];

// Backend URL (Update this if the backend runs on a different host/port)
const backendUrl = "http://127.0.0.1:5000";

const questionBox = document.getElementById("question-box");
const resultBox = document.getElementById("result-box");

async function loadQuestions() {
    try {
        const response = await fetch(`${backendUrl}/questions`);
        const questions = await response.json();
        showQuestion(questions[currentQuestionIndex], questions);
    } catch (error) {
        console.error("Failed to load questions", error);
    }
}

function showQuestion(question, questions) {
    questionBox.innerHTML = `
        <h2>Question ${question.id}</h2>
        <p>${question.question}</p>
        <div class="options">
            ${question.options
                .map(
                    (option) => `
                    <label>
                        <input type="radio" name="answer" value="${option}" 
                            ${answers[currentQuestionIndex] === option ? "checked" : ""} required>
                        ${option}
                    </label>
                `
                )
                .join("")}
        </div>
        <div class="buttons">
            <button id="back-button" onclick="previousQuestion()" ${currentQuestionIndex === 0 ? 'disabled' : ''}>Back</button>
            <button onclick="nextQuestion(${questions.length})">Next</button>
        </div>
    `;
}

function nextQuestion(totalQuestions) {
    const selectedOption = document.querySelector('input[name="answer"]:checked');
    if (!selectedOption) {
        alert("Please select an answer!");
        return;
    }

    answers[currentQuestionIndex] = selectedOption.value;
    currentQuestionIndex++;

    if (currentQuestionIndex < totalQuestions) {
        fetch(`${backendUrl}/questions`)
            .then((response) => response.json())
            .then((questions) => {
                showQuestion(questions[currentQuestionIndex], questions);
            });
    } else {
        showResult();
    }
}

function previousQuestion() {
    const selectedOption = document.querySelector('input[name="answer"]:checked');
    if (selectedOption) {
        answers[currentQuestionIndex] = selectedOption.value;
    }

    currentQuestionIndex--;

    fetch(`${backendUrl}/questions`)
        .then((response) => response.json())
        .then((questions) => {
            showQuestion(questions[currentQuestionIndex], questions);
        });
}

async function showResult() {
    try {
        const response = await fetch(`${backendUrl}/submit`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ answers }),
        });
        const result = await response.json();
        questionBox.classList.add("hidden");
        resultBox.classList.remove("hidden");
        resultBox.innerHTML = `
            <h2>Your Mental Age</h2>
            <p>Your mental age is: <strong>${result.mental_age}</strong> years!</p>
            <p>You are classified as: <strong>${result.category}</strong></p>
            <button onclick="window.location.reload()">Try Again</button>
        `;
    } catch (error) {
        console.error("Failed to calculate result", error);
    }
}

window.onload = loadQuestions;
