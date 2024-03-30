document.addEventListener('DOMContentLoaded', function() {
    const recordButton = document.getElementById('recordButton');
    const playButton = document.getElementById('playButton');
    const resultContainer = document.getElementById('resultContainer');
    const BACKEND_URL = window.origin

    let mediaRecorder;
    let audioChunks = [];
    let recordedAudioBlob;

    recordButton.addEventListener('click', startRecording);
    playButton.addEventListener('click', playRecording);

    async function startRecording() {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);

        mediaRecorder.ondataavailable = function(event) {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = function() {
            recordedAudioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            sendAudioToServer(recordedAudioBlob);
            audioChunks = [];
            playButton.disabled = false;  // Enable play button after recording
        };

        mediaRecorder.start();
        recordButton.textContent = 'Stop Recording';
        recordButton.removeEventListener('click', startRecording);
        recordButton.addEventListener('click', stopRecording);
    }

    function stopRecording() {
        mediaRecorder.stop();
        recordButton.textContent = 'Record';
        recordButton.removeEventListener('click', stopRecording);
        recordButton.addEventListener('click', startRecording);
    }

    function playRecording() {
        const audio = new Audio(URL.createObjectURL(recordedAudioBlob));
        audio.play();
    }

    async function sendAudioToServer(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'audio.wav');

        try {
            const response = await fetch(`${BACKEND_URL}/speech-to-text/`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();  // Parse the response as JSON
            console.log('Speech-to-Text Result:', result);

            // Display the result on the webpage
            if (result.result && result.result.length > 0) {
                resultContainer.textContent = result.result[0];
            } else {
                resultContainer.textContent = 'No transcription available.';
            }
        } catch (error) {
            console.error('Error sending audio to server:', error);
            resultContainer.textContent = 'Unable to transcribe audio.';
        }
    }

    function showNotification(message, type) {
        if (Notification.permission === 'granted') {
            new Notification(message, { icon: 'path/to/your/icon.png' });
        } else if (Notification.permission !== 'denied') {
            Notification.requestPermission().then(permission => {
                if (permission === 'granted') {
                    new Notification(message, { icon: 'path/to/your/icon.png' });
                }
            });
        }
    }

    const modelSelect = document.getElementById('modelSelect');
    const getConfigButton = document.getElementById('getConfigButton');
    const configContainer = document.getElementById('configContainer');

    // Populate the model select dropdown
    fetch(`${BACKEND_URL}/get-model-names/`)
        .then(response => response.json())
        .then(modelNames => {
            modelNames.forEach(modelName => {
                const option = document.createElement('option');
                option.value = modelName;
                option.textContent = modelName;
                modelSelect.appendChild(option);
            });
        })
        .catch(error => console.error('Error fetching model names:', error));

        getConfigButton.addEventListener('click', async function() {
            const selectedModel = modelSelect.value;

            try {
                // Fetch the configuration for the selected model
                const response = await fetch(`${BACKEND_URL}/get-config/?model_name=${encodeURIComponent(selectedModel)}`);
                const responseJson = await response.json();

                // Display the configuration on the webpage
                configContainer.innerHTML = '<h3>Model Configuration:</h3>';
                const ul = document.createElement('ul');

                // Create a dropdown menu for each item in the response
                responseJson.items.forEach(item => {
                    const li = document.createElement('li');
                    li.textContent = `${item.name}: `;

                    const select = createDropdown(item.options, item.attributes_name);
                    li.appendChild(select);
                    ul.appendChild(li);
                });

                configContainer.appendChild(ul);

                // Add button to change model config
                const changeConfigButton = createChangeConfigButton(selectedModel, responseJson);
                configContainer.appendChild(changeConfigButton);
            } catch (error) {
                console.error('Error fetching model config:', error);
                configContainer.innerHTML = '<p>Error fetching model config.</p>';
            }
        });

        function createDropdown(options, key) {
            const select = document.createElement('select');
            options.forEach(optionValue => {
                const option = document.createElement('option');
                option.value = optionValue;
                option.textContent = optionValue;
                select.appendChild(option);
            });
            return select;
        }

        function createChangeConfigButton(selectedModel, responseJson) {
            const changeConfigButton = document.createElement('button');
            changeConfigButton.textContent = 'Change Config';

            changeConfigButton.addEventListener('click', async function() {
                const newConfig = { items: [] };

                // Get user-selected values for each dropdown
                configContainer.querySelectorAll('li').forEach(li => {
                    const itemName = li.textContent.split(':')[0].trim();
                    const selectedItem = li.querySelector('select').value;
                    const item = responseJson.items.find(item => item.name === itemName);
                    if (item) {
                        newConfig.items.push({
                            name: item.name,
                            descriptions: item.descriptions,
                            attributes_name: item.attributes_name,
                            options: [selectedItem]
                        });
                    }
                });

                try {
                    // Send new config to server
                    const response = await fetch(`${BACKEND_URL}/change-model/?model_name=${encodeURIComponent(selectedModel)}`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'accept': 'application/json',
                        },
                        body: JSON.stringify(newConfig),
                    });

                    const result = await response.json();
                    console.log('Model config changed:', result);
                    showNotification('Model config changed successfully!', 'success');
                } catch (error) {
                    console.error('Error changing model config:', error);
                    showNotification('Error changing model config.', '');
                }
            });

            return changeConfigButton;
        }
});
