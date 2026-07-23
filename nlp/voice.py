"""
Voice Assistant Module.
Browser-based speech recognition (STT) and text-to-speech (TTS)
using Web Speech API via Streamlit components.
"""

import streamlit.components.v1 as components


def get_voice_input_component(height: int = 120) -> None:
    """
    Render a browser-based voice input widget using Web Speech API.
    The recognized text is sent back to Streamlit via query params.
    """
    html_code = """
    <div style="text-align: center; padding: 10px;">
        <button id="voiceBtn" onclick="startListening()" 
                style="padding: 12px 28px; font-size: 16px;
                       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                       color: white; border: none; border-radius: 25px;
                       cursor: pointer; transition: all 0.3s ease;
                       box-shadow: 0 4px 15px rgba(102,126,234,0.4);">
            🎤 Click to Speak
        </button>
        <p id="voiceStatus" style="color: #888; margin-top: 8px; font-size: 14px;">
            Click the button and speak your query
        </p>
        <p id="voiceResult" style="color: #4CAF50; font-weight: bold; 
           font-size: 16px; margin-top: 5px;"></p>
    </div>

    <script>
    function startListening() {
        if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
            document.getElementById('voiceStatus').innerText = 
                '❌ Speech recognition not supported in this browser.';
            return;
        }

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.lang = 'en-IN';
        recognition.interimResults = false;
        recognition.maxAlternatives = 1;

        document.getElementById('voiceBtn').style.background = 
            'linear-gradient(135deg, #f5576c 0%, #ff6e7f 100%)';
        document.getElementById('voiceBtn').innerText = '🔴 Listening...';
        document.getElementById('voiceStatus').innerText = 'Listening... Speak now!';

        recognition.start();

        recognition.onresult = function(event) {
            const text = event.results[0][0].transcript;
            document.getElementById('voiceResult').innerText = '📝 "' + text + '"';
            document.getElementById('voiceStatus').innerText = 'Got it! Processing...';
            
            // Send to Streamlit via session storage
            window.parent.postMessage({type: 'voice_input', text: text}, '*');
            
            // Also store in a way Streamlit can access
            const stInput = window.parent.document.querySelector(
                'input[data-testid="stChatInputTextArea"], textarea[data-testid="stChatInputTextArea"]'
            );
            if (stInput) {
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype, 'value').set || 
                    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                nativeInputValueSetter.call(stInput, text);
                stInput.dispatchEvent(new Event('input', { bubbles: true }));
            }
        };

        recognition.onerror = function(event) {
            document.getElementById('voiceStatus').innerText = '❌ Error: ' + event.error;
            resetBtn();
        };

        recognition.onend = function() {
            resetBtn();
        };
        
        function resetBtn() {
            document.getElementById('voiceBtn').style.background = 
                'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            document.getElementById('voiceBtn').innerText = '🎤 Click to Speak';
        }
    }
    </script>
    """
    components.html(html_code, height=height)


def text_to_speech_component(text: str, lang: str = "en-IN", height: int = 0) -> None:
    """
    Play text-to-speech in the browser using Web Speech Synthesis API.
    """
    # Escape text for JavaScript
    safe_text = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")

    html_code = f"""
    <script>
    (function() {{
        if ('speechSynthesis' in window) {{
            // Cancel any ongoing speech
            window.speechSynthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance('{safe_text}');
            utterance.lang = '{lang}';
            utterance.rate = 0.9;
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            
            // Small delay to ensure voices are loaded
            setTimeout(() => {{
                window.speechSynthesis.speak(utterance);
            }}, 200);
        }}
    }})();
    </script>
    """
    components.html(html_code, height=height)
