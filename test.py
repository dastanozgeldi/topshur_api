import requests


def test_transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        files = {"file": (file_path, audio_file, "audio/wav")}
        url = "http://127.0.0.1:8000/transcribe"
        response = requests.post(url, files=files)
        if response.status_code == 200:
            result = response.json()
            print("Transcription result:", result)
        else:
            print(f"Failed to transcribe audio. Status code: {response.status_code}")


if __name__ == "__main__":
    test_audio_path = "dataset/2.wav"
    test_transcribe_audio(test_audio_path)
