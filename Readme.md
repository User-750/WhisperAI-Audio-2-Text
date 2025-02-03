A Very Basic Audio to Text Converter.

This Project uses the Whisper Automatic Speech Recognition AI model from OpenAI. 

You need to run a Few Commands before you can start using this.

1. Create a Virtual Environment. Not Necessary but i recommend that you do it. 
python -m venv venv
venv\Scripts\activate  

2. Run the command below to Install the Necessary packages.
pip install transformers torch librosa

3. Change the location to your Project's Location i.e. Where your Whisper.py File is at Line 116. 

4. IMPORTANT: As of now, only MP3 Files work. More Functionalities & Support will be added in the future.

5. Put the MP3 file in the same folder as the Whisper.py file. The MP3 File has to be named, audiofile.mp3. As i said this is experimental and more functionalities will be added in the future. 

6. When you run Whisper.py for the first time, it will pull Whisper-Large-v3 Model from Huggingface Repository. This is about 3gb and will take some time. but this is only for the first time. 

7. After all of this setup and assuming everything worked fine, you should see a audiofile.txt in the same folder. This folder contains the speech that was transcribed. 

8. I have included a audiofile.mp3 for a sample test. you can use it to test the working. 

9. Thank You. 
