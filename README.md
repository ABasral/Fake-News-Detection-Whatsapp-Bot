# Fake-News-Detection-Whatsapp-Bot

“Fake news” is a term that has come to mean different things to different people. At its core, we are defining
“fake news” as those news stories that are false: the story itself is fabricated, with no verifiable facts, sources
or quotes. Sometimes these stories may be propaganda that is intentionally designed to mislead the reader, or
may be designed as “clickbait” written for economic incentives (the writer profits on the number of people
who click on the story). In recent years, fake news stories have proliferated via social media, in part because
they are so easily and quickly shared online.
Fake news is an invention – a lie created out of nothing – that takes the appearance of real news with the aim
of deceiving people. This is what is important to remember: the information is false, but it seems true.

Requirements:
-------------

- A Twilio account
- A Twilio whatsapp sandbox
- Python 3
- Flask
- ngork
- Tensorflow


We are using the LIAR Dataset by William Yang Wang which he used in his research paper titled "Liar, Liar
Pants on Fire": A New Benchmark Dataset for Fake News Detection.
The original dataset comes with following columns:
- Column 1: the ID of the statement ([ID].json)
- Column 2: the label
- Column 3: the statement
- Column 4: the subject(s)
- Column 5: the speaker
- Column 6: the speaker's job title
- Column 7: the state info
- Column 8: the party affiliation
- Column 9-13: the total credit history count, including the current statement
- 9: barely true counts
- 10: false counts
- 11: half true counts
- 12: mostly true counts
- 13: pants on fire counts
- Column 14: the context (venue / location of the speech or statement)
For the simplicity, we have converted it to 2 column format:
- Column 1: Statement (News headline or text)
- Column 2: Label (Label class contains: True, False)

Running the app:
----------------

1. Inside the project directory run `Run app.py`

2. Your Flask app will need to be visible from the web so Twilio can send requests to it. Ngrok lets us do this. With it installed, run the following command in your terminal in the directory your code is in. Run `ngrok http 5000` in a new terminal tab.

3. Grab that ngrok URL to configure twilio whatsapp sandbox.


And we’re good to go! Let’s test our application on WhatsApp! We can send some news headlines or facts to this sandbox and get predictions in return if everything works as expected.


