# Basic libraries
import time
import tweepy
import json
import numpy as np
import os
import preprocessor as p

# Emoji libraries. Requires pyTorch == 1.0.1
import emoji

from bot.torchMoji.torchmoji.sentence_tokenizer import SentenceTokenizer
from bot.torchMoji.torchmoji.model_def import torchmoji_emojis
from bot.torchMoji.torchmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH


consumer_key = os.getenv("consumer_key")
consumer_secret = os.getenv("consumer_secret")
access_token = os.getenv("access_token")
access_token_secret = os.getenv("access_token_secret")

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
auth.secure = True


EMOJIS = ":joy: :unamused: :weary: :sob: :heart_eyes: \
:pensive: :ok_hand: :blush: :heart: :smirk: \
:grin: :notes: :flushed: :100: :sleeping: \
:relieved: :relaxed: :raised_hands: :two_hearts: :expressionless: \
:sweat_smile: :pray: :confused: :kissing_heart: :heartbeat: \
:neutral_face: :information_desk_person: :disappointed: :see_no_evil: :tired_face: \
:v: :sunglasses: :rage: :thumbsup: :cry: \
:sleepy: :yum: :triumph: :hand: :mask: \
:clap: :eyes: :gun: :persevere: :smiling_imp: \
:sweat: :broken_heart: :yellow_heart: :musical_note: :speak_no_evil: \
:wink: :skull: :confounded: :smile: :stuck_out_tongue_winking_eye: \
:angry: :no_good: :muscle: :facepunch: :purple_heart: \
:sparkling_heart: :blue_heart: :grimacing: :sparkles:".split(' ')


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


def find_emoji(tweet):
    tweet = p.clean(tweet)

    # Tokenizing using dictionary
    with open(VOCAB_PATH, 'r') as f:
        vocabulary = json.load(f)

    st = SentenceTokenizer(vocabulary, 30)

    # Loading model
    model = torchmoji_emojis(PRETRAINED_PATH)
    # Running predictions
    tokenized, _, _ = st.tokenize_sentences([tweet])
    # Get sentence probability
    prob = model(tokenized)[0]

    # Top emoji id
    emoji_id = top_elements(prob, 1)

    # map to emojis
    tweet_emoji = map(lambda x: EMOJIS[x], emoji_id)
    return emoji.emojize(' '.join(tweet_emoji), use_aliases=True)


class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api):
        super().__init__()
        self.api = api

    def on_status(self, status):
        if len(str(status.text).split(" ")) > 0 and "@realdonaldtrump" not in str(status.text).lower() \
                and not status.in_reply_to_status_id and not status.retweeted \
                and not hasattr(status, 'retweeted_status'):
            tweet_emoji = find_emoji(status.text)
            self.api.update_status(tweet_emoji + " https://twitter.com/realdonaldtrump/status/" + str(status.id))
            print("Tweeted! Sleeping for 30 mins to not spam...")
            time.sleep(60*30)
            print("Woke up :D")
        else:
            pass


def main():
    api = tweepy.API(auth)

    listener = MyStreamListener(api)
    stream = tweepy.Stream(auth=api.auth, listener=listener)
    stream.filter(follow=[str(api.get_user(screen_name='realDonaldTrump').id)])


if __name__ == '__main__':
    main()
