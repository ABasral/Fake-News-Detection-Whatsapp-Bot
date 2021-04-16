import pickle


def detecting_fake_news(var):    

    load_model = pickle.load(open('model/final_model.sav', 'rb'))
    prediction = load_model.predict([var])
    prob = load_model.predict_proba([var])

    return prediction, prob
