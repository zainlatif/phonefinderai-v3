from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import joblib
import os
import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/classifier.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '../model/vectorizer.pkl')
PHONE_SPECS_PATH = os.path.join(os.path.dirname(__file__), '../data/phone_specs.csv')

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
phone_specs = pd.read_csv(PHONE_SPECS_PATH)

class ActionFindPhone(Action):
    def name(self) -> Text:
        return "action_find_phone"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        feature = next(tracker.get_latest_entity_values("feature"), "")
        brand = next(tracker.get_latest_entity_values("brand"), "")
        price = next(tracker.get_latest_entity_values("price"), "")

        query_text = f"{feature} {brand} {price}"
        X = vectorizer.transform([query_text])
        category = model.predict(X)[0]

        # Filter recommendations by category, price, and brand if provided
        filtered = phone_specs[phone_specs['category'] == category]
        if price:
            if '-' in price:
                try:
                    low, high = map(int, price.split('-'))
                    filtered = filtered[(filtered['price'] >= low) & (filtered['price'] <= high)]
                except Exception:
                    pass
            elif price.startswith('<'):
                try:
                    high = int(price[1:])
                    filtered = filtered[filtered['price'] <= high]
                except Exception:
                    pass
            elif price.startswith('>'):
                try:
                    low = int(price[1:])
                    filtered = filtered[filtered['price'] >= low]
                except Exception:
                    pass
        if brand:
            filtered = filtered[filtered['name'].str.contains(brand, case=False)]

        recommendations = filtered['name'].tolist()
        if recommendations:
            message = f"Recommended phones: {', '.join(recommendations[:5])}"
        else:
            message = "Sorry, I couldn't find any phones matching your criteria."

        dispatcher.utter_message(text=message)
        return []
