import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Modell und Labels laden (Cache, um nur einmal zu laden)
@st.cache(allow_output_mutation=True)
def lade_modell_und_labels():
    modell = load_model("keras_Model.h5", compile=False)
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return modell, labels

modell, labels = lade_modell_und_labels()

# Streamlit Seiteneinstellungen
st.set_page_config(
    page_title="T-Shirt Farberkennung",
    layout="centered"
)

st.title("🌈 T-Shirt Farberkennung für Menschen mit Farbenblindheit")
st.write(
    "Laden Sie ein Bild Ihres T-Shirts hoch. Das System erkennt, ob es rot, blau oder schwarz ist, "
    "und zeigt die Zuverlässigkeit in Prozent an.\n\n"
    "Die Ergebnisse sind deutlich in Textform dargestellt, um die Zugänglichkeit zu gewährleisten."
)

# Upload-Fenster
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Bild öffnen und anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bild auf 224x224 skalieren
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # Bild in numpy-Array umwandeln
    image_array = np.asarray(image_resized)

    # Normalisieren
    normalized_image = (image_array.astype(np.float32) / 127.5) - 1

    # Daten vorbereiten
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image

    # Vorhersage
    prediction = modell.predict(data)[0]
    max_confidence = np.max(prediction)
    max_index = np.argmax(prediction)
    predicted_label = labels[max_index]

    # Ergebnis interpretieren
    if max_confidence > 0.5:
        confidence_percent = round(max_confidence * 100, 2)
        if predicted_label.lower() == "rot":
            ergebnis_text = f"🟥 Rotes T-Shirt erkannt, Zuverlässigkeit: {confidence_percent} %."
        elif predicted_label.lower() == "blau":
            ergebnis_text = f"🔵 Blaues T-Shirt erkannt, Zuverlässigkeit: {confidence_percent} %."
        elif predicted_label.lower() == "schwarz":
            ergebnis_text = f"⚫ Schwarzes T-Shirt erkannt, Zuverlässigkeit: {confidence_percent} %."
        else:
            ergebnis_text = "Die erkannte Farbe ist unklar. Bitte laden Sie ein anderes Foto hoch."
    else:
        ergebnis_text = "🛑 Die Farbe des T-Shirts ist nicht zu erkennen. Bitte laden Sie ein anderes Foto hoch."

    # Ergebnis anzeigen
    st.markdown(f"## Ergebnis\n\n{ergebnis_text}")
else:
    st.info("Bitte laden Sie ein Bild hoch, um die Erkennung durchzuführen.")

# Hinweise zur Barrierefreiheit
st.markdown("""
---

**Hinweis zur Barrierefreiheit:**  
Diese Anwendung ist so gestaltet, dass sie für Menschen mit Farbenblindheit zugänglich ist.  
Die Ergebnisse werden klar in Textform dargestellt, mit großer Schrift und deutlichen Symbolen, um die Verständlichkeit zu erhöhen.  
Für eine bessere Zugänglichkeit empfehlen wir die Nutzung auf Geräten mit Sprachausgaben oder Screenreadern.
""")
