##########################################
# Projet : DeepFace - Facial-Emotion-Recognition
# Auteur : Stéphane Meurisse
# Contact : stephane.meurisse@example.com
# Site Web : https://www.codeandcortex.fr
# LinkedIn : https://www.linkedin.com/in/st%C3%A9phane-meurisse-27339055/
# Date : 6 octobre 2024
# Licence : Ce programme est un logiciel libre : vous pouvez le redistribuer selon les termes de la Licence Publique Générale GNU v3
##########################################

# pip install opencv-python streamlit pandas numpy deepface XlsxWriter altair

import os
import cv2
import logging
import streamlit as st
import pandas as pd
import numpy as np
from deepface import DeepFace
import xlsxwriter
import altair as alt

# Configuration du logger pour Streamlit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


def afficher_log(message):
    logger.info(message)
    st.text(message)


def creer_repertoire_si_non_existant(chemin_repertoire):
    try:
        if not os.path.exists(chemin_repertoire):
            os.makedirs(chemin_repertoire)
            afficher_log(f"Répertoire créé : {chemin_repertoire}")
        else:
            afficher_log(f"Répertoire déjà existant : {chemin_repertoire}")
    except Exception as e:
        afficher_log(f"Erreur lors de la création du répertoire {chemin_repertoire}: {e}")


def extraire_images_de_la_video(video_path, output_path, interval_seconds=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        afficher_log(f"Erreur : impossible d'ouvrir la vidéo {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval_frames = int(fps * interval_seconds)

    current_frame = 0
    image_num = 0
    sous_repertoire_images = os.path.join(output_path, "images_extraites")
    creer_repertoire_si_non_existant(sous_repertoire_images)

    afficher_log(f"Extraction des frames toutes les {interval_seconds} secondes...")
    afficher_log(f"Vidéo : {fps} fps, {total_frames} frames, durée totale : {total_frames / fps:.2f} secondes")

    while current_frame < total_frames:
        ret, frame = cap.read()
        if not ret:
            afficher_log(f"Erreur lors de la lecture de la frame {current_frame}")
            break

        if current_frame % interval_frames == 0:
            image_num += 1
            image_path = os.path.join(sous_repertoire_images, f"image_{image_num}.png")
            cv2.imwrite(image_path, frame)
            # afficher_log(f"Frame {current_frame} : Image sauvegardée sous {image_path}")

        current_frame += 1

    cap.release()
    afficher_log(f"Extraction des frames terminée. {image_num} images extraites.")

    return sous_repertoire_images


def analyser_emotions(chemin_images, output_path):
    fichier_excel = os.path.join(output_path, "emotions_scores.xlsx")
    workbook = xlsxwriter.Workbook(fichier_excel)
    worksheet = workbook.add_worksheet()
    worksheet.write(0, 0, 'Image')
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'dominant_emotion']

    for idx, emotion in enumerate(emotions):
        worksheet.write(0, idx + 1, emotion)

    row = 1
    emotion_data = {emotion: [] for emotion in emotions}

    if not os.path.exists(chemin_images):
        afficher_log(f"Erreur : le répertoire des images n'existe pas {chemin_images}")
        return pd.DataFrame()

    for image_filename in os.listdir(chemin_images):
        image_path = os.path.join(chemin_images, image_filename)

        if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            afficher_log(f"Fichier ignoré : {image_filename}")
            continue

        # afficher_log(f"Analyse des émotions pour l'image : {image_path}")

        try:
            result = DeepFace.analyze(img_path=image_path,
                                      actions=['emotion'],
                                      enforce_detection=False,
                                      detector_backend='opencv')
            # afficher_log(f"Résultat brut de DeepFace : {result}")

            if isinstance(result, list) and len(result) > 0:
                result = result[0]

            if isinstance(result, dict) and 'emotion' in result:
                emotion_scores = result['emotion']
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                # afficher_log(f"Émotion dominante calculée : {dominant_emotion}")
                # afficher_log(f"Scores des émotions : {emotion_scores}")

                worksheet.write(row, 0, image_filename)
                for idx, emotion in enumerate(emotions[:-1]):
                    score = emotion_scores.get(emotion, 0)
                    worksheet.write(row, idx + 1, score)
                    emotion_data[emotion].append(score)
                worksheet.write(row, len(emotions), dominant_emotion)
                emotion_data['dominant_emotion'].append(dominant_emotion)
                row += 1

                # Affichage des émotions sur l'image
                img = cv2.imread(image_path)
                face_region = result.get('region', {})
                x, y, w, h = face_region.get('x', 0), face_region.get('y', 0), face_region.get('w', 0), face_region.get(
                    'h', 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                emotion_text = f"{dominant_emotion}: {emotion_scores[dominant_emotion]:.2f}%"
                cv2.putText(img, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                output_image_path = os.path.join(output_path, f"annotated_{image_filename}")
                cv2.imwrite(output_image_path, img)
                # afficher_log(f"Image annotée sauvegardée : {output_image_path}")
            else:
                afficher_log(f"Aucune émotion détectée pour {image_filename}.")
        except Exception as e:
            afficher_log(f"Erreur détaillée lors de l'analyse de {image_filename}: {str(e)}")
            afficher_log(f"Type d'erreur : {type(e).__name__}")

    workbook.close()
    afficher_log(f"Analyse des émotions terminée. Scores enregistrés dans {fichier_excel}.")

    # Vérifier que toutes les listes ont la même longueur
    min_length = min(len(v) for v in emotion_data.values() if v)
    emotion_data = {k: v[:min_length] for k, v in emotion_data.items() if v}

    return pd.DataFrame(emotion_data)


def quantifier_emotions_dominantes(df):
    """Reconstruire un DataFrame basé sur le nombre d'occurrences des émotions dominantes."""
    emotion_counts = df['dominant_emotion'].value_counts().reset_index()
    emotion_counts.columns = ['dominant_emotion', 'count']
    return emotion_counts


# Interface Streamlit
st.title("FER - Analyse d'émotions avec DeepFace")

# Test avec une seule image
st.header("Test de DeepFace avec une image")
test_image_path = st.text_input("Chemin vers une image de test", "chemin/vers/une/image/test.jpg")
if st.button("Tester DeepFace"):
    try:
        result = DeepFace.analyze(img_path=test_image_path,
                                  actions=['emotion'],
                                  enforce_detection=False,
                                  detector_backend='opencv')
        st.write(f"Résultat du test : {result}")
    except Exception as e:
        st.write(f"Erreur lors du test : {str(e)}")

# Répertoire de sortie pour les images et résultats
st.header("Répertoire de sortie")
repertoire_sortie = st.text_input("Spécifiez le répertoire de sortie", "output")

if not os.path.exists(repertoire_sortie):
    os.makedirs(repertoire_sortie)

# Drag and Drop pour uploader une vidéo
st.header("Déposez votre fichier MP4")
fichier_video_local = st.file_uploader("Déposez un fichier vidéo local (MP4)", type=["mp4"])

if fichier_video_local:
    fichier_video_path = os.path.join(repertoire_sortie, fichier_video_local.name)
    with open(fichier_video_path, "wb") as f:
        f.write(fichier_video_local.getbuffer())

    st.write(f"Vidéo locale enregistrée avec succès : {fichier_video_path}")

    # Lancer l'extraction des images
    if st.button("Lancer l'extraction des images"):
        chemin_images = extraire_images_de_la_video(fichier_video_path, repertoire_sortie, interval_seconds=1)

        if chemin_images:
            # Lancer l'analyse des émotions
            df_emotions = analyser_emotions(chemin_images, repertoire_sortie)

            if not df_emotions.empty:
                st.write("Analyse des émotions terminée.")
                st.dataframe(df_emotions)

                # Graphique 1 : Line Chart with Datum (secondes remplacées)
                st.header("Graphique 1 : Évolution des émotions - Line Chart ")
                st.caption("Ce graphique montre les scores totaux pour chaque émotion à chaque seconde.")
                seconds = list(range(len(df_emotions)))
                df_reset = df_emotions.reset_index().melt('index', var_name='Emotion', value_name='Score')

                # Ajout de la vérification des données
                # st.write("Vérification des données pour le Graphique 2 :")
                # st.dataframe(df_reset)  # Vérifier les données utilisées pour le graphique

                # Création du Graphique 1
                line_chart = alt.Chart(df_reset).mark_line().encode(
                    x=alt.X('index:Q', title='Secondes'),
                    y=alt.Y('Score:Q', title='Score d\'émotion'),
                    color='Emotion:N'
                )
                st.altair_chart(line_chart, use_container_width=True)
                line_chart.save(os.path.join(repertoire_sortie, "line_chart_emotions.png"))

                # Vérifier si les données contiennent des valeurs manquantes et les remplir avec 0
                df_reset.fillna(0, inplace=True)

                # Graphique 2 : Streamgraph des émotions sur 1 seconde (sans lissage)
                st.header("Graphique 2 : Streamgraph des émotions sur 1 seconde")
                st.caption("Ce graphique montre les variations des émotions à chaque seconde (sans lissage).")

                # Utilisation de stack=True au lieu de 'center'
                streamgraph = alt.Chart(df_reset).mark_area().encode(
                    alt.X('index:Q', title='Secondes'),
                    alt.Y('Score:Q', stack=True, title='Score d\'émotion'),
                    alt.Color('Emotion:N')
                )

                st.altair_chart(streamgraph, use_container_width=True)
                streamgraph.save(os.path.join(repertoire_sortie, "streamgraph_1s.png"))

                # Graphique 3 : Bar Chart with Rounded Edges (Barres avec arrondis)
                st.header("Graphique 3 : Bar Chart")
                st.caption(
                    "Ce graphique affiche les scores d'émotions totaux pour chaque émotion sur toutes les images.")
                bar_chart = alt.Chart(df_reset).mark_bar(
                    cornerRadiusTopLeft=3,
                    cornerRadiusTopRight=3
                ).encode(
                    x=alt.X('Emotion:N', title='Émotion'),
                    y=alt.Y('sum(Score):Q', title='Score total'),
                    color='Emotion:N'
                )
                st.altair_chart(bar_chart, use_container_width=True)
                bar_chart.save(os.path.join(repertoire_sortie, "bar_chart_rounded_edges.png"))

                # Graphique 4 : Courbes lissées sur 5 secondes (Lissage sur 5 secondes)
                st.header("Graphique 4 : Courbes lissées sur 5 secondes")
                st.caption("Le DataFrame est lissé sur une fenêtre de 5 secondes avec .rolling(window=5).mean(). "
                           "Chaque valeur sera la moyenne des 5 dernières secondes pour lisser les données.")
                df_smoothed = df_emotions.drop(columns=['dominant_emotion']).rolling(
                    window=5).mean()  # Lissage sur une fenêtre de 5 secondes
                df_reset_smoothed = df_smoothed.reset_index().melt('index', var_name='Emotion', value_name='Score')
                line_chart_smoothed = alt.Chart(df_reset_smoothed).mark_line().encode(
                    x=alt.X('index:Q', title='Secondes'),
                    y=alt.Y('Score:Q', title='Score d\'émotion lissé (5 secondes)'),
                    color='Emotion:N'
                )
                st.altair_chart(line_chart_smoothed, use_container_width=True)
                line_chart_smoothed.save(os.path.join(repertoire_sortie, "emotions_lissees_5s.png"))

                # Graphique supplémentaire : Graphique de flux des émotions lissées sur 5 secondes (streamgraph)
                st.header("Graphique 5 : Streamgraph des émotions lissées sur 5 secondes")
                st.caption("Ce graphique montre les variations des émotions sur une fenêtre de lissage de 5 secondes.")
                streamgraph_smoothed = alt.Chart(df_reset_smoothed).mark_area().encode(
                    alt.X('index:Q', title='Secondes'),
                    alt.Y('Score:Q', stack='center', title='Score d\'émotion lissé (5 secondes)'),
                    alt.Color('Emotion:N')
                )
                st.altair_chart(streamgraph_smoothed, use_container_width=True)
                streamgraph_smoothed.save(os.path.join(repertoire_sortie, "streamgraph_lisse_5s.png"))

                # Graphique 6 : Bar Chart des émotions dominantes
                st.header("Graphique 6 : Bar Chart des émotions dominantes")
                emotion_counts = quantifier_emotions_dominantes(df_emotions)
                dominant_bar_chart = alt.Chart(emotion_counts).mark_bar().encode(
                    x=alt.X('dominant_emotion:N', title='Émotion dominante'),
                    y=alt.Y('count:Q', title='Nombre d\'occurrences'),
                    color='dominant_emotion:N'
                )
                st.altair_chart(dominant_bar_chart, use_container_width=True)
                dominant_bar_chart.save(os.path.join(repertoire_sortie, "bar_chart_dominant_emotion.png"))

                afficher_log(f"Extraction et analyse des images terminées.")
                afficher_log(f"Graphiques exportés dans le répertoire : {repertoire_sortie}")
            else:
                st.write("Aucune émotion n'a pu être analysée.")


