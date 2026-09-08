import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ----------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# ----------------------------------------------------
st.set_page_config(
    page_title="Clasificador de ODS con LSTM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base de datos con información de los 16 ODS soportados por el modelo.
# Se usan tonos suaves (pasteles) en lugar de los colores oficiales saturados.
ODS_INFO = {
    1: {"nombre": "Fin de la pobreza", "color": "#E8A0A8"},
    2: {"nombre": "Hambre cero", "color": "#EFCB8E"},
    3: {"nombre": "Salud y bienestar", "color": "#9BCBA1"},
    4: {"nombre": "Educación de calidad", "color": "#DE9BA1"},
    5: {"nombre": "Igualdad de género", "color": "#F2A79B"},
    6: {"nombre": "Agua limpia y saneamiento", "color": "#96D3E3"},
    7: {"nombre": "Energía asequible y no contaminante", "color": "#F5DD9B"},
    8: {"nombre": "Trabajo decente y crecimiento económico", "color": "#C793A4"},
    9: {"nombre": "Industria, innovación e infraestructura", "color": "#F2B48D"},
    10: {"nombre": "Reducción de las desigualdades", "color": "#E39FB8"},
    11: {"nombre": "Ciudades y comunidades sostenibles", "color": "#F5C68F"},
    12: {"nombre": "Producción y consumo responsables", "color": "#D6BD8B"},
    13: {"nombre": "Acción por el clima", "color": "#96BC9A"},
    14: {"nombre": "Vida submarina", "color": "#8FC3DE"},
    15: {"nombre": "Vida de ecosistemas terrestres", "color": "#A9D194"},
    16: {"nombre": "Paz, justicia e instituciones sólidas", "color": "#8FAFCC"}
}

NUM_ODS = len(ODS_INFO)


def etiqueta_ods(numero, con_numero=True):
    """Devuelve el nombre completo del ODS, opcionalmente con su número."""
    nombre = ODS_INFO[numero]["nombre"]
    return f"ODS {numero} - {nombre}" if con_numero else nombre


# ----------------------------------------------------
# CARGA DE ARTEFACTOS DEL MODELO
# ----------------------------------------------------
# Directorio donde vive este script (app.py), independientemente de cuál sea
# el directorio de trabajo actual desde el que Streamlit lo ejecute. Todas las
# rutas de los artefactos se resuelven relativas a esto para evitar errores de
# "archivo no encontrado" cuando el cwd no coincide con la carpeta del script
# (algo común en Streamlit Community Cloud si app.py vive en una subcarpeta).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def artifact_path(filename):
    return os.path.join(BASE_DIR, filename)


@st.cache_resource
def load_artifacts():
    model = None
    tokenizer = None
    label_encoder = None

    try:
        import tensorflow as tf

        model_keras_path = artifact_path("ods_lstm_model.keras")
        model_h5_path = artifact_path("ods_lstm_model.h5")
        tokenizer_path = artifact_path("tokenizer.pickle")
        label_encoder_path = artifact_path("label_encoder.pickle")

        if os.path.exists(model_keras_path):
            model = tf.keras.models.load_model(model_keras_path)
        elif os.path.exists(model_h5_path):
            model = tf.keras.models.load_model(model_h5_path)

        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, "rb") as f:
                tokenizer = pickle.load(f)

        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, "rb") as f:
                label_encoder = pickle.load(f)

    except Exception as e:
        st.error(f"No se pudieron cargar los artefactos del modelo: {e}")

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_artifacts()


def _seq_to_probs_dict(raw_probs_row, classes):
    """Convierte una fila de probabilidades crudas del modelo en {numero_ods: probabilidad}."""
    ods_prob_dict = {int(cls): float(raw_probs_row[idx]) for idx, cls in enumerate(classes)}
    for ods_num in ODS_INFO.keys():
        if ods_num not in ods_prob_dict:
            ods_prob_dict[ods_num] = 0.0
    return ods_prob_dict


def predict_ods(text_list, max_len=50):
    """Devuelve, para cada texto, un diccionario {numero_ods: probabilidad}."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    seqs = tokenizer.texts_to_sequences(text_list)
    padded = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    probs = model.predict(padded)

    classes = label_encoder.classes_
    return [_seq_to_probs_dict(p, classes) for p in probs]


def predict_ods_debug(text, max_len=50):
    """
    Igual que predict_ods pero para un solo texto, y además devuelve la
    información intermedia del pipeline (secuencia, padding, vocabulario,
    probabilidades crudas) para poder depurar discrepancias con Colab.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    seq = tokenizer.texts_to_sequences([text])[0]
    padded = pad_sequences([seq], maxlen=max_len, padding="post", truncating="post")
    raw_probs = model.predict(padded)[0]

    classes = label_encoder.classes_
    probs_dict = _seq_to_probs_dict(raw_probs, classes)

    debug_info = {
        "texto_original": text,
        "secuencia_tokenizada": seq,
        "longitud_secuencia_sin_padding": len(seq),
        "secuencia_con_padding": padded[0].tolist(),
        "max_len_usado": max_len,
        "tamano_vocabulario_tokenizer": len(tokenizer.word_index),
        "clases_label_encoder": classes.tolist() if hasattr(classes, "tolist") else list(classes),
        "probabilidades_crudas": raw_probs.tolist(),
    }
    return probs_dict, debug_info


# ----------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------
st.title("Clasificador de Objetivos de Desarrollo Sostenible (ODS)")
st.markdown(
    f"Herramienta de procesamiento de lenguaje natural que asigna un texto a uno o varios "
    f"de los {NUM_ODS} Objetivos de Desarrollo Sostenible cubiertos por el modelo, "
    f"utilizando una red neuronal recurrente LSTM."
)

if model is None or tokenizer is None or label_encoder is None:
    try:
        archivos_en_carpeta = os.listdir(BASE_DIR)
    except Exception:
        archivos_en_carpeta = ["(no se pudo listar la carpeta)"]

    st.error(
        "No se encontró el modelo entrenado. Coloca los archivos "
        "`ods_lstm_model.keras`, `tokenizer.pickle` y `label_encoder.pickle` en la misma "
        "carpeta que este script (`app.py`) para poder usar la aplicación."
    )
    with st.expander("Información de diagnóstico"):
        st.write("Carpeta donde se está buscando los artefactos:")
        st.code(BASE_DIR)
        st.write("Archivos encontrados en esa carpeta:")
        st.code("\n".join(sorted(archivos_en_carpeta)))

        st.write("Verificación de integridad de cada archivo:")
        import zipfile

        for nombre in ["ods_lstm_model.keras", "tokenizer.pickle", "label_encoder.pickle"]:
            ruta = artifact_path(nombre)
            if not os.path.exists(ruta):
                st.write(f"- `{nombre}`: no existe en esa ruta.")
                continue

            tamano_bytes = os.path.getsize(ruta)
            linea = f"- `{nombre}`: {tamano_bytes:,} bytes"

            if nombre.endswith(".keras"):
                es_zip_valido = zipfile.is_zipfile(ruta)
                linea += " — ZIP válido" if es_zip_valido else " — **NO es un ZIP válido (archivo corrupto)**"

            st.write(linea)

        st.caption(
            "Compara el tamaño en bytes de cada archivo contra el que ves en GitHub o en tu "
            "máquina local/Colab. Si el tamaño no coincide, o si el .keras no es un ZIP "
            "válido, el archivo se corrompió al subirlo o clonarlo (revisa configuración de "
            "Git para archivos binarios, o vuelve a subirlo)."
        )
    st.stop()

tab1, tab2, tab3 = st.tabs([
    "Clasificación individual",
    "Clasificación por lotes",
    "Cómo funciona"
])

# ====================================================
# TAB 1: CLASIFICACIÓN INDIVIDUAL
# ====================================================
with tab1:
    st.subheader("Análisis de un texto o meta de proyecto")

    col_input, col_examples = st.columns([2, 1])

    with col_examples:
        st.markdown("**Ejemplos rápidos**")
        ex1 = "Fomentar el acceso a agua potable limpia y construir redes de alcantarillado rural."
        ex2 = "Promover la alfabetización digital y mejorar el equipamiento educativo en escuelas públicas."
        ex3 = "Implementar sistemas de energía solar fotovoltaica para reducir las emisiones de carbono."

        if st.button("Ejemplo: Agua"):
            st.session_state["text_input"] = ex1
        if st.button("Ejemplo: Educación"):
            st.session_state["text_input"] = ex2
        if st.button("Ejemplo: Energía"):
            st.session_state["text_input"] = ex3

    with col_input:
        user_text = st.text_area(
            "Escribe o pega el texto a clasificar:",
            value=st.session_state.get(
                "text_input",
                "Como resultado, un mayor y mejorado acceso al agua limpia y saneamiento "
                "beneficiará a las comunidades rurales."
            ),
            height=130
        )
        analyze_btn = st.button("Analizar texto", type="primary", use_container_width=True)

    if user_text.strip():
        probs, debug_info = predict_ods_debug(user_text)

        sorted_ods = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        top1_ods, top1_prob = sorted_ods[0]
        top2_ods, top2_prob = sorted_ods[1]
        top3_ods, top3_prob = sorted_ods[2]

        info_top1 = ODS_INFO[top1_ods]

        st.divider()

        # A. RESULTADO PRINCIPAL
        st.markdown("### Resultado principal")

        card_html = f"""
        <div style="
            background-color: {info_top1['color']};
            color: #2B2B2B;
            padding: 20px 24px;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        ">
            <h2 style="margin:0; color: #2B2B2B; font-size: 26px; font-weight: 600;">
                ODS {top1_ods}: {info_top1['nombre']}
            </h2>
            <p style="margin-top: 8px; font-size: 16px; opacity: 0.85;">
                Confianza del modelo: <strong>{top1_prob * 100:.2f}%</strong>
            </p>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # B. TOP 3 RELACIONADOS
        st.markdown("### ODS más relacionados")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(label=etiqueta_ods(top1_ods), value=f"{top1_prob*100:.1f}%")
        with c2:
            st.metric(label=etiqueta_ods(top2_ods), value=f"{top2_prob*100:.1f}%")
        with c3:
            st.metric(label=etiqueta_ods(top3_ods), value=f"{top3_prob*100:.1f}%")

        st.divider()

        # C. DISTRIBUCIÓN DE PROBABILIDADES (con nombre completo de cada ODS)
        st.markdown(f"### Distribución de probabilidades para los {NUM_ODS} ODS")

        chart_df = pd.DataFrame([
            {
                "ODS": etiqueta_ods(k),
                "Probabilidad (%)": v * 100,
                "Color": ODS_INFO[k]["color"]
            }
            for k, v in probs.items()
        ])
        chart_df = chart_df.sort_values(by="Probabilidad (%)", ascending=True)

        fig = px.bar(
            chart_df,
            x="Probabilidad (%)",
            y="ODS",
            orientation="h",
            text="Probabilidad (%)"
        )

        fig.update_traces(
            marker_color=chart_df["Color"],
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )
        fig.update_layout(
            height=620,
            xaxis=dict(range=[0, max(chart_df["Probabilidad (%)"]) * 1.15]),
            yaxis_title=None,
            xaxis_title="Probabilidad (%)",
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(size=13),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

        # D. PANEL DE DEPURACIÓN
        with st.expander("Panel de depuración (pipeline de tokenización y probabilidades crudas)"):
            st.markdown(
                "Usa esta información para comparar contra lo que obtienes en Colab con el "
                "mismo texto: si la secuencia tokenizada, el tamaño del vocabulario o las "
                "probabilidades crudas no coinciden, el problema está en los artefactos "
                "cargados (modelo, tokenizer o label encoder), no en el modelo entrenado."
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Longitud de la secuencia (sin padding)", debug_info["longitud_secuencia_sin_padding"])
                st.metric("max_len usado", debug_info["max_len_usado"])
            with col_b:
                st.metric("Tamaño del vocabulario del tokenizer", debug_info["tamano_vocabulario_tokenizer"])
                st.metric("Número de clases del label encoder", len(debug_info["clases_label_encoder"]))

            st.markdown("**Secuencia tokenizada (sin padding)**")
            st.code(str(debug_info["secuencia_tokenizada"]))

            st.markdown("**Secuencia con padding (la que realmente recibe el modelo)**")
            st.code(str(debug_info["secuencia_con_padding"]))

            st.markdown("**Clases del label encoder (orden real de las neuronas de salida)**")
            st.code(str(debug_info["clases_label_encoder"]))

            st.markdown("**Probabilidades crudas del modelo (una por cada neurona de salida, en ese orden)**")
            raw_probs_df = pd.DataFrame({
                "Índice de salida": range(len(debug_info["probabilidades_crudas"])),
                "ODS (según label encoder)": debug_info["clases_label_encoder"],
                "Probabilidad cruda": debug_info["probabilidades_crudas"]
            })
            st.dataframe(raw_probs_df, use_container_width=True, hide_index=True)

# ====================================================
# TAB 2: CLASIFICACIÓN POR LOTES
# ====================================================
with tab2:
    st.subheader("Clasificación masiva desde un archivo")
    st.write("Sube un archivo con varios textos para obtener la predicción de ODS de cada uno.")

    uploaded_file = st.file_uploader("Selecciona un archivo CSV o Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_batch = pd.read_csv(uploaded_file)
            else:
                df_batch = pd.read_excel(uploaded_file)

            st.success(f"Archivo cargado correctamente. Total de filas: {len(df_batch)}")
            st.dataframe(df_batch.head(3), use_container_width=True)

            text_col = st.selectbox("Selecciona la columna que contiene los textos:", df_batch.columns)

            if st.button("Procesar lote completo", type="primary"):
                with st.spinner("Procesando textos con el modelo..."):
                    texts_list = df_batch[text_col].astype(str).tolist()
                    batch_results = predict_ods(texts_list)

                    predicted_ods_num = []
                    predicted_ods_name = []
                    confidence_scores = []

                    for res in batch_results:
                        sorted_res = sorted(res.items(), key=lambda x: x[1], reverse=True)
                        top_ods, top_conf = sorted_res[0]
                        predicted_ods_num.append(top_ods)
                        predicted_ods_name.append(etiqueta_ods(top_ods))
                        confidence_scores.append(round(top_conf * 100, 2))

                    df_result = df_batch.copy()
                    df_result["ODS predicho (número)"] = predicted_ods_num
                    df_result["ODS predicho (nombre)"] = predicted_ods_name
                    df_result["Confianza (%)"] = confidence_scores

                st.subheader("Resultados de la clasificación")
                st.dataframe(df_result, use_container_width=True)

                csv_data = df_result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Descargar resultados en CSV",
                    data=csv_data,
                    file_name="ods_clasificados_resultados.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

# ====================================================
# TAB 3: CÓMO FUNCIONA
# ====================================================
with tab3:
    st.subheader("Cómo funciona la clasificación de texto con LSTM")
    st.markdown(
        f"La aplicación utiliza un modelo de procesamiento de lenguaje natural (NLP) que "
        f"convierte el texto en una predicción sobre los {NUM_ODS} ODS cubiertos por el "
        f"modelo, mediante tres pasos principales."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**1. Tokenización y padding**")
        st.markdown(
            "El texto se divide en palabras y cada una se convierte en un número entero "
            "según un vocabulario predefinido. Por ejemplo, *\"agua limpia\"* se convierte en "
            "una secuencia como `[45, 128]`. Luego, todas las secuencias se ajustan a una "
            "longitud fija de 50 mediante padding."
        )

    with col2:
        st.markdown("**2. Capa de embedding**")
        st.markdown(
            "Cada número se transforma en un vector de 64 dimensiones. A diferencia de una "
            "codificación simple, esta representación aprende relaciones de significado: "
            "palabras como *\"río\"* y *\"agua\"* quedan cerca en el espacio vectorial."
        )

    with col3:
        st.markdown("**3. Red LSTM**")
        st.markdown(
            "La secuencia se procesa palabra por palabra manteniendo un estado de memoria, "
            "lo que permite capturar el contexto y el orden de las palabras. Una capa final "
            f"con función Softmax genera la probabilidad de cada uno de los {NUM_ODS} ODS."
        )

    st.divider()

    st.markdown("### Código del proceso de inferencia")
    st.code("""
# 1. Convertir el texto en números
secuencia = tokenizer.texts_to_sequences([texto_usuario])

# 2. Ajustar la longitud de la secuencia
secuencia_padded = pad_sequences(secuencia, maxlen=50, padding='post')

# 3. Obtener las probabilidades con el modelo
probabilidades = model.predict(secuencia_padded)

# 4. Seleccionar el ODS con mayor probabilidad
ods_predicho = np.argmax(probabilidades)
    """, language="python")

    st.markdown("### Arquitectura del modelo")
    st.markdown(
        "- `Embedding(input_dim=10000, output_dim=64, input_length=50)`\n"
        "- `SpatialDropout1D(0.2)`\n"
        "- `LSTM(units=64, dropout=0.2)`\n"
        "- `Dense(units=16, activation='softmax')`"
    )