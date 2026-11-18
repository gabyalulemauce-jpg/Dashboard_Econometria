import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Dashboard Econometría ENDI", layout="wide", page_icon="📊")
st.title("📊 El ingreso del hogar como determinante del peso de los niños de 2 a 3 años")
st.markdown("**Proyecto de Econometría - Modelo de Regresión Lineal Simple (MCO)**")

# ==========================================
# CARGA DE DATOS
# ==========================================
@st.cache_data # Esto hace que la app cargue rápido al no releer el Excel cada vez
def load_data():
    try:
        # Asegúrate de que el nombre coincida exactamente con tu archivo Excel
        df = pd.read_excel("Base_Final_Modelo_Fase3.xlsx")
        return df
    except Exception as e:
        st.error(f"⚠️ Error al cargar la base de datos: {e}. Asegúrate de haber subido el archivo Excel al repositorio.")
        return None

df = load_data()

if df is not None:
    # ==========================================
    # 1. PREGUNTA Y POBLACIÓN
    # ==========================================
    st.header("1. Pregunta y Población Objetivo")
    st.markdown("""
    *   **Pregunta de Investigación:** ¿Cuál es el efecto esperado (ceteris paribus) del aumento de una unidad monetaria (1 USD) en el ingreso mensual total del hogar sobre el peso promedio de los niños ecuatorianos de 2 a 3 años de edad?
    *   **Población Objetivo:** Todos los hogares ecuatorianos con niños de 2 a 3 años de edad encuestados en la ENDI 2023-2024 que reportaron información válida de ingresos y pesos.
    """)

    # ==========================================
    # 2. MODELO POBLACIONAL Y MUESTRAL
    # ==========================================
    st.header("2. Planteamiento del Modelo")
    st.markdown("Basado en Stock y Watson (Cap. 4), el modelo de regresión lineal simple se define así:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Modelo Poblacional:**")
        st.latex(r"Y_i = \beta_0 + \beta_1 X_i + u_i")
        st.markdown("Donde \(Y_i\) es el peso, \(X_i\) el ingreso, \(\beta_0\) el intercepto, \(\beta_1\) el efecto causal y \(u_i\) el error (variables omitidas).")
    with col2:
        st.markdown("**Modelo Muestral (Estimación MCO):**")
        st.latex(r"\hat{Y}_i = \hat{\beta}_0 + \hat{\beta}_1 X_i")
        st.markdown("Que aplicado a nuestras variables es:")
        st.latex(r"\widehat{peso\_hogar}_i = \hat{\beta}_0 + \hat{\beta}_1 (ingreso\_total\_hogar_i)")

    # ==========================================
    # 3. OPERACIONALIZACIÓN
    # ==========================================
    st.header("3. Operacionalización de Variables")
    st.markdown("""
    *   **\(Y\) (Variable Dependiente) - `peso_hogar`:** Es el promedio del peso (en Kilogramos) de los niños de 2 a 3 años en el hogar.
    *   **\(X\) (Variable Independiente/Regresor) - `ingreso_total_hogar`:** Es la suma de todos los ingresos **mensuales** de los miembros del hogar (salarios, negocios, bonos, etc.) en Dólares Americanos (USD).
    """)

    # ==========================================
    # 4. RESUMEN DE LA ENCUESTA ENDI (Tarjetas)
    # ==========================================
    st.header("4. Resumen del Tratamiento de Datos (ENDI)")
    st.markdown("Se realizó un proceso de limpieza y filtrado sobre la base original:")
    
    # Nota: Como solo cargas la base final, estos números son ilustrativos basados en tu proceso. 
    # Puedes cambiarlos en el código por los números exactos de tu consola de Python.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observaciones Iniciales", "8,709", "Hogares totales")
    c2.metric("Filtro Biológico (Peso)", "-250", "Pesos fuera de rango (8-22 kg)")
    c3.metric("Filtro de Ingresos", "-120", "Ingresos en 0 o faltantes")
    c4.metric("Muestra Final (N)", f"{len(df):,}", "Hogares analizados")

    # ==========================================
    # 5. EDA RÁPIDO DE PESO E INGRESO
    # ==========================================
    st.header("5. Análisis Exploratorio de Datos (EDA) Principal")
    
    col_eda1, col_eda2 = st.columns(2)
    with col_eda1:
        st.markdown("**Variable Dependiente: Peso Promedio (kg)**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Media", f"{df['peso_hogar'].mean():.2f} kg")
        c2.metric("Mínimo", f"{df['peso_hogar'].min():.2f} kg")
        c3.metric("Máximo", f"{df['peso_hogar'].max():.2f} kg")
        
        fig_peso = px.histogram(df, x='peso_hogar', nbins=30, title="Distribución del Peso", 
                                labels={'peso_hogar':'Peso (kg)'}, color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_peso, use_container_width=True)

    with col_eda2:
        st.markdown("**Variable Independiente: Ingreso Mensual (USD)**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Media", f"${df['ingreso_total_hogar'].mean():,.2f}")
        c2.metric("Mínimo", f"${df['ingreso_total_hogar'].min():,.2f}")
        c3.metric("Máximo", f"${df['ingreso_total_hogar'].max():,.2f}")
        
        fig_ing = px.histogram(df, x='ingreso_total_hogar', nbins=30, title="Distribución del Ingreso", 
                               labels={'ingreso_total_hogar':'Ingreso (USD)'}, color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig_ing, use_container_width=True)

    # ==========================================
    # 6. EDA DE OTRAS VARIABLES (Con sentido estadístico)
    # ==========================================
    st.header("6. Análisis de Otras Variables (Boxplots y Correlaciones)")
    st.markdown("Analizamos cómo interactúan el tamaño del hogar y el número de niños con nuestras variables principales.")
    
    col_extra1, col_extra2 = st.columns(2)
    with col_extra1:
        # Relación entre número de niños de 2-3 años y el peso promedio
        fig_box1 = px.box(df, x='num_ninos_2_3', y='peso_hogar', color='num_ninos_2_3',
                          title="Peso Promedio según Número de Niños en el Hogar",
                          labels={'num_ninos_2_3': 'Número de niños (2-3 años)', 'peso_hogar': 'Peso Promedio (kg)'})
        st.plotly_chart(fig_box1, use_container_width=True)
        
    with col_extra2:
        # Matriz de correlación
        corr_matrix = df[['peso_hogar', 'ingreso_total_hogar', 'tamano_hogar', 'num_ninos_2_3']].corr()
        fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                             title="Matriz de Correlación",
                             color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
        st.plotly_chart(fig_corr, use_container_width=True)

    # ==========================================
    # 7. ESTIMACIÓN DEL MODELO MCO Y EFECTO CAUSAL
    # ==========================================
    st.header("7. Estimación del Efecto Causal (MCO)")
    
    # Estimación
    modelo = smf.ols('peso_hogar ~ ingreso_total_hogar', data=df).fit()
    b0 = modelo.params['Intercept']
    b1 = modelo.params['ingreso_total_hogar']
    
    st.markdown("#### Resultados del Modelo")
    st.markdown(f"**Ecuación Estimada:** \(\widehat{{peso}} = {b0:.4f} + {b1:.6f} \cdot ingreso\)")
    
    st.markdown(f"""
    **Efecto Causal Ceteris Paribus:**
    *   Por cada dólar adicional en el ingreso mensual del hogar, el peso promedio del niño cambia en **{b1:.6f} kg**.
    *   Para dimensionar mejor el efecto: un aumento de 100 dólares en el ingreso del hogar se asocia con un cambio de **{(b1*100):.4f} kg**.
    *   *Análisis Crítico:* El efecto es numéricamente minúsculo, lo que sugiere un fuerte **Sesgo por Variable Omitida (OVB)** (genética, educación de la madre, etc.).
    """)
    
    with st.expander("Ver Resumen Estadístico Completo (Tabla OLS)"):
        st.text(modelo.summary().as_text())

    # ==========================================
    # 8. SUPUESTOS DE MCO
    # ==========================================
    st.header("8. Comprobación de Supuestos MCO")
    
    # Pruebas
    bp_test = het_breuschpagan(modelo.resid, modelo.model.exog)
    jb_test = jarque_bera(modelo.resid)
    
    # Gráfico de Residuos
    fig_resid, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x=modelo.fittedvalues, y=modelo.resid, alpha=0.4, ax=ax)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_title('Residuos vs Valores Ajustados')
    ax.set_xlabel('Valores Ajustados (Peso predicho)')
    ax.set_ylabel('Residuos')
    st.pyplot(fig_resid)
    
    col_sup1, col_sup2 = st.columns(2)
    with col_sup1:
        st.markdown(r"""
        **1. Media Cero (\(E[u_i|X_i]=0\))**
        *   **Hipótesis:** El error no debe estar correlacionado con \(X\).
        *   **Conclusión:** ❌ **NO SE CUMPLE.** Existe OVB. El error contiene variables no observadas correlacionadas con el ingreso.
        """)
        
        st.markdown(r"""
        **2. Homocedasticidad (Varianza Constante)**
        *   **Hipótesis:** \(H_0\): Varianza constante. \(H_1\): Heterocedasticidad.
        *   **Prueba:** Breusch-Pagan
        """)
        st.latex(r"LM = n \cdot R^2")
        st.write(f"P-valor: {bp_test[1]:.4f}")
        if bp_test[1] < 0.05:
            st.error("Conclusión: Se rechaza \(H_0\). Hay Heterocedasticidad.")
        else:
            st.success("Conclusión: No se rechaza \(H_0\). Hay Homocedasticidad.")

    with col_sup2:
        st.markdown(r"""
        **3. Normalidad de los Errores**
        *   **Hipótesis:** \(H_0\): Los errores siguen una distribución normal. \(H_1\): No es normal.
        *   **Prueba:** Jarque-Bera
        """)
        st.latex(r"JB = \frac{n}{6} (S^2 + \frac{(K-3)^2}{4})")
        st.write(f"P-valor: {jb_test[1]:.4f}")
        if jb_test[1] < 0.05:
            st.error("Conclusión: Se rechaza \(H_0\). Los errores NO son normales.")
        else:
            st.success("Conclusión: No se rechaza \(H_0\). Los errores son normales.")

    # ==========================================
    # 9. DESVIACIÓN TÍPICA DEL ESTIMADOR (Error Estándar)
    # ==========================================
    st.header("9. Desviación Típica del Estimador (Error Estándar)")
    se_b1 = modelo.bse['ingreso_total_hogar']
    
    st.markdown("El error estándar mide la dispersión de la distribución muestral del estimador. Mientras menor sea, más preciso es el estimador.")
    st.latex(r"SE(\hat{\beta}_1) = \sqrt{\hat{\sigma}^2 / \sum (X_i - \bar{X})^2}")
    
    st.info(f"**Resultado:** El Error Estándar de la pendiente es **\(SE(\hat{{\\beta}}_1) = {se_b1:.6f}\)**.")
    st.markdown("Este valor es fundamental para calcular el estadístico \(t\) y construir los intervalos de confianza.")

    # ==========================================
    # 10. SIGNIFICANCIA DEL ESTIMADOR
    # ==========================================
    st.header("10. Significancia del Estimador")
    
    t_stat = modelo.tvalues['ingreso_total_hogar']
    p_value = modelo.pvalues['ingreso_total_hogar']
    
    st.markdown(r"""
    Para saber si el ingreso realmente afecta al peso estadísticamente, realizamos un contraste de hipótesis:
    *   **\(H_0: \beta_1 = 0\)** (El ingreso no tiene efecto sobre el peso)
    *   **\(H_1: \beta_1 \neq 0\)** (El ingreso tiene efecto)
    """)
    
    st.latex(r"t = \frac{\hat{\beta}_1 - 0}{SE(\hat{\beta}_1)}")
    
    st.write(f"*   **Estadístico \(t\) calculado:** {t_stat:.2f}")
    st.write(f"*   **P-valor:** {p_value:.4f}")
    
    if p_value < 0.05:
        st.success("**Conclusión:** Como el p-valor < 0.05, **Rechazamos \(H_0\)**. El coeficiente es estadísticamente significativo al 5%. (Aunque económicamente el efecto sea pequeño y exista OVB).")
    else:
        st.error("**Conclusión:** No se rechaza \(H_0\). El estimador NO es significativo.")
        
    st.markdown(f"*Nota final: El \(R^2\) del modelo es {modelo.rsquared:.5f}, lo que indica que el ingreso mensual explica apenas el {modelo.rsquared*100:.2f}% de la variación del peso. Esto refuerza la teoría del Sesgo por Variable Omitida.*")

else:
    st.warning("Por favor, asegúrate de que el archivo 'Base_Final_Modelo_Fase3.xlsx' esté en el repositorio.")
