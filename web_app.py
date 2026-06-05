import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
import json
import time
import os
from dotenv import load_dotenv

# .env dosyasındaki değişkenleri ortam değişkenlerine yükler
load_dotenv()

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Crimson Desert - QA Asistanı", page_icon="⚔️", layout="wide")

# --- 1. AYARLAR VE API YÖNETİMİ ---
# API anahtarları artık .env dosyasından okunuyor (güvenlik için)
API_ANAHTARLARI = [
    os.getenv("GEMINI_API_KEY_1"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
]

if "aktif_anahtar_indeksi" not in st.session_state:
    st.session_state.aktif_anahtar_indeksi = 0

@st.cache_resource
def modelleri_kur(indeks):
    genai.configure(api_key=API_ANAHTARLARI[indeks])
    trafik_polisi = genai.GenerativeModel('gemini-2.5-flash', generation_config={"response_mime_type": "application/json"})
    cevaplayici = genai.GenerativeModel('gemini-2.5-flash')
    return trafik_polisi, cevaplayici

trafik_polisi_model, cevaplayici_model = modelleri_kur(st.session_state.aktif_anahtar_indeksi)

def guvenli_istek_yap(model, prompt, json_mi=False):
    max_deneme = len(API_ANAHTARLARI) * 2
    for deneme in range(max_deneme):
        try:
            cevap = model.generate_content(prompt)
            return json.loads(cevap.text) if json_mi else cevap.text
        except Exception as e:
            hata_mesaji = str(e).lower()
            if "429" in hata_mesaji or "quota" in hata_mesaji or "exhausted" in hata_mesaji:
                st.session_state.aktif_anahtar_indeksi = (st.session_state.aktif_anahtar_indeksi + 1) % len(API_ANAHTARLARI)
                global trafik_polisi_model, cevaplayici_model
                trafik_polisi_model, cevaplayici_model = modelleri_kur(st.session_state.aktif_anahtar_indeksi)
                model = trafik_polisi_model if json_mi else cevaplayici_model
                time.sleep(3)
            else:
                time.sleep(5)
    return {} if json_mi else "Üzgünüm, API limitleri aşıldı ve yanıt üretilemedi."

# --- 2. VERİTABANI BAĞLANTISI (Cache Kaldırıldı) ---

# Veritabanını her seferinde güvenli bir şekilde başlatan fonksiyon
def veritabanini_getir():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    client = chromadb.PersistentClient(path="./oyun_rag_veritabani")
    return client.get_collection(name="crimson_desert_hatalar", embedding_function=ef)

# Streamlit'in session_state (oturum) hafızasını kullanarak güvenli bağlantı yapıyoruz
if "koleksiyon" not in st.session_state:
    st.session_state.koleksiyon = veritabanini_getir()

koleksiyon = st.session_state.koleksiyon

# --- 3. UYGULAMA MANTIĞI ---
def sorguyu_analiz_et(kullanici_sorusu):
    prompt = f"""
    Sen bir veritabanı yönlendiricisisin. Görevin kullanıcının sorusunu analiz edip bir ChromaDB arama filtresi oluşturmak.
    Kullanıcı bir süreden bahsediyorsa 'oynama_suresi' (dakika) filtresini kullan. (1 saat = 60 dakika).
    Kullanıcı iade etmekten bahsediyorsa 'iade_edildi' (true) filtresini kullan.
    ÖNEMLİ CHROMADB KURALI: Eğer birden fazla filtre kullanman gerekiyorsa, KESİNLİKLE bunları "$and" operatörü içine bir liste olarak koymalısın.
    Aşağıdaki JSON formatında yanıt ver:
    {{
        "arama_metni": "Vektör araması yapılacak temel kavramlar",
        "filtreler": {{}}
    }}
    Kullanıcı Sorusu: "{kullanici_sorusu}"
    """
    return guvenli_istek_yap(trafik_polisi_model, prompt, json_mi=True)

# --- 4. ARAYÜZ TASARIMI ---
st.title("⚔️ Crimson Desert - QA & İçgörü Asistanı")
st.markdown("Geliştirici ekibe yardımcı olmak için tasarlanmış, doğrudan oyuncu geri bildirimlerinden beslenen yapay zeka asistanı.")

# Sohbet geçmişini sakla
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = []

# Önceki mesajları ekranda göster
for mesaj in st.session_state.mesajlar:
    with st.chat_message(mesaj["rol"]):
        st.markdown(mesaj["icerik"])

# Kullanıcı yeni bir girdi yaptığında
if prompt := st.chat_input("Yönetici Sorusu (Örn: İade edenler zırhlardan memnun mu?)"):
    # Kullanıcı mesajını ekle ve göster
    st.session_state.mesajlar.append({"rol": "user", "icerik": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Asistanın yanıtını oluştur
    with st.chat_message("assistant"):
        with st.status("Veritabanı analiz ediliyor...", expanded=True) as status:
            # 1. Trafik Polisi
            st.write("🕵️ Trafik Polisi sorguyu filtreliyor...")
            analiz_sonucu = sorguyu_analiz_et(prompt)
            arama_metni = analiz_sonucu.get("arama_metni", "")
            filtreler = analiz_sonucu.get("filtreler", {})
            st.json({"Arama Terimi": arama_metni, "Filtreler": filtreler})

            # 2. Vektör Araması
            st.write("🔍 ChromaDB'de belgeler aranıyor...")
            if filtreler:
                sonuclar = koleksiyon.query(query_texts=[arama_metni], n_results=5, where=filtreler)
            else:
                sonuclar = koleksiyon.query(query_texts=[arama_metni], n_results=5)

            # Ebeveyn-Çocuk Araması (Kümeleri Bulma)
            bulunan_kume_idleri = set()
            for meta in sonuclar['metadatas'][0]:
                k_id = meta.get('kume_id', -1)
                belge_tipi = meta.get('belge_tipi', '')
                if belge_tipi == 'tekil_yorum' and k_id != -1:
                    bulunan_kume_idleri.add(k_id)
            
            ekstra_ozet_metinleri = []
            for kume_id in bulunan_kume_idleri:
                ozet_sorgusu = koleksiyon.get(where={"$and": [{"belge_tipi": "kume_ozeti"}, {"kume_id": kume_id}]})
                if ozet_sorgusu['documents']:
                    ekstra_ozet_metinleri.append(ozet_sorgusu['documents'][0])

            # Metinleri Birleştir
            bulunan_metinler_listesi = []
            if ekstra_ozet_metinleri:
                for ozet in ekstra_ozet_metinleri:
                    bulunan_metinler_listesi.append(f"[KÜME GENEL ÖZETİ]: {ozet}\n")
            for i in range(len(sonuclar['documents'][0])):
                metin = sonuclar['documents'][0][i]
                metadata = sonuclar['metadatas'][0][i]
                bulunan_metinler_listesi.append(f"[Metadata: {metadata}]\nYorum: {metin}\n")
            
            bulunan_metinler_birlestirilmis = "\n".join(bulunan_metinler_listesi)
            
            # Geliştirici için şeffaflık kutusu
            with st.expander("📚 LLM'in Okuduğu Ham Kaynakları Gör"):
                st.text(bulunan_metinler_birlestirilmis)

            # 3. Final Cevap Üretimi
            st.write("✍️ Asistan raporu yazıyor...")
            final_prompt = f"""
            Sen geliştirici ekibe içeriden yardım eden bir teknik destek asistanısın.
            Amacın, veritabanından çekilen aşağıdaki oyuncu geri bildirimlerini okuyup, ekibin neleri düzeltmesi gerektiğini özetlemektir.
            Eğer bağlamda [KÜME GENEL ÖZETİ] varsa, önce o büyük resmi belirterek konuya gir.
            
            Kurallar:
            1. Çok uzun destanlar yazma. Doğrudan sorunları söyle.
            2. Aşırı teknik jargona boğma, sade ve net ol.
            3. 'Değerli oyuncularımız' gibi müşteri hizmetleri ağzı veya resmi formatlar KULLANMA.
            4. Sadece kısa bir giriş yap, sorunları maddeler halinde özetle ve bitir.
            
            İncelemen Gereken Veritabanı Kayıtları:
            {bulunan_metinler_birlestirilmis}
            
            Ekibin Sorusu: {prompt}
            """
            
            final_cevap = guvenli_istek_yap(cevaplayici_model, final_prompt)
            status.update(label="Analiz Tamamlandı!", state="complete", expanded=False)

        # Sonucu ekrana bas ve geçmişe kaydet
        st.markdown(final_cevap)
        st.session_state.mesajlar.append({"rol": "assistant", "icerik": final_cevap})