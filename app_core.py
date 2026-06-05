import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

client = chromadb.PersistentClient(path="./chroma_db")
koleksiyon = client.get_collection(name="crimson_desert")

def benzer_yorumlari_bul(kullanici_sorusu):
  soru_vektoru = model.encode(kullanici_sorusu)
  sonuclar = koleksiyon.query(query_embeddings=[soru_vektoru], n_results=3)
  bulunan_yorumlar = []
  for yorum_verisi in sonuclar['metadatas'][0]: 
    bulunan_yorumlar.append(yorum_verisi['review'])   
  return bulunan_yorumlar

# Modüler fonksiyonumuzun dışarıda kullanımı:
# soru = input("Lütfen Crimson Desert ile ilgili sorunuzu yazın: ")
# cikan_sonuclar = benzer_yorumlari_bul(soru)