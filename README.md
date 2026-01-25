# Chat with Video: Multimodal Video Search Engine

Videoyu yukleyip dogal dilde soru sorarak en ilgili sahneleri zaman damgasi ile
bulan, gercek bir multimodal RAG sistemidir. Amac, modern VLM ve vektor arama
teknolojilerini tek bir akista birlestiren, uygulamali bir ornek sunmaktir.

## Kullanilan teknolojiler

- VLM: `microsoft/Florence-2-base` ile sahne caption uretimi
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Vektor veritabani: LanceDB
- Arayuz: Gradio
- Dil: Python

## Neler yapiyor?

- Videodan kare ornekleme (varsayilan: 1 FPS)
- Florence-2 ile zengin sahne aciklamalari (VLM)
- Sentence-Transformers embedding + LanceDB ile vektor arama
- En alakali kareyi ve timestamp'i gosterme

## Akis

1. Videodan kare ornekleme
2. Florence-2 ile caption uretimi
3. Caption embedding + LanceDB ile arama
4. Sorgu sonucu en alakali kare ve timestamp

## Kurulum

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Calistirma

```bash
python app.py
```

Gradio arayuzunde once videoyu indeksle, ardindan sorgu sor.

## Kullanim

1. **Indeksleme** sekmesinden videoyu yukle
2. FPS degerini sec (varsayilan 1)
3. **Indeksle** butonuna bas ve islem bitene kadar bekle
4. **Sorgu** sekmesinde dogal dil sorusu sor

Ornek sorular:

- "Kirmizi sapkali adam ne zaman kadraja girdi?"
- "Kavga ne zaman basladi?"
- "Biri yere dustugu ani bul"

## Ipuclari

- Varsayilan VLM: `microsoft/Florence-2-base`
- Varsayilan embedding: `sentence-transformers/all-MiniLM-L6-v2`
- Tum veriler `data/` altinda saklanir (frame ve LanceDB)
- Ilk calistirmada modeller indirilecegi icin yavas olabilir
- Windows'ta HuggingFace cache icin symlink uyarisi alirsan:
  - Gelistirici modunu acabilir veya Python'u yonetici olarak calistirabilirsin
  - Uyariyi kapatmak icin:
    ```powershell
    $env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
    ```

## Neden etkileyici?

- Gercek dunyadan video uzerinde semantik arama
- VLM + vektor arama ile uygulamali RAG ornegi
- Gradio arayuzu ile hizli prototipleme
