# Görüntü İşleme Stüdyosu

Web tabanlı, etkileşimli görüntü işleme ve AI için veri ön işleme uygulaması. Portfolyo projesi olarak geliştirilmiştir.

---

## Özellikler

### Klasik Görüntü İşleme
- **Temel:** Gri tonlama (ITU-R BT.601), ikili eşikleme, döndürme, kırpma, yeniden boyutlandırma
- **Renk:** RGB ↔ HSV dönüşümü
- **İyileştirme:** Histogram eşitleme, parlaklık ayarı
- **Filtreler:** Gauss, ortalama, medyan bulanıklaştırma; tuz-biber gürültüsü ekleme
- **Morfolojik:** Genişletme, aşındırma, açma, kapatma
- **Kenar / segmentasyon:** Uyarlamalı eşikleme, Sobel kenar tespiti
- **İki görsel:** Görüntü toplama ve çarpma

### AI için Veri Ön İşleme
- **Toplu işlem:** Birden fazla görseli aynı ayarlarla ön işleme
- **Hedef model tipleri:** Genel CNN, mobil/hafif, gri tonlama odaklı ön ayarlar
- **Seçenekler:** Gürültü giderme (Gauss, ortalama, medyan), histogram eşitleme, parlaklık, renk modu (RGB / gri / HSV), hedef boyut
- **Kaynak limitleri:** Dosya sayısı ve toplam boyut sınırları (canlı ortam için)
- **Çıktı:** Önizleme galerisi, işlem logu, işlenmiş görsellerin ZIP indirmesi

---

## Teknoloji

- **Python 3**
- **NumPy** — Tüm görüntü işlemleri (OpenCV/PIL yalnızca okuma/yazma için değil, çekirdek işlemler NumPy)
- **Pillow** — Görüntü yükleme/kaydetme (Gradio ile uyum)
- **Gradio** — Web arayüzü

---

## Kurulum ve Yerel Çalıştırma

```bash
cd "görüntü işleme web"
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
python app.py
```

Tarayıcıda açılan adresi (örn. `http://127.0.0.1:7860`) kullanın.

---

## Canlıya Alma (Portfolyo / Web Sitesi)

### Seçenek 1: Hugging Face Spaces (ücretsiz)
1. [Hugging Face](https://huggingface.co) hesabı açın.
2. Yeni bir **Space** oluşturun, **Gradio** SDK seçin.
3. Bu klasördeki `app.py`, `image_processor.py`, `requirements.txt` dosyalarını Space repo’ya yükleyin.
4. Space’in **Settings** → **Space hardware** bölümünden uygun CPU (ve gerekirse RAM) seçin.
5. Space otomatik build alır; bitince “App” sekmesinden canlı linki alırsınız. Bu linki portfolyo veya web sitenize ekleyebilirsiniz.

### Seçenek 2: Web sitenize gömme (iframe)
Canlı Gradio linkini (HF Spaces veya kendi sunucunuz) bir iframe ile sitenize ekleyebilirsiniz:

```html
<iframe
  src="https://your-gradio-app-url"
  width="100%"
  height="600"
  frameborder="0"
  allow="microphone"
  title="Görüntü İşleme Stüdyosu"
></iframe>
```

### Seçenek 3: Hetzner + Coolify (önerilen)
Proje Docker ile çalışacak şekilde hazırdır; Coolify ile Hetzner VPS’te tek tıkla deploy edebilirsiniz.

1. **Hetzner Cloud** üzerinde bir sunucu açın (örn. CX22/CX32, Ubuntu 22.04).
2. Sunucuya **Coolify** kurun ([Coolify dokümantasyonu](https://coolify.io/docs)).
3. Coolify panelinde **New Resource** → **Docker Image** (veya **Git** ile doğrudan repo).
4. **Build:** Dockerfile kullanın; **Build Pack** olarak Dockerfile seçin, context = bu proje klasörü (veya Git repo URL’i).
5. **Port:** Container port **7860** olarak ayarlayın (Gradio bu portu kullanır).
6. **Domain:** İsterseniz bir domain/subdomain bağlayın; Coolify Let’s Encrypt SSL’i otomatik alır.
7. Deploy’a basın; uygulama `https://your-domain` üzerinden yayında olur.

Yerel test için Docker:
```bash
docker build -t goruntu-isleme .
docker run -p 7860:7860 goruntu-isleme
```
Tarayıcıda `http://localhost:7860` açılır.

### Seçenek 4: Kendi sunucunuz (Docker’sız)
- `python app.py` ile çalıştırıp `share=True` ile geçici public link alabilirsiniz.
- Kalıcı yayın için VPS’te process manager (systemd, supervisor) ve ters proxy (nginx) ile 7860 portunu yayınlayabilirsiniz.

---

## Proje Yapısı

```
görüntü işleme web/
├── app.py              # Gradio arayüzü ve pipeline
├── image_processor.py  # Görüntü işleme fonksiyonları (NumPy)
├── requirements.txt
├── Dockerfile          # Coolify / Hetzner deploy
├── .dockerignore
└── README.md
```

---

## Lisans

Eğitim ve portfolyo amaçlı kullanım için geliştirilmiştir.
