import gradio as gr
import numpy as np
import os
from fastapi import FastAPI
from PIL import Image, ImageOps
import tempfile
import shutil
import io
import zipfile
from image_processor import ImageProcessor

"""
Görüntü İşleme Stüdyosu
-----------------------
Bu uygulama, çeşitli görüntü işleme tekniklerini içeren interaktif bir web arayüzü sunar. 
Temel özellikler:
1. Görüntü yükleme ve işleme
2. Görüntü filtreleri ve dönüşümleri uygulama
3. Histogram analizi

Geliştiriciler tarafından eğitim ve araştırma amaçlı kullanım için tasarlanmıştır.
"""

# Görüntü işleme modelini başlat
image_processor = ImageProcessor()

# Gradio arayüzünü oluştur
with gr.Blocks() as demo:
    with gr.Tab("Klasik Görüntü İşleme"):
        # Görüntü işleme sayfası
        editor_page = gr.Column(visible=True)
        with editor_page:
            gr.Markdown("""
            ## Klasik Görüntü İşleme Araçları
            **Tek görüntülü işlemler** (gri dönüşüm, binary, döndürme, filtreler vb.) yalnızca **ilk alana** yüklediğiniz görüntüyü kullanır; ikinci alanı doldurmanıza gerek yoktur.
            **İki görüntülü işlemler** (Toplama, Çarpma) için **hem 1. hem 2. alana** görüntü yüklemeniz gerekir; diğer tüm işlemler için sadece 1. görüntü yeterlidir.
            """)
            # Üstte, yan yana ve sabit yükseklikte giriş/çıkış görselleri
            with gr.Row():
                input_image = gr.Image(label="Görüntü Yükle", type="numpy", height=300)
                input_image2 = gr.Image(label="İkinci Görüntü (Aritmetik)", type="numpy", height=300)
                output_image = gr.Image(label="İşlenmiş Görüntü", height=300)
            # Altında, işlemler ve parametreler 3 sütuna bölünsün
            with gr.Row():
                # 1. Sütun: Temel işlemler
                with gr.Column():
                    gr.Markdown("### Temel İşlemler")
                    grayscale_btn = gr.Button("Gri Dönüşüm")
                    binary_thresh = gr.Slider(0, 255, value=127, label="Binary Eşik Değeri")
                    binary_btn = gr.Button("Binary Dönüşüm")
                    rotate_angle = gr.Slider(-180, 180, value=0, label="Döndürme Açısı")
                    rotate_btn = gr.Button("Döndür")
                    crop_x1 = gr.Number(value=0, label="Kırpma X1")
                    crop_y1 = gr.Number(value=0, label="Kırpma Y1")
                    crop_x2 = gr.Number(value=100, label="Kırpma X2")
                    crop_y2 = gr.Number(value=100, label="Kırpma Y2")
                    crop_btn = gr.Button("Kırp")
                    resize_scale = gr.Slider(0.1, 3.0, value=1.0, label="Yaklaştırma/Uzaklaştırma (Scale)")
                    resize_btn = gr.Button("Yeniden Boyutlandır")
                    rgb2hsv_btn = gr.Button("RGB → HSV Dönüşümü")

                # 2. Sütun: Filtreler, Gürültü, Parlaklık
                with gr.Column():
                    gr.Markdown("### Filtreler & Gürültü & Parlaklık")
                    hist_eq_btn = gr.Button("Histogram Eşitleme")
                    brightness_factor = gr.Slider(-100, 100, value=0, label="Parlaklık Faktörü")
                    brightness_btn = gr.Button("Parlaklık Ayarı")
                    gauss_kernel = gr.Slider(1, 15, value=3, step=2, label="Gaussian Kernel Boyutu")
                    gauss_sigma = gr.Slider(0.1, 5.0, value=1.0, label="Gaussian Sigma")
                    gauss_btn = gr.Button("Gaussian Bulanıklaştırma")
                    mean_kernel = gr.Slider(1, 15, value=3, step=2, label="Ortalama Filtre Kernel Boyutu")
                    mean_btn = gr.Button("Ortalama Filtre")
                    median_kernel = gr.Slider(1, 15, value=3, step=2, label="Medyan Filtre Kernel Boyutu")
                    median_btn = gr.Button("Medyan Filtre")
                    blur_kernel = gr.Slider(1, 15, value=5, step=2, label="Blurring Kernel Boyutu")
                    blur_btn = gr.Button("Blurring (Bulanıklaştırma)")
                    noise_prob = gr.Slider(0.0, 0.2, value=0.02, label="Gürültü Olasılığı (Salt&Pepper)")
                    noise_btn = gr.Button("Gürültü Ekle (Salt&Pepper)")

                # 3. Sütun: Morfolojik, Kenar, Aritmetik
                with gr.Column():
                    gr.Markdown("### Morfolojik, Kenar & Aritmetik İşlemler")
                    morph_kernel = gr.Slider(1, 15, value=3, step=2, label="Morfolojik Kernel Boyutu")
                    dilate_btn = gr.Button("Genişletme (Dilation)")
                    erode_btn = gr.Button("Aşındırma (Erosion)")
                    opening_btn = gr.Button("Açma (Opening)")
                    closing_btn = gr.Button("Kapama (Closing)")
                    adapt_win = gr.Slider(3, 31, value=11, step=2, label="Adaptif Eşikleme Pencere Boyutu")
                    adapt_c = gr.Number(value=2, label="Adaptif Eşikleme C")
                    adapt_btn = gr.Button("Adaptif Eşikleme")
                    sobel_btn = gr.Button("Sobel Kenar Algılama")
                    add_btn = gr.Button("Toplama")
                    multiply_btn = gr.Button("Çarpma")

        # Tek görüntülü işlemler: görüntü yoksa çıktı üretme (hata önleme)
        def with_image(f):
            def wrapped(img, *args, **kwargs):
                if img is None:
                    return None
                return f(img, *args, **kwargs)
            return wrapped

        def crop_safe(img, x1, y1, x2, y2):
            if img is None:
                return None
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            h, w = img.shape[:2]
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                return None
            return image_processor.crop_image(img, x1, y1, x2, y2)

        grayscale_btn.click(with_image(image_processor.to_grayscale), inputs=[input_image], outputs=[output_image])
        binary_btn.click(with_image(lambda img, t: image_processor.binary_conversion(img, threshold=int(t))), inputs=[input_image, binary_thresh], outputs=[output_image])
        rotate_btn.click(with_image(lambda img, a: image_processor.rotate_image(img, a)), inputs=[input_image, rotate_angle], outputs=[output_image])
        crop_btn.click(crop_safe, inputs=[input_image, crop_x1, crop_y1, crop_x2, crop_y2], outputs=[output_image])
        resize_btn.click(with_image(lambda img, s: image_processor.resize_image(img, float(s))), inputs=[input_image, resize_scale], outputs=[output_image])
        rgb2hsv_btn.click(with_image(image_processor.rgb_to_hsv), inputs=[input_image], outputs=[output_image])
        hist_eq_btn.click(with_image(image_processor.histogram_equalization), inputs=[input_image], outputs=[output_image])
        brightness_btn.click(with_image(lambda img, f: image_processor.adjust_brightness(img, int(f))), inputs=[input_image, brightness_factor], outputs=[output_image])
        gauss_btn.click(with_image(lambda img, k, s: image_processor.convolve(img, image_processor.gaussian_kernel(int(k), float(s)))), inputs=[input_image, gauss_kernel, gauss_sigma], outputs=[output_image])
        mean_btn.click(with_image(lambda img, k: image_processor.mean_filter(img, int(k))), inputs=[input_image, mean_kernel], outputs=[output_image])
        median_btn.click(with_image(lambda img, k: image_processor.median_filter(img, int(k))), inputs=[input_image, median_kernel], outputs=[output_image])
        blur_btn.click(with_image(lambda img, k: image_processor.blur(img, int(k))), inputs=[input_image, blur_kernel], outputs=[output_image])
        noise_btn.click(with_image(lambda img, p: image_processor.add_salt_pepper_noise(img, float(p))), inputs=[input_image, noise_prob], outputs=[output_image])
        dilate_btn.click(with_image(lambda img, k: image_processor.dilate(img, int(k))), inputs=[input_image, morph_kernel], outputs=[output_image])
        erode_btn.click(with_image(lambda img, k: image_processor.erode(img, int(k))), inputs=[input_image, morph_kernel], outputs=[output_image])
        opening_btn.click(with_image(lambda img, k: image_processor.opening(img, int(k))), inputs=[input_image, morph_kernel], outputs=[output_image])
        closing_btn.click(with_image(lambda img, k: image_processor.closing(img, int(k))), inputs=[input_image, morph_kernel], outputs=[output_image])
        adapt_btn.click(with_image(lambda img, w, c: image_processor.adaptive_threshold(img, int(w), int(c))), inputs=[input_image, adapt_win, adapt_c], outputs=[output_image])
        sobel_btn.click(with_image(image_processor.sobel_edge_detection), inputs=[input_image], outputs=[output_image])

        def add_images_safe(img1, img2):
            # İki görüntü gerektiren işlem: biri eksikse çıktı üretme
            if img1 is None or img2 is None:
                return None
            return image_processor.add_images(img1, img2)

        def multiply_images_safe(img1, img2):
            # İki görüntü gerektiren işlem: biri eksikse çıktı üretme
            if img1 is None or img2 is None:
                return None
            return image_processor.multiply_images(img1, img2)

        add_btn.click(add_images_safe, inputs=[input_image, input_image2], outputs=[output_image])
        multiply_btn.click(multiply_images_safe, inputs=[input_image, input_image2], outputs=[output_image])

    with gr.Tab("AI için Veri Ön İşleme"):
        gr.Markdown("""
        ## Model Eğitimi Öncesi Veri Hazırlama (Preprocessing Pipeline)
        Bu sayfa, derin öğrenme modellerine gidecek görüntüleri hazırlamak için esnek bir ön işleme hattı sunar.
        Denoising, kontrast/renk ayarı ve yeniden boyutlandırma gibi adımları seçip tüm veri setinize uygulayabilir,
        çıktıyı ZIP dosyası olarak indirebilirsiniz.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                dataset_files = gr.Files(
                    label="Görüntü Dosyaları (En fazla 50 adet)",
                    file_count="multiple",
                    type="filepath"
                )
                model_type = gr.Dropdown(
                    choices=["ImageNet-tipi RGB CNN", "Gri tonlamalı CNN", "Segmentasyon / Maske"],
                    value="ImageNet-tipi RGB CNN",
                    label="Hedef Model Tipi"
                )
                gr.Markdown("**Denoising**")
                denoise_type = gr.Radio(
                    choices=["none", "gaussian", "mean", "median"],
                    value="none",
                    label="Gürültü Giderme Türü"
                )
                denoise_kernel = gr.Slider(1, 15, value=3, step=2, label="Kernel Boyutu")
                denoise_sigma = gr.Slider(0.1, 5.0, value=1.0, label="Gaussian Sigma")
                
                gr.Markdown("**Kontrast & Parlaklık**")
                use_hist_eq = gr.Checkbox(label="Histogram Eşitleme Uygula", value=False)
                brightness_ai = gr.Slider(-50, 50, value=0, label="Parlaklık Ofseti")
                
                gr.Markdown("**Renk & Boyut**")
                color_mode = gr.Dropdown(
                    choices=["rgb", "grayscale", "hsv_h"],
                    value="rgb",
                    label="Renk Modu"
                )
                target_size = gr.Slider(64, 512, value=224, step=32, label="Hedef Görüntü Boyutu (kare)")
                preprocess_btn = gr.Button("Veri Setini Ön İşle", elem_id="ai-preprocess-btn")
            
            with gr.Column(scale=1):
                preview_gallery = gr.Gallery(
                    label="Örnek Ön İşlenmiş Görüntüler",
                    show_label=True,
                    height=400
                )
                preprocess_log = gr.Textbox(
                    label="İşlem Özeti",
                    lines=10,
                    interactive=False
                )
                download_zip = gr.File(
                    label="İşlenmiş Veri Seti (ZIP olarak indir)",
                    interactive=False
                )

        MAX_FILES = 50
        MAX_FILE_MB = 5
        MAX_TOTAL_MB = 200

        def preset_from_model_type(model_type_value):
            """
            Hedef model tipine göre önerilen ön işleme ayarlarını döndürür.
            Sadece UI bileşenlerinin varsayılan değerlerini günceller.
            """
            mt = (model_type_value or "").lower()
            if "grayscale" in mt or "gri" in mt:
                # Gri tonlamalı CNN için: median filtresi, histogram eşitleme, grayscale, orta boy
                return (
                    gr.update(value="median"),
                    gr.update(value=3),
                    gr.update(value=1.0),
                    gr.update(value=True),
                    gr.update(value=0),
                    gr.update(value="grayscale"),
                    gr.update(value=128),
                )
            if "segmentasyon" in mt or "maske" in mt:
                # Segmentasyon için: hafif gaussian, histogram eşitleme açık, renk modu genelde grayscale
                return (
                    gr.update(value="gaussian"),
                    gr.update(value=5),
                    gr.update(value=1.0),
                    gr.update(value=True),
                    gr.update(value=0),
                    gr.update(value="grayscale"),
                    gr.update(value=256),
                )
            # Varsayılan: ImageNet-tipi RGB CNN
            return (
                gr.update(value="gaussian"),
                gr.update(value=3),
                gr.update(value=1.0),
                gr.update(value=False),
                gr.update(value=0),
                gr.update(value="rgb"),
                gr.update(value=224),
            )

        def limit_files(files):
            """
            Dosyalar yüklendiği anda sayıyı kontrol eder.
            50 sınırı aşılıyorsa seçim temizlenir ve kullanıcıya uyarı verilir.
            """
            if not files:
                return files, ""
            count = len(files)
            if count > MAX_FILES:
                return None, f"En fazla {MAX_FILES} dosya seçebilirsiniz. Seçilen dosya sayısı: {count}"
            return files, ""

        def ai_preprocess_pipeline(
            files,
            model_type,
            denoise_type,
            denoise_kernel,
            denoise_sigma,
            use_hist_eq,
            brightness_ai,
            color_mode,
            target_size
        ):
            if not files:
                return [], "Lütfen en az bir görüntü dosyası yükleyin.", None
            
            original_count = len(files)
            if original_count > MAX_FILES:
                # Sınırı aşma durumunda işlem yapma, sadece uyarı ver
                return [], f"En fazla {MAX_FILES} dosya işlenebilir. Yüklenen dosya sayısı: {original_count}", None
            
            total_bytes = 0
            for f in files:
                path = f.name if hasattr(f, "name") else f
                size = os.path.getsize(path)
                if size > MAX_FILE_MB * 1024 * 1024:
                    return [], f"'{os.path.basename(path)}' dosyası {MAX_FILE_MB} MB sınırını aşıyor."
                total_bytes += size
                if total_bytes > MAX_TOTAL_MB * 1024 * 1024:
                    return [], f"Toplam dosya boyutu {MAX_TOTAL_MB} MB sınırını aşıyor."
            
            processed_images = []
            
            for f in files:
                try:
                    path = f.name if hasattr(f, "name") else f
                    img_pil = Image.open(path).convert("RGB")
                    img_np = np.array(img_pil)
                    
                    processed = image_processor.preprocess_for_model(
                        img_np,
                        denoise_type=denoise_type,
                        kernel_size=int(denoise_kernel),
                        sigma=float(denoise_sigma),
                        use_hist_eq=use_hist_eq,
                        brightness=int(brightness_ai),
                        color_mode=color_mode,
                        target_size=int(target_size),
                    )
                    
                    if processed is None:
                        continue
                    
                    processed_images.append(processed)
                except Exception as e:
                    # Tekil dosya hatası, logta belirtelim ama diğerlerine devam edelim
                    continue
            
            if not processed_images:
                return [], "Hiçbir dosya işlenemedi. Lütfen dosyaları kontrol edin.", None

            # İşlenmiş görüntüleri ZIP olarak bellekte oluştur
            tmp_zip_path = None
            try:
                tmp_fd, tmp_zip_path = tempfile.mkstemp(suffix=".zip")
                os.close(tmp_fd)
                with zipfile.ZipFile(tmp_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for idx, proc_img in enumerate(processed_images):
                        img_pil = Image.fromarray(proc_img)
                        buf = io.BytesIO()
                        img_pil.save(buf, format="PNG")
                        buf.seek(0)
                        zf.writestr(f"image_{idx+1:04d}.png", buf.read())
            except Exception:
                tmp_zip_path = None
            
            info_lines = [
                f"Toplam işlenen dosya: {len(processed_images)}",
            ]
            info_lines.extend([
                f"Hedef model tipi: {model_type}",
                f"Gürültü giderme: {denoise_type} (kernel={int(denoise_kernel)}, sigma={float(denoise_sigma):.2f})" if denoise_type != "none" else "Gürültü giderme: yok",
                f"Histogram eşitleme: {'açık' if use_hist_eq else 'kapalı'}",
                f"Parlaklık ofseti: {int(brightness_ai)}",
                f"Renk modu: {color_mode}",
                f"Hedef boyut: {int(target_size)}x{int(target_size)}",
            ])
            if tmp_zip_path is not None:
                info_lines.append(f"ZIP dosyası hazır: {os.path.basename(tmp_zip_path)}")
            
            return processed_images, "\n".join(info_lines), tmp_zip_path

        model_type.change(
            preset_from_model_type,
            inputs=[model_type],
            outputs=[
                denoise_type,
                denoise_kernel,
                denoise_sigma,
                use_hist_eq,
                brightness_ai,
                color_mode,
                target_size,
            ],
        )

        dataset_files.change(
            limit_files,
            inputs=[dataset_files],
            outputs=[dataset_files, preprocess_log],
        )

        preprocess_btn.click(
            ai_preprocess_pipeline,
            inputs=[
                dataset_files,
                model_type,
                denoise_type,
                denoise_kernel,
                denoise_sigma,
                use_hist_eq,
                brightness_ai,
                color_mode,
                target_size,
            ],
            outputs=[preview_gallery, preprocess_log, download_zip],
        )

    # CSS Düzeltmesi - Buton stilleri
    gr.Markdown("""
    <style>
        .gr-button.lg {
            margin: 3px 0 !important;
            border-radius: 6px !important;
            min-height: 45px !important;
            background: #F6F4EB !important;
            border: 1px solid #2D3250 !important;
            color: #2D3250 !important;
            transition: all 0.2s ease !important;
            font-weight: 500 !important;
        }
        .gr-button.lg:hover {
            background: #2D3250 !important;
            color: white !important;
            transform: translateY(-1px) !important;
        }
    </style>
    """)

# Coolify/reverse proxy için: FastAPI + mount_gradio_app + uvicorn
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    print(f"Starting on http://0.0.0.0:{port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=port) 
