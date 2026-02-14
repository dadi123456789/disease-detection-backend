# Backend API - نظام الكشف عن الأمراض الصوتية

## 📋 المحتويات

هذا المشروع يحتوي على Backend API المكتوب بـ Flask للكشف عن الأمراض من الصوت.

### الملفات:
- `app.py` - الكود الرئيسي للـ API
- `requirements.txt` - المكتبات المطلوبة
- `unified_model_phase2.h5` - **يجب إضافته من مشروعك!**
- `scaler.pkl` - **يجب إضافته من مشروعك!**

---

## 🚀 خطوات الرفع على Render

### الخطوة 1: إضافة ملفات النموذج

**⚠️ مهم جداً:**

انسخ هذين الملفين من مشروعك:
```
من: D:\detection_maladies\models\
انسخ:
  - unified_model_phase2.h5
  - scaler.pkl

إلى: هذا المجلد (backend-render)
```

**بدون هذين الملفين، الـ API لن يعمل!**

---

### الخطوة 2: رفع على GitHub

```bash
# في Git Bash أو Command Prompt
cd path/to/backend-render

# تهيئة Git
git init
git add .
git commit -m "Initial commit"

# إنشاء repository على GitHub:
# 1. اذهب إلى https://github.com
# 2. اضغط "New Repository"
# 3. الاسم: disease-detection-backend
# 4. اختر Public
# 5. اضغط "Create Repository"

# ثم:
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/disease-detection-backend.git
git push -u origin main
```

---

### الخطوة 3: الربط مع Render

1. **اذهب إلى**: https://render.com
2. **سجّل دخول** (أو سجّل حساب جديد بـ GitHub)
3. **Dashboard → New +**
4. اختر **"Web Service"**
5. **Connect Repository**: اختر `disease-detection-backend`
6. **الإعدادات:**
   ```
   Name: disease-detection-api
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app
   Instance Type: Free
   ```
7. اضغط **"Create Web Service"**
8. **انتظر 5-10 دقائق** حتى يكتمل النشر

---

### الخطوة 4: احصل على الـ URL

بعد النشر الناجح، ستجد URL مثل:
```
https://disease-detection-api.onrender.com
```

**احفظ هذا الـ URL!** ستحتاجه في تطبيق Android.

---

## 🧪 اختبار الـ API

### 1. اختبار الصحة:
افتح المتصفح:
```
https://YOUR-APP-NAME.onrender.com/health
```

يجب أن ترى:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "scaler_loaded": true
}
```

### 2. اختبار باستخدام curl:
```bash
curl -X POST https://YOUR-APP-NAME.onrender.com/predict \
  -F "audio=@test_audio.wav"
```

---

## 📝 ملاحظات مهمة

### حجم الملفات:
- `unified_model_phase2.h5`: ~50-150 MB
- `scaler.pkl`: ~1-5 MB

### وقت التحميل الأول:
- أول طلب قد يأخذ 30-60 ثانية (Render يوقظ الخادم)
- الطلبات التالية: 3-10 ثواني

### الخطة المجانية:
- ✅ مجانية تماماً
- ⚠️ ينام بعد 15 دقيقة من عدم الاستخدام
- ⚠️ 750 ساعة/شهر (كافي للاستخدام الطبيعي)

---

## 🔧 استكشاف الأخطاء

### خطأ: "Model file not found"
**الحل**: تأكد من رفع `unified_model_phase2.h5` و `scaler.pkl`

### خطأ: "Out of memory"
**الحل**: النموذج كبير جداً للخطة المجانية - حاول استخدام Railway أو Hugging Face

### خطأ: "Build failed"
**الحل**: تأكد من `requirements.txt` صحيح

---

## 📂 هيكل المشروع

```
backend-render/
├── app.py                        # Flask API
├── requirements.txt              # المكتبات
├── unified_model_phase2.h5       # النموذج (أضفه!)
├── scaler.pkl                    # Scaler (أضفه!)
├── .gitignore                    # ملفات Git المتجاهلة
└── README.md                     # هذا الملف
```

---

## ✅ جاهز للنشر!

بعد إضافة ملفات النموذج، المشروع جاهز للرفع على Render.
