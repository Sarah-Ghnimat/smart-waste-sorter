"""Arabic smart waste sorter - Streamlit app (local inference, browser Arabic speech)."""
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

import waste_core as wc

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "waste_model.pt"
CUSTOM_DIR = ROOT / "data" / "custom"
cfg = wc.load_config(ROOT / "classes.json")
S = cfg["settings"]

st.set_page_config(page_title="فرز النفايات الذكي", page_icon="♻️", layout="centered")
st.markdown("""
<style>
html, body, .stApp, [data-testid="stSidebar"], [data-testid="stMarkdownContainer"] {
  direction: rtl; text-align: right; font-family: "Segoe UI", Tahoma, Arial, sans-serif; }
.hero {background: linear-gradient(135deg,#2e7d32,#66bb6a); color:#fff; padding:1.2rem 1.5rem;
  border-radius:16px; margin-bottom:1rem;}
.hero h1 {margin:0; font-size:2rem; color:#fff;} .hero p {margin:.3rem 0 0; opacity:.95;}
.card {background:#fff; border-radius:14px; padding:1.1rem 1.3rem; margin:.8rem 0;
  box-shadow:0 2px 10px rgba(0,0,0,.12);}
.card .item {font-size:1.7rem; font-weight:700;} .card .bin {font-size:1.25rem; margin:.3rem 0;}
.card .msg {font-size:1.1rem; line-height:1.9;} .card .conf {color:#555; font-size:.95rem; margin-top:.5rem;}
.disclaimer {background:#fff8e1; border:1px solid #ffe082; color:#5d4037; padding:.7rem 1rem;
  border-radius:10px; font-size:.95rem; margin-top:1rem;}
</style>""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>♻️ فرز النفايات الذكي</h1>'
            '<p>صوّر العنصر أو ارفع صورة، وسأقترح الحاوية المناسبة وأقرأ النتيجة بصوت عالٍ</p></div>',
            unsafe_allow_html=True)
DISCLAIMER = ("⚠️ هذا التطبيق يساعد على الفرز فقط، ولا يضمن أن العنصر مقبول في نظام إعادة التدوير المحلي. "
              "تحقق دائمًا من تعليمات المدرسة أو اسأل المشرف.")


def speech_widget(text, auto):
    t = json.dumps(text, ensure_ascii=False).replace("</", "<\\/")
    html = """
<div dir="rtl" style="font-family:Tahoma,Arial,sans-serif">
 <button id="s" style="font-size:16px;padding:8px 16px;border:0;border-radius:8px;background:#2e7d32;color:#fff;cursor:pointer">🔊 اقرأ النتيجة</button>
 <button id="r" style="font-size:16px;padding:8px 16px;border:0;border-radius:8px;background:#558b2f;color:#fff;cursor:pointer">🔁 أعد القراءة</button>
 <div id="st" style="margin-top:6px;color:#8d4e00;font-size:14px"></div>
</div>
<script>
const TEXT = __TEXT__; const AUTO = __AUTO__;
const st = document.getElementById('st');
function speak() {
  if (!('speechSynthesis' in window)) { st.textContent = 'هذا المتصفح لا يدعم النطق.'; return; }
  speechSynthesis.cancel();
  const voices = speechSynthesis.getVoices();
  const v = voices.find(x => x.lang.toLowerCase().startsWith('ar')) || null;
  const u = new SpeechSynthesisUtterance(TEXT);
  u.lang = v ? v.lang : 'ar-SA'; if (v) u.voice = v; u.rate = 0.9;
  st.textContent = v ? '' : 'لم أجد صوتًا عربيًا في هذا المتصفح/الجهاز. جرّب Microsoft Edge أو ثبّت الصوت العربي في ويندوز.';
  speechSynthesis.speak(u);
}
document.getElementById('s').onclick = speak;
document.getElementById('r').onclick = speak;
if (AUTO) setTimeout(speak, 700);  // browsers may block this until the page has had a click
</script>"""
    components.html(html.replace("__TEXT__", t).replace("__AUTO__", "true" if auto else "false"), height=95)


def get_image(key, label_cam, label_up):
    mode = st.radio("طريقة الإدخال", ["📷 الكاميرا", "🖼️ رفع صورة"], horizontal=True, key=f"mode_{key}")
    f = st.camera_input(label_cam, key=f"cam_{key}") if mode.startswith("📷") else \
        st.file_uploader(label_up, type=["jpg", "jpeg", "png", "webp"], key=f"up_{key}")
    return ImageOps.exif_transpose(Image.open(f)).convert("RGB") if f else None


@st.cache_resource(show_spinner="جارٍ تحميل النموذج…")
def get_model(path, mtime):
    import model_utils as mu
    return mu.load_checkpoint(path)


with st.sidebar:
    st.header("⚙️ الإعدادات")
    auto_speech = st.checkbox("القراءة التلقائية للنتيجة", value=False,
                              help="قد يمنعها المتصفح قبل أن تضغط على الصفحة مرة واحدة؛ زر القراءة يعمل دائمًا.")
    min_conf = st.slider("أدنى ثقة لقبول النتيجة", 0.30, 0.95, float(S["min_confidence"]), 0.05)
    min_margin = st.slider("أدنى فارق عن ثاني أقرب نتيجة", 0.0, 0.5, float(S["min_margin"]), 0.05)
    st.caption("رفع الحدّين يجعل التطبيق يقول «غير متأكد» أكثر، وهذا أفضل من تخمين خاطئ.")

tab_run, tab_teach, tab_about = st.tabs(["♻️ صنّف", "🧠 علّم النموذج", "ℹ️ معلومات"])

with tab_run:
    if not MODEL_PATH.exists():
        st.warning("لا يوجد نموذج مدرَّب بعد (models/waste_model.pt). نفّذ الخطوات في ملف README: "
                   "prepare_data.bat ثم train_model.bat. / No trained model found yet - see README.")
    else:
        try:
            model, names, meta = get_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        except Exception as e:
            st.error(f"تعذّر تحميل النموذج: {e}")
            model = None
        if model is not None:
            img = get_image("run", "التقط صورة للعنصر", "ارفع صورة للعنصر")
            if img is not None:
                import model_utils as mu
                probs = mu.predict(model, img)
                dec = wc.decide(probs, names, min_conf, min_margin, wc.group_map(cfg))
                res = wc.build_result(dec, cfg)
                bin_html = f'<div class="bin">🗑️ الحاوية المقترحة: <b>{res["bin"]}</b></div>' if res["bin"] else ""
                conf_html = ""
                if res["kind"] == "ok":
                    conf_html = (f'<div class="conf">ثقة النموذج: {round(dec["confidence"]*100)}٪ — '
                                 'هذا تقدير من النموذج وليس ضمانًا للصحة.</div>')
                st.markdown(f'<div class="card" style="border-right:10px solid {res["color"]}">'
                            f'<div class="item">{res["title"]}</div>{bin_html}'
                            f'<div class="msg">{res["spoken"]}</div>{conf_html}</div>', unsafe_allow_html=True)
                speech_widget(res["spoken"], auto_speech)
                with st.expander("أقرب ٣ توقعات"):
                    for n, p in dec["top"]:
                        st.write(f"{cfg['classes'].get(n, {}).get('ar_name', n)} — {round(p*100)}٪")

with tab_teach:
    st.write("أضف صورًا حقيقية للعناصر التي ستعرضها (قناني، أكواب، أكياس شيبس…) لتحسين النموذج. "
             "التقط ١٥ صورة على الأقل لكل نوع بزوايا وإضاءة مختلفة، ثم أعد التدريب بتشغيل train_model.bat.")
    options = {c["ar_name"]: k for k, c in cfg["classes"].items()}
    label = st.selectbox("نوع العنصر", list(options))
    timg = get_image("teach", "التقط صورة للتعليم", "ارفع صورة للتعليم")
    if timg is not None and st.button("💾 احفظ الصورة"):
        d = CUSTOM_DIR / options[label]
        d.mkdir(parents=True, exist_ok=True)
        timg.thumbnail((800, 800))
        timg.save(d / f"{datetime.now():%Y%m%d_%H%M%S_%f}.jpg", quality=90)
        st.success("تم الحفظ ✅")
    counts = {k: len(list((CUSTOM_DIR / k).glob("*.jpg"))) if (CUSTOM_DIR / k).is_dir() else 0 for k in cfg["classes"]}
    st.table({"النوع": [cfg["classes"][k]["ar_name"] for k in counts], "عدد الصور": list(counts.values())})
    st.caption("نصيحة: أضف نوع «لا يوجد عنصر» (يد فارغة، طاولة، خلفية الغرفة) حتى لا يخمّن النموذج عند غياب العنصر.")

with tab_about:
    st.write("يعمل التطبيق محليًا على الحاسوب دون إنترنت ودون مفاتيح API. النموذج يتعرف على الشكل المرئي فقط، "
             "ولا يستطيع معرفة مادة العبوة بدقة، ولذلك قد يطلب منك التحقق من الرمز المكتوب عليها.")
    if MODEL_PATH.exists():
        try:
            _, _, meta = get_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
            st.info(f"دقة الاختبار المسجّلة: {meta.get('test_accuracy', 0)*100:.1f}٪ على {meta.get('n_test', '?')} صورة اختبار "
                    f"(تاريخ التدريب: {meta.get('trained_at', '?')}). قد تكون الدقة أقل بكثير على عناصر العرض الحقيقية.")
        except Exception:
            pass

st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
