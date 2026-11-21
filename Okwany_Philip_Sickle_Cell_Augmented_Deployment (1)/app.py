from hashlib import sha256
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import streamlit as st

from inference import analyze_image, load_augmented_model


ROOT = Path(__file__).resolve().parent
SAMPLES = {'Positive field': ROOT / 'positive_example.jpg',
           'Negative field': ROOT / 'negative_example.jpg'}

st.set_page_config(page_title='Sickle-cell image analysis', layout='wide')


@st.cache_resource(show_spinner=False)
def shared_model():
    return load_augmented_model()


st.title('Sickle-cell image analysis')
st.caption('Augmented model · Blood smear research demonstration')

source = st.radio('Image source', ['Upload an image', 'Try a held-out sample'], horizontal=True)
if source == 'Upload an image':
    uploaded = st.file_uploader('Choose a microscope image', type=['jpg', 'jpeg', 'png', 'tif', 'tiff'],
                                max_upload_size=16)
    data = uploaded.getvalue() if uploaded is not None else None
    filename = uploaded.name if uploaded is not None else ''
    reference = None
else:
    choice = st.selectbox('Held-out test image', list(SAMPLES))
    sample = SAMPLES[choice]
    data, filename = sample.read_bytes(), sample.name
    reference = choice.removesuffix(' field')

if not data:
    st.session_state.pop('analysis', None)
    st.stop()

image_key = sha256(data).hexdigest()
if st.button('Analyze image', type='primary'):
    st.session_state.pop('analysis', None)
    try:
        with st.spinner('Loading the model and analyzing your image…'):
            models = shared_model()
            result = analyze_image(data, models)
    except (UnidentifiedImageError, Image.DecompressionBombError, ValueError) as exc:
        st.error(f'Choose a readable microscope image. {exc}')
    except (RuntimeError, OSError) as exc:
        print(f'Application error: {exc}', flush=True)
        st.error('The analysis could not finish. Try again or check the app logs.')
    else:
        st.session_state.analysis = (image_key, result)

current = st.session_state.get('analysis')
if current is None or current[0] != image_key:
    st.session_state.pop('analysis', None)
    st.stop()

result = current[1]
label, score, boxes, masks = st.columns(4)
label.metric('Field prediction', result['prediction'])
score.metric('Detector score', f"{result['score']:.3f}")
boxes.metric('Candidate boxes', result['boxes'])
masks.metric('Nonempty masks', result['masks'])
if reference:
    st.caption(f'Held-out reference label {reference}')

left, right = st.columns(2)
left.image(result['original'], caption='Input image', width='stretch')
right.image(result['overlay'], caption='Gold detection boxes and green segmentation masks', width='stretch')
st.download_button('Download mask PNG', result['mask_png'],
                   file_name=f'{Path(filename).stem}_mask.png', mime='image/png', on_click='ignore')
